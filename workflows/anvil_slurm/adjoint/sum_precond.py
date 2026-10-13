#!/usr/bin/env python3
"""Pure-Python replacement for `xsum_preconditioned_kernels`.

Reproduces sum_preconditioned_kernels.f90 exactly for this build's settings
(USE_ALPHA_BETA_RHO=.true., USE_HESS_SUM=.true., USE_SOURCE_MASK=.false.,
THRESHOLD_HESS=1e-3, REG=''). Per GLL point and per proc:

    total_hess   = sum_events |hess_kernel|                 # >= 0
    maxh         = max(total_hess) over ALL procs           # the ONLY global term
    h            = total_hess / maxh        (or all-ones if maxh < 1e-18)
    invH         = where(h > THRESHOLD, 1/h, 1/THRESHOLD)
    total_kernel = (sum_events <name>_clip) * invH          # name in alpha/beta/rho

Writes proc*_{alpha,beta,rho}_kernel_clip.bin and proc*_hess_inv_kernel.bin
(= invH, same side-output the Fortran writes) directly into <out> (default
KERNELS/condi_sum) -- so this also skips the OUTPUT_SUM -> condi_sum mv that the
old MPI sum job did. The smooth step (step 7) then reads KERNELS/condi_sum.

The summed Hessian is identical for all three fields, so it is computed once in a
first pass (which also yields the global max via a one-float reduction), cached to
--tmpdir, then applied in a second pass. No MPI: parallel over procs with a pool.

Usage:
  python3 sum_precond.py [--input KERNELS] [--out KERNELS/condi_sum]
                         [--nproc 1250] [--jobs N] [--threshold 1e-3]
                         [--tmpdir <scratch>] [--keep-tmp]
"""
import argparse
import os
import shutil
import sys
import tempfile
from functools import partial
from multiprocessing import Pool

import numpy as np

# Local modules live in ./lib next to this script. Resolve via __file__ (NOT cwd)
# so the script imports correctly no matter where a (chained/automated) job runs it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from sem_io import read_sem_bin, write_sem_bin

FIELDS = ("alpha_kernel_clip", "beta_kernel_clip", "rho_kernel_clip")
HESS_INV_NAME = "hess_inv_kernel"
EXCLUDE_PREFIXES = ("condi_sum", "smooth_sum")


def list_events(input_dir):
    evs = []
    for d in sorted(os.listdir(input_dir)):
        if not os.path.isdir(os.path.join(input_dir, d)):
            continue
        if d == "smooth_sum" or d.startswith("condi_sum"):
            continue
        evs.append(d)
    return evs


def _sum_hess(p, hess_dir, events, tmpdir):
    """Pass 1: total_hess = sum_e |hess|; cache it; return its max."""
    total = None
    for ev in events:
        path = os.path.join(hess_dir, ev, f"proc{p:06d}_hess_kernel.bin")
        h = np.abs(read_sem_bin(path))
        total = h if total is None else total + h
    np.save(os.path.join(tmpdir, f"hess_{p:06d}.npy"), total)
    return float(total.max())


def _apply(p, input_dir, out_dir, events, tmpdir, threshold, maxh):
    """Pass 2: build invH from cached total_hess and global maxh, apply, write."""
    total_hess = np.load(os.path.join(tmpdir, f"hess_{p:06d}.npy"))

    # Keep float32 (CUSTOM_REAL) throughout to mirror invert_hess arithmetic.
    maxh = np.float32(maxh)
    threshold = np.float32(threshold)
    if maxh < np.float32(1.e-18):
        # invert_hess: Hessian ~ 0 -> reset to ones, then 1/1 = 1 everywhere.
        invH = np.ones_like(total_hess)
    else:
        h = (total_hess / maxh).astype(np.float32)
        invH = np.where(h > threshold,
                        (np.float32(1.0) / h),
                        (np.float32(1.0) / threshold)).astype(np.float32)

    for name in FIELDS:
        tot = None
        for ev in events:
            path = os.path.join(input_dir, ev, f"proc{p:06d}_{name}.bin")
            v = read_sem_bin(path)
            tot = v if tot is None else tot + v
        tot = tot * invH
        write_sem_bin(os.path.join(out_dir, f"proc{p:06d}_{name}.bin"), tot)

    write_sem_bin(os.path.join(out_dir, f"proc{p:06d}_{HESS_INV_NAME}.bin"), invH)
    return p


def main():
    ap = argparse.ArgumentParser(description="Python xsum_preconditioned_kernels")
    ap.add_argument("--input", default="KERNELS",
                    help="dir of per-event subdirs holding the *_clip.bin "
                         "(= INPUT_KERNELS)")
    ap.add_argument("--hess-input", default=None,
                    help="dir holding the un-clipped proc*_hess_kernel.bin "
                         "per event (default: same as --input). Point this at "
                         "ORI_KERNELS to skip the hess-linking step.")
    ap.add_argument("--out", default=os.path.join("KERNELS", "condi_sum"))
    ap.add_argument("--nproc", type=int, default=1250)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK",
                                               os.cpu_count() or 1)))
    ap.add_argument("--threshold", type=float, default=1.e-3,
                    help="THRESHOLD_HESS (constants_tomography.h)")
    ap.add_argument("--tmpdir", default=None,
                    help="scratch dir for cached summed Hessian "
                         "(default: a temp dir next to --out)")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    hess_input = args.hess_input or args.input
    events = list_events(args.input)
    if not events:
        sys.exit(f"no event subdirs found under {args.input}")
    os.makedirs(args.out, exist_ok=True)

    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix="sumhess_",
                                             dir=os.path.dirname(args.out) or ".")
    os.makedirs(tmpdir, exist_ok=True)

    print(f"sum: events={len(events)}  nproc={args.nproc}  jobs={args.jobs}  "
          f"threshold={args.threshold:g}", flush=True)
    print(f"input={args.input}  hess_input={hess_input}  "
          f"out={args.out}  tmp={tmpdir}", flush=True)

    procs = list(range(args.nproc))
    try:
        # Pass 1: summed |Hessian| per proc + global max (one-float reduction).
        f1 = partial(_sum_hess, hess_dir=hess_input, events=events,
                     tmpdir=tmpdir)
        with Pool(processes=args.jobs) as pool:
            local_max = pool.map(f1, procs, chunksize=8)
        maxh = max(local_max)
        print(f"pass1 done: global Hessian max = {maxh:.6e}", flush=True)

        # Pass 2: precondition + sum each field, write outputs.
        f2 = partial(_apply, input_dir=args.input, out_dir=args.out,
                     events=events, tmpdir=tmpdir, threshold=args.threshold,
                     maxh=maxh)
        done = 0
        with Pool(processes=args.jobs) as pool:
            for _ in pool.imap_unordered(f2, procs, chunksize=8):
                done += 1
                if done % 250 == 0:
                    print(f"  pass2 {done}/{args.nproc} procs", flush=True)
        print(f"sum done: {done} procs written to {args.out}", flush=True)
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
