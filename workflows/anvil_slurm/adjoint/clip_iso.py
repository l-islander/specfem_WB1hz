#!/usr/bin/env python3
"""Pure-Python replacement for `xclip_sem` (vertical-only iso workflow), with an
optional per-event receiver taper.

Clips the alpha/beta/rho event kernels to [-cv, cv] per GLL point and writes
proc*_<name>_clip.bin into <out>/<event>/. The clip is an exact per-point clamp
(clip_sem.f90: np.clip(v, -cv, cv)).

With --mask, a per-event receiver taper is applied AFTER the clip, before the
files are written -- this is the analogue of SPECFEM's source mask
(locate_sources.f90 calc_mask_source, applied per-event in
sum_preconditioned_kernels.f90), but tapering around the *receivers* (the
adjoint-source locations), which is where the near-receiver kernel singularity
sits. The mask is a multiplicative anisotropic Gaussian notch built per (event,
proc) from each event's output_list_stations.txt; see receiver_mask.py. Masking
is amplitude-independent (0 at the receiver -> 1 away), so it suppresses the
imprint regardless of its height; applying it per-event before the sum prevents
the same-sign near-receiver lobes of reused stations (e.g. UO.RAID, 19 events)
from accumulating.

The work unit is one proc (the mesh is read once per proc, then reused across
all events); kernels are regenerated from --ori on every run, so masked output
is idempotent (no double-masking) and re-runnable without backing anything up.

Usage:
  python3 clip_iso.py --clip 1E-5 [--ori ORI_KERNELS] [--out KERNELS]
                      [--nproc 1250] [--jobs N]
                      [--mask --topo-dir DATABASES_MPI
                       --sigma-h 1700 --sigma-v 500 [--n-sigma 4]]
"""
import argparse
import glob
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np

# Local modules live in ./lib next to this script. Resolve via __file__ (NOT cwd)
# so the script imports correctly no matter where a (chained/automated) job runs it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from sem_io import read_sem_bin, write_sem_bin

DEFAULT_NAMES = ("alpha_kernel", "beta_kernel", "rho_kernel")


def list_events(ori):
    return sorted(d for d in os.listdir(ori)
                  if os.path.isdir(os.path.join(ori, d)))


def find_stations_file(event):
    """Locate an event's receiver list (mesh coords) across the group_* runs."""
    hits = glob.glob(
        f"group_*/RUN/{event}/OUTPUT_FILES/output_list_stations.txt")
    if not hits:
        raise FileNotFoundError(
            f"no output_list_stations.txt for event {event}")
    return hits[0]


_COORDS = {}   # per-worker cache: proc -> (xg, yg, zg); loaded once per proc


def _process_task(task, ori, out, names, cv, mask_cfg):
    """Clip (and optionally mask) all names for ONE (proc, event) pair.

    PARALLEL OVER (proc, event), NOT over proc alone. The original loop
    parallelised over mesh partitions -- right on Anvil (1250 procs), but at
    NPROC=1 that is a single task, so one worker did all 150 events serially
    (~70 min) while the rest idled (bit 2026-08-29). Coordinates are loaded
    once per worker per proc and cached in the module global.
    """
    p, ev = task
    m = None
    if mask_cfg is not None:
        if p not in _COORDS:
            from mesh_coords import read_proc_coords
            _COORDS[p] = read_proc_coords(mask_cfg["topo"], p, cache=False)
        from receiver_mask import build_mask
        xg, yg, zg = _COORDS[p]
        m = build_mask(xg, yg, zg, mask_cfg["rec"][ev],
                       mask_cfg["sigma_h"], mask_cfg["sigma_v"],
                       mask_cfg["n_sigma"])
    out_ev = os.path.join(out, ev)
    nfiles = 0
    for name in names:
        arr = read_sem_bin(os.path.join(ori, ev, f"proc{p:06d}_{name}.bin"))
        np.clip(arr, -cv, cv, out=arr)
        if m is not None:
            arr = arr * m
        write_sem_bin(os.path.join(out_ev, f"proc{p:06d}_{name}_clip.bin"), arr)
        nfiles += 1
    return nfiles


def _mem_cap_jobs(requested):
    """Each worker holds coords + float64 mask temporaries (~4.5 GB at this
    mesh); cap the pool so it cannot swap the box."""
    try:
        avail_kb = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1])
        cap = max(1, int(avail_kb / 1024 / 1024 / 4.5))
    except Exception:
        cap = requested
    return max(1, min(requested, cap))


def main():
    ap = argparse.ArgumentParser(description="Python xclip_sem (+receiver mask)")
    ap.add_argument("--clip", required=True,
                    help="clip magnitude cv; clamps to [-cv, cv] (e.g. 1E-5)")
    ap.add_argument("--ori", default="ORI_KERNELS")
    ap.add_argument("--out", default="KERNELS")
    ap.add_argument("--names", default=",".join(DEFAULT_NAMES),
                    help="comma-separated kernel names (no _clip suffix)")
    ap.add_argument("--nproc", type=int, default=1250)
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK",
                                               os.cpu_count() or 1)))
    # receiver taper (off unless --mask)
    ap.add_argument("--mask", action="store_true",
                    help="apply per-event receiver taper after clipping")
    ap.add_argument("--topo-dir", default="DATABASES_MPI",
                    help="dir with proc*_external_mesh.bin (for GLL coords)")
    ap.add_argument("--sigma-h", type=float, default=1700.0,
                    help="horizontal taper sigma (m); ~1 element edge")
    ap.add_argument("--sigma-v", type=float, default=500.0,
                    help="vertical taper sigma (m); ~1 element edge")
    ap.add_argument("--n-sigma", type=float, default=4.0,
                    help="receiver cull radius in horizontal sigmas")
    args = ap.parse_args()

    cv = float(args.clip)
    names = tuple(s for s in args.names.split(",") if s)
    events = list_events(args.ori)
    if not events:
        sys.exit(f"no event subdirs found under {args.ori}")

    # output event dirs (created up front so workers never race on mkdir)
    for ev in events:
        os.makedirs(os.path.join(args.out, ev), exist_ok=True)

    mask_cfg = None
    if args.mask:
        from receiver_mask import read_station_coords
        rec = {ev: read_station_coords(find_stations_file(ev))
               for ev in events}
        nrec = {ev: r.shape[0] for ev, r in rec.items()}
        mask_cfg = dict(topo=args.topo_dir, rec=rec,
                        sigma_h=args.sigma_h, sigma_v=args.sigma_v,
                        n_sigma=args.n_sigma)
        print(f"receiver taper ON: sigma_h={args.sigma_h:g} "
              f"sigma_v={args.sigma_v:g} n_sigma={args.n_sigma:g} "
              f"topo={args.topo_dir}", flush=True)
        print(f"  receivers/event: min={min(nrec.values())} "
              f"max={max(nrec.values())} "
              f"total={sum(nrec.values())}", flush=True)

    print(f"clip: cv=+/-{cv:g}  names={names}  nproc={args.nproc}  "
          f"events={len(events)}  jobs={args.jobs}  mask={args.mask}",
          flush=True)

    f = partial(_process_task, ori=args.ori, out=args.out, names=names,
                cv=cv, mask_cfg=mask_cfg)
    tasks = [(p, ev) for p in range(args.nproc) for ev in events]
    jobs = _mem_cap_jobs(min(args.jobs, len(tasks)))
    print(f"  pool: {jobs} workers over {len(tasks)} (proc,event) tasks", flush=True)
    done = nfiles = 0
    with Pool(processes=jobs) as pool:
        for n in pool.imap_unordered(f, tasks, chunksize=1):
            done += 1
            nfiles += n
            if done % 25 == 0:
                print(f"  {done}/{len(tasks)} tasks ({nfiles} files)",
                      flush=True)
    print(f"clip done: {done} tasks, {nfiles} files written", flush=True)


if __name__ == "__main__":
    sys.exit(main())
