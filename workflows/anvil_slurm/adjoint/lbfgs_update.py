#!/usr/bin/env python3
"""Apply a TRUE L-BFGS model update, in pure Python (shared partition, no MPI).

Replaces xadd_model_iso for the isotropic (vp,vs,rho) inversion. The model
update is purely pointwise (m_new = m*exp(perturbation) per GLL value), so it
needs no mesh and no MPI -- unlike xadd_model_iso, which runs 1250 MPI ranks
only to do an element-wise exp() plus diagnostic volume integrals.

Why "true" L-BFGS
-----------------
xadd_model_iso normalizes the search direction by its max and rescales to
step_fac, which DISCARDS the quasi-Newton step length (so you must line-search
step_fac every iteration). Here we keep the L-BFGS natural step:

    g   = smoothed gradient  d(misfit)/d(ln m)      (alpha->vp, beta->vs)
    r   = H_k g              (two-loop, lbfgs_direction.compute_direction)
    d   = -r                 (descent direction)
    a   = min(1, max_pert / max|d|)                  (full L-BFGS step, capped)
    vp_new  = vp  * exp(a * d_alpha)
    vs_new  = vs  * exp(a * d_beta)
    rho_new = rho * exp(a * RHO_SCALING * d_beta)    (RHO_SCALING = 0.33)

For iterations >= 2 the L-BFGS direction is well-scaled, so a = 1 (the cap is
not hit) and NO line search is needed -- just monitor misfit. The cap is a
trust-region safeguard that mainly engages on iteration 1 (steepest descent,
empty history) and on any over-large step.

Outputs (drop-in for xadd_model_iso, plain names -- NO _new suffix):
    <out>/proc*_vp.bin  proc*_vs.bin  proc*_rho.bin
so the result can be used directly as the next iteration's INPUT_MODEL.

Depth-band cap (documented hook, off by default)
-------------------------------------------------
xadd uses USE_DEPTH_RANGE_MAXIMUM (z in -5000..-500 m) when computing its max,
so near-surface spikes don't shrink the step. With clipping (clip ~1e-5) the
spikes are already removed, so the default here is a GLOBAL cap (no mesh / no z
needed). To reproduce the depth band, pass --depth-band Z_BOTTOM Z_TOP together
with --z-dir DIR containing proc{rank:06d}_z.npy (float32, same per-rank sizes
as the model); the cap's max|d| is then restricted to that z range. (z is not
extracted here; build proc*_z.npy once from external_mesh.bin if you want this.)

Usage
-----
  python lbfgs_update.py --nproc N --grad KERNELS/smooth_sum \\
        --model INPUT_MODEL --out OUTPUT_MODEL --state LBFGS_STATE \\
        [--m 5] [--max-pert 0.03] [--params alpha,beta] [--tag iter02] \\
        [--depth-band -5000 -500 --z-dir Z_COORDS]
"""
import argparse
import os
import sys
import numpy as np

# Local modules live in ./lib next to this script. Resolve via __file__ (NOT cwd)
# so the script imports correctly no matter where a (chained/automated) job runs it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import lbfgs_direction as lb   # shared, validated two-loop + SEM I/O + state

RHO_SCALING = 0.33             # matches USE_RHO_SCALING in constants_tomography.h
MODEL_OF = {"alpha": "vp", "beta": "vs"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, required=True)
    ap.add_argument("--grad", required=True, help="current smoothed-gradient dir")
    ap.add_argument("--model", required=True, help="current model dir (vp,vs,rho)")
    ap.add_argument("--out", required=True, help="output model dir (vp,vs,rho)")
    ap.add_argument("--state", required=True, help="L-BFGS state dir (persistent)")
    ap.add_argument("--m", type=int, default=5, help="L-BFGS history length")
    ap.add_argument("--max-pert", type=float, default=0.03,
                    help="max |ln-model change| at any point (trust-region cap)")
    ap.add_argument("--no-clamp", action="store_true",
                    help="line-search mode: a = max_pert/max|d| (drop the min(1,.) "
                         "quasi-Newton cap) so the largest |ln-model change| is "
                         "EXACTLY max_pert -- makes max_pert the true step-length "
                         "knob and stops distinct step values collapsing to a=1.")
    ap.add_argument("--params", default="alpha,beta")
    ap.add_argument("--tag", default="", help="iteration label for the log")
    ap.add_argument("--depth-band", type=float, nargs=2, metavar=("ZBOT", "ZTOP"),
                    default=None, help="restrict the cap's max|d| to ZBOT<z<ZTOP")
    ap.add_argument("--z-dir", default=None,
                    help="dir with proc{rank:06d}_z.npy (required by --depth-band)")
    args = ap.parse_args()

    params = [p.strip() for p in args.params.split(",") if p.strip()]
    if params != ["alpha", "beta"]:
        sys.exit("this updater expects --params alpha,beta (iso vp/vs + scaled rho)")

    # ---- L-BFGS direction r = H_k g_k (also appends/prunes the state) ----
    r, sizes = lb.compute_direction(args.nproc, args.grad, args.model,
                                    args.state, args.m, params, args.tag)
    d = -r                                  # descent direction
    per_param = int(sizes.sum())
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    # ---- trust-region step length a = min(1, max_pert / max|d|) ----
    if args.depth_band is not None:
        if not args.z_dir:
            sys.exit("--depth-band requires --z-dir with proc*_z.npy")
        zbot, ztop = sorted(args.depth_band)          # zbot < ztop, both negative
        zcat = np.concatenate([np.load(os.path.join(
            args.z_dir, f"proc{rk:06d}_z.npy")).astype(np.float64)
            for rk in range(args.nproc)])              # one block (per-param) of z
        mask1 = (zcat > zbot) & (zcat < ztop)          # depth band, one param block
        mask = np.concatenate([mask1, mask1])          # alpha + beta blocks
        if not mask.any():
            sys.exit(f"--depth-band {zbot}..{ztop}: no points in range")
        dmax = float(np.max(np.abs(d[mask])))
        print(f"  cap: depth-band {zbot}..{ztop} m, {int(mask1.sum())} pts/param")
    else:
        dmax = float(np.max(np.abs(d)))
        print("  cap: global max|d|")

    if dmax <= 0.0:
        sys.exit("max|d| = 0; nothing to update")
    if args.no_clamp:
        a = args.max_pert / dmax            # line search: |Δlnm|max == max_pert exactly
        print(f"  max|d|={dmax:.4e}  max_pert={args.max_pert}  step a={a:.4f}"
              f"  (NO-CLAMP line-search: |Δlnm|max=max_pert)")
    else:
        a = min(1.0, args.max_pert / dmax)
        capped = a < 1.0
        print(f"  max|d|={dmax:.4e}  max_pert={args.max_pert}  step a={a:.4f}"
              f"  ({'CAPPED (SD-like)' if capped else 'full L-BFGS step a=1'})")

    # ---- apply pointwise: m_new = m * exp(a * d_param) ; write plain names ----
    os.makedirs(args.out, exist_ok=True)
    d_alpha = d[:per_param]
    d_beta = d[per_param:2 * per_param]
    gmin = {"vp": np.inf, "vs": np.inf, "rho": np.inf}
    gmax = {"vp": -np.inf, "vs": -np.inf, "rho": -np.inf}
    for rk in range(args.nproc):
        i0, i1 = int(offs[rk]), int(offs[rk + 1])
        da = d_alpha[i0:i1]
        db = d_beta[i0:i1]
        upd = {
            "vp":  np.exp(a * da),
            "vs":  np.exp(a * db),
            "rho": np.exp(a * RHO_SCALING * db),
        }
        for mf, fac in upd.items():
            m_old = lb.read_sem(os.path.join(args.model, f"proc{rk:06d}_{mf}.bin"))
            m_new = (m_old.astype(np.float64) * fac).astype("<f4")
            lb.write_sem(os.path.join(args.out, f"proc{rk:06d}_{mf}.bin"), m_new)
            gmin[mf] = min(gmin[mf], float(m_new.min()))
            gmax[mf] = max(gmax[mf], float(m_new.max()))

    print(f"  wrote new model -> {args.out} (proc*_{{vp,vs,rho}}.bin, {args.nproc} ranks)")
    for mf in ("vp", "vs", "rho"):
        print(f"    {mf}_new min/max: {gmin[mf]:.4g} / {gmax[mf]:.4g}")
    if min(gmin["vp"], gmin["vs"], gmin["rho"]) <= 0.0:
        print("  WARNING: non-positive value in new model -- check step/max_pert", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
