#!/usr/bin/env python3
"""Workflow-level L-BFGS search direction for the isotropic (alpha,beta)
inversion, keeping your custom (external) adjoint sources untouched.

Why this exists
---------------
specfem3d-4.1.1_wb has no isotropic quasi-Newton update tool (xadd_model_iso /
xmodel_update do steepest descent only). The integrated xinverse_problem_for_model
framework has L-BFGS but insists on computing its own misfit/adjoint sources,
which would discard your multi-band, zero-horizontal external measurement
pipeline. So we run the L-BFGS two-loop recursion here and feed the result to
the existing xadd_model_iso, exactly like old_script/old_cg_direction.py.

Spaces / sign conventions
-------------------------
- Gradient g_k  : the smoothed kernels proc*_{alpha,beta}_kernel_clip_smooth.bin.
                  These are d(misfit)/d(ln vp), d(misfit)/d(ln vs)  (relative,
                  Hessian-preconditioned in the sum step already).
- Model   x_k   : ln(vp), ln(vs) read from INPUT_MODEL (proc*_vp.bin, proc*_vs.bin).
                  alpha<->vp, beta<->vs.
- L-BFGS pairs  : s_j = x_{j+1}-x_j,  y_j = g_{j+1}-g_j, kept for the last m
                  iterations; pairs with <s,y> <= 0 are skipped (curvature
                  safeguard).
- Two-loop      : r ~= H_k g_k  with H0 = gamma*I, gamma = <s_last,y_last>/<y_last,y_last>.
                  L-BFGS descent direction d_k = -r.
- xadd_model_iso descends along -(field) it reads, so we write field = -d_k = r.
  Iteration 1 (no history): r = g_k -> identical to steepest descent, and the
  snapshot seeds the history for iteration 2.

State layout (LBFGS_STATE/)
---------------------------
  index.txt                 ordered "<counter> <iteration_tag>" lines
  snap_<counter>/x.npy      concatenated ln-model   [alpha-part | beta-part]
  snap_<counter>/g.npy      concatenated gradient   [alpha-part | beta-part]
  snap_<counter>/sizes.npy  per-rank element counts (to split back to SEM)
Only the last (m+1) snapshots are kept.

Usage
-----
  python lbfgs_direction.py --nproc N --grad KERNELS/smooth_sum \\
        --model INPUT_MODEL --out INPUT_GRADIENT_LBFGS --state LBFGS_STATE \\
        --m 5 [--params alpha,beta] [--tag iter02]
"""
import argparse
import os
import sys
import numpy as np

SUFFIX = "_kernel_clip_smooth"
MODEL_OF = {"alpha": "vp", "beta": "vs"}   # gradient param -> model file name


def read_sem(path):
    """Read a Fortran sequential-unformatted single-record float32 array."""
    with open(path, "rb") as f:
        raw = f.read()
    n1 = np.frombuffer(raw[:4], dtype="<i4")[0]
    n2 = np.frombuffer(raw[4 + n1:8 + n1], dtype="<i4")[0]
    if n1 != n2:
        raise ValueError(f"record markers differ ({n1} != {n2}) in {path}")
    return np.frombuffer(raw[4:4 + n1], dtype="<f4").copy()


def write_sem(path, arr):
    """Write a float32 array as a Fortran sequential-unformatted record."""
    arr = np.asarray(arr, dtype="<f4")
    marker = np.array([arr.nbytes], dtype="<i4")
    with open(path, "wb") as f:
        marker.tofile(f)
        arr.tofile(f)
        marker.tofile(f)


def fname(d, rank, param):
    return os.path.join(d, f"proc{rank:06d}_{param}{SUFFIX}.bin")


def load_concat_grad(grad_dir, nproc, params):
    """Concatenate proc*_<param>_kernel_clip_smooth.bin over ranks & params.
    Returns (vector float64, per-rank sizes for the FIRST param)."""
    parts = []
    sizes = []
    for p in params:
        for rank in range(nproc):
            a = read_sem(fname(grad_dir, rank, p))
            parts.append(a)
            if p == params[0]:
                sizes.append(a.size)
    return np.concatenate(parts).astype(np.float64), np.array(sizes, dtype=np.int64)


def load_concat_logmodel(model_dir, nproc, params):
    """Concatenate ln(model) over ranks & params (alpha->vp, beta->vs)."""
    parts = []
    for p in params:
        mf = MODEL_OF[p]
        for rank in range(nproc):
            a = read_sem(os.path.join(model_dir, f"proc{rank:06d}_{mf}.bin"))
            a = np.asarray(a, dtype=np.float64)
            if np.any(a <= 0):
                raise ValueError(f"non-positive model value in proc{rank:06d}_{mf}.bin")
            parts.append(np.log(a))
    return np.concatenate(parts)


def write_split(vec, out_dir, sizes, nproc, params):
    """Write a concatenated vector back to per-rank SEM kernel files."""
    os.makedirs(out_dir, exist_ok=True)
    off = 0
    per_param = sizes.sum()
    for ip, p in enumerate(params):
        base = ip * per_param
        off = 0
        for rank in range(nproc):
            n = int(sizes[rank])
            write_sem(fname(out_dir, rank, p), vec[base + off: base + off + n])
            off += n


def load_index(state):
    idx = os.path.join(state, "index.txt")
    if not os.path.exists(idx):
        return []
    out = []
    with open(idx) as f:
        for line in f:
            line = line.strip()
            if line:
                c, *tag = line.split()
                out.append((int(c), tag[0] if tag else ""))
    return sorted(out)


def save_index(state, entries):
    with open(os.path.join(state, "index.txt"), "w") as f:
        for c, tag in entries:
            f.write(f"{c} {tag}\n")


def compute_direction(nproc, grad_dir, model_dir, state, m, params, tag=""):
    """Run the L-BFGS two-loop recursion and update the persistent state.

    Loads the current gradient g_k (from grad_dir) and ln-model x_k (from
    model_dir), builds (s,y) pairs from the stored snapshots plus the current
    point, runs the recursion, then appends the current snapshot to `state` and
    prunes to the last (m+1). Empty history -> r = g_k (steepest descent).

    Returns (r, sizes): r = H_k g_k (so the L-BFGS descent direction is d = -r);
    sizes = per-rank element counts for ONE param (to split r back to SEM files).
    Both lbfgs_direction.py (writes -d_k = r as a field) and lbfgs_update.py
    (applies the model update) call this, so the math lives in one place.
    """
    os.makedirs(state, exist_ok=True)

    # ---- current point ----
    g_k, sizes = load_concat_grad(grad_dir, nproc, params)
    x_k = load_concat_logmodel(model_dir, nproc, params)
    if x_k.size != g_k.size:
        raise ValueError(f"model/grad vector size mismatch ({x_k.size} vs {g_k.size})")

    # ---- load stored snapshots (oldest -> newest) ----
    entries = load_index(state)
    snaps = []  # (counter, x, g)
    for c, _ in entries:
        d = os.path.join(state, f"snap_{c:04d}")
        if os.path.exists(os.path.join(d, "x.npy")):
            snaps.append((c, np.load(os.path.join(d, "x.npy")),
                          np.load(os.path.join(d, "g.npy"))))

    # ---- build (s,y) pairs from consecutive points incl. current (newest) ----
    points = [(x, g) for (_, x, g) in snaps] + [(x_k, g_k)]
    pairs = []  # list of dicts: s, y, rho
    for j in range(len(points) - 1):
        s = points[j + 1][0] - points[j][0]
        y = points[j + 1][1] - points[j][1]
        sy = float(np.dot(s, y))
        if sy > 0.0:
            pairs.append({"s": s, "y": y, "rho": 1.0 / sy})
        else:
            print(f"  skip pair {j}: <s,y>={sy:.3e} <= 0 (non-positive curvature)")
    pairs = pairs[-m:]  # keep most recent m

    # ---- two-loop recursion: r ~= H_k g_k ----
    q = g_k.copy()
    if not pairs:
        print(f"  no usable history -> steepest descent (tag={tag or 'n/a'})")
        r = q
    else:
        alphas = [0.0] * len(pairs)
        for i in range(len(pairs) - 1, -1, -1):     # newest -> oldest
            a = pairs[i]["rho"] * float(np.dot(pairs[i]["s"], q))
            alphas[i] = a
            q -= a * pairs[i]["y"]
        last = pairs[-1]
        gamma = float(np.dot(last["s"], last["y"]) / np.dot(last["y"], last["y"]))
        print(f"  L-BFGS: {len(pairs)} pair(s), gamma(H0)={gamma:.6e} "
              f"(tag={tag or 'n/a'})")
        r = gamma * q
        for i in range(len(pairs)):                  # oldest -> newest
            b = pairs[i]["rho"] * float(np.dot(pairs[i]["y"], r))
            r += pairs[i]["s"] * (alphas[i] - b)

    # ---- append current snapshot, prune to last (m+1) ----
    next_c = (entries[-1][0] + 1) if entries else 0
    d = os.path.join(state, f"snap_{next_c:04d}")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "x.npy"), x_k.astype(np.float32))
    np.save(os.path.join(d, "g.npy"), g_k.astype(np.float32))
    np.save(os.path.join(d, "sizes.npy"), sizes)
    entries.append((next_c, tag or f"snap{next_c}"))

    keep = entries[-(m + 1):]
    for c, _ in entries[:-(m + 1)]:
        old = os.path.join(state, f"snap_{c:04d}")
        if os.path.isdir(old):
            import shutil
            shutil.rmtree(old)
    save_index(state, keep)
    print(f"  state updated: {len(keep)} snapshot(s) kept in {state}")

    return r, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, required=True)
    ap.add_argument("--grad", required=True, help="current smoothed-gradient dir")
    ap.add_argument("--model", required=True, help="current model dir (INPUT_MODEL)")
    ap.add_argument("--out", required=True, help="output dir for field = -d_k")
    ap.add_argument("--state", required=True, help="L-BFGS state dir")
    ap.add_argument("--m", type=int, default=5, help="L-BFGS history length")
    ap.add_argument("--params", default="alpha,beta")
    ap.add_argument("--tag", default="", help="optional iteration tag for the log")
    args = ap.parse_args()

    params = [p.strip() for p in args.params.split(",") if p.strip()]
    r, sizes = compute_direction(args.nproc, args.grad, args.model,
                                 args.state, args.m, params, args.tag)

    # field = -d_k = r  (xadd negates -> descends d_k = -r)
    write_split(r.astype(np.float32), args.out, sizes, args.nproc, params)
    print(f"  wrote field for {len(params)} param(s) x {args.nproc} ranks -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
