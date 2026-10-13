"""How far is (cross-basis) REWINDOW L-BFGS from steepest descent?

For each stage-2 update k (gradient m<k>to<k+1>), rebuild the L-BFGS state as
it would have been (snapshots x_j = ln M_j, g_j = gradient m<j>to<j+1> for
j<k), run the two-loop recursion on g_k, and compare the resulting direction
with plain steepest descent -g_k:

    cos(d_LBFGS, d_SD)  over the full (alpha|beta) vector and beta-only.

Because the loop normalizes every candidate to max|dln m| = alpha (line-search
max-pert mode), direction SHAPE is the only thing L-BFGS changes -- the cosine
is the operative distance. Also reports gamma (H0 scaling) and skipped pairs.

float32 throughout; peak RAM ~6 GB at k=4.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "adjoint", "lib"))
from lbfgs_direction import read_sem

def grad_vec(k):
    d = f"{ROOT}/work/adjoint/m{k}to{k+1}"
    return np.concatenate([read_sem(f"{d}/proc000000_{p}_kernel_clip_smooth.bin")
                           for p in ("alpha", "beta")]).astype(np.float32)

def model_vec(k):
    d = f"{ROOT}/models/M{k}"
    return np.concatenate([np.log(read_sem(f"{d}/proc000000_{p}.bin"))
                           for p in ("vp", "vs")]).astype(np.float32)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 4
half = None
print("k  pairs(kept/all)  gamma      cos_full  cos_beta   (1=identical direction)")
for k in range(1, KMAX + 1):
    g_k = grad_vec(k)
    half = g_k.size // 2
    pairs = []
    for j in range(k):
        s = model_vec(j + 1) - model_vec(j)
        y = grad_vec(j + 1) - grad_vec(j)
        sy = float(np.dot(s.astype(np.float64), y.astype(np.float64)))
        if sy > 0:
            pairs.append((s, y, 1.0 / sy))
        del s, y
    kept = len(pairs)
    # two-loop
    q = g_k.astype(np.float64).copy()
    al = []
    for s, y, rho in reversed(pairs):
        a = rho * float(np.dot(s.astype(np.float64), q))
        al.append(a); q -= a * y.astype(np.float64)
    if pairs:
        s_l, y_l, _ = pairs[-1]
        gamma = float(np.dot(s_l.astype(np.float64), y_l.astype(np.float64)) /
                      np.dot(y_l.astype(np.float64), y_l.astype(np.float64)))
    else:
        gamma = 1.0
    q *= gamma
    for (s, y, rho), a in zip(pairs, reversed(al)):
        b = rho * float(np.dot(y.astype(np.float64), q))
        q += (a - b) * s.astype(np.float64)
    d_lb = -q.astype(np.float32)          # L-BFGS direction
    d_sd = -g_k                           # steepest descent
    cf = cosine(d_lb, d_sd)
    cb = cosine(d_lb[half:], d_sd[half:])
    print(f"{k}  {kept}/{k}              {gamma:9.3e}  {cf:8.4f}  {cb:8.4f}")
    del pairs, q, d_lb, d_sd, g_k
