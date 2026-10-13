"""Uncertainty proxy #1: gradient stability across independent fresh-window picks.

Under REWINDOW every archived gradient m<k>to<k+1> was measured on its own
independently picked window set -- effectively bootstrap resamples of the data.
Per GLL point we ask: across the N gradients, how consistent is the SIGN of the
beta (Vs) kernel?  frac = max(n_pos, n_neg)/N in [0.5, 1].  High amplitude +
high consistency = data-demanded structure; low amplitude = converged/unsampled
(consistency there is meaningless and is masked).

Caveat: successive gradients see an evolving model, so they are not fully
independent -- where the inversion has converged the gradient amplitude
collapses and sign becomes noise. Read the two panels JOINTLY.

Outputs figures/gradient_stability_<tag>.png for each stack.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "adjoint", "lib"))
from sem_io import read_sem_bin

STACKS = {
    "c1_oldmesh": {
        "xyz": f"{ROOT}/models_band3-18/gradients/band1_mesh_xyz.npy",
        "grads": [f"{ROOT}/models_c1/gradients/m{i}to{i+1}" for i in range(10)],
    },
    "s2_finemesh": {
        "xyz": f"{ROOT}/models_b2_try1/gradients/b2_mesh_xyz.npy",
        "grads": [f"{ROOT}/work/adjoint/m{i}to{i+1}" for i in range(3)],
    },
}
DEPTHS = [3, 8]; HALF = 0.5; WORKERS = 16

for tag, cfg in STACKS.items():
    xyz = np.load(cfg["xyz"])
    G = []
    for g in cfg["grads"]:
        p = os.path.join(g, "proc000000_beta_kernel_clip_smooth.bin")
        if os.path.exists(p):
            G.append(read_sem_bin(p))
    G = np.vstack(G); N = len(G)
    print(f"{tag}: {N} gradients x {G.shape[1]:,} pts")
    mean = G.mean(0)
    frac = np.maximum((G > 0).sum(0), (G < 0).sum(0)) / N

    X = (xyz[0] - xyz[0].min()) / 1e3; Y = (xyz[1] - xyz[1].min()) / 1e3
    Z = xyz[2] / 1e3
    xr = np.linspace(X.min(), X.max(), 200); yr = np.linspace(Y.min(), Y.max(), 360)
    GXg, GYg = np.meshgrid(xr, yr)
    fig, axs = plt.subplots(2, len(DEPTHS), figsize=(4.6*len(DEPTHS), 15),
                            constrained_layout=True)
    for c, dkm in enumerate(DEPTHS):
        m = np.abs(Z + dkm) < HALF
        tree = cKDTree(np.column_stack([X[m], Y[m]]))
        d, i = tree.query(np.column_stack([GXg.ravel(), GYg.ravel()]), k=8, workers=WORKERS)
        w = 1.0/np.maximum(d, 1e-4)**2
        mn = ((w*mean[m][i]).sum(1)/w.sum(1)).reshape(GXg.shape)
        fr = ((w*frac[m][i]).sum(1)/w.sum(1)).reshape(GXg.shape)
        amp_mask = np.abs(mn) < np.percentile(np.abs(mn), 30)   # converged/unsampled
        fr_masked = np.ma.array(fr, mask=amp_mask)
        v = np.percentile(np.abs(mn), 99)
        im0 = axs[0, c].pcolormesh(GXg, GYg, mn, cmap="RdBu", vmin=-v, vmax=v, shading="auto")
        axs[0, c].set_title(f"mean beta kernel  z={dkm} km"); axs[0, c].set_aspect("equal")
        im1 = axs[1, c].pcolormesh(GXg, GYg, fr_masked, cmap="Blues", vmin=0.5, vmax=1.0,
                                   shading="auto")
        axs[1, c].set_facecolor("#dddbd4")
        axs[1, c].set_title(f"sign consistency (N={N})  z={dkm} km"); axs[1, c].set_aspect("equal")
    fig.colorbar(im0, ax=axs[0, :].tolist(), shrink=0.7, label="mean dchi/dlnVs")
    fig.colorbar(im1, ax=axs[1, :].tolist(), shrink=0.7,
                 label="sign agreement fraction (gray = low-amplitude mask)")
    fig.suptitle(f"Gradient stability across {N} independent window picks — {tag}", fontsize=13)
    out = f"{ROOT}/figures/gradient_stability_{tag}.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    hi = (frac[np.abs(mean) >= np.percentile(np.abs(mean), 70)]).mean()
    print(f"  mean sign-consistency in top-30% amplitude regions: {hi:.3f}  -> {out}")
