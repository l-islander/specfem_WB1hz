"""Uncertainty proxy #2: reciprocal-pair dt scatter = direct data-error estimate.

Ambient-noise virtual sources are also receivers, so the measured set contains
reciprocal pairs: dt(A->B) from source dir A_Z, and dt(B->A) from B_Z. Both
sample the SAME path, so their dt should agree; the spread of (dt_AB - dt_BA)
is twice the per-measurement error (assuming independent errors):

    sigma_dt = MAD(dt_AB - dt_BA) * 1.4826 / sqrt(2)

Reads the CURRENT measured result (read-only). Per band: matched pairs,
scatter plot, robust sigma.
"""
import glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RESULT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/work/forward/result_M3"
BANDS = ["2_6", "3_10", "6_18"]
COL = {"2_6": "#2a78d6", "3_10": "#eb6834", "6_18": "#1baf7a"}

fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
print(f"result: {RESULT}")
for ax, band in zip(axs, BANDS):
    dt = {}
    for f in glob.glob(f"{RESULT}/*_Z/OPT_{band}/*.adjinfo"):
        src = f.split(os.sep)[-3][:-2]
        rec = ".".join(os.path.basename(f).split(".")[:2])
        dt[(src, rec)] = float(open(f).read().split()[0])
    fwd, rev = [], []
    seen = set()
    for (a, b), v in dt.items():
        if (b, a) in dt and (b, a) not in seen:
            fwd.append(v); rev.append(dt[(b, a)]); seen.add((a, b))
    fwd = np.array(fwd); rev = np.array(rev)
    if len(fwd) == 0:
        ax.set_title(f"{band}: no reciprocal pairs"); continue
    diff = fwd - rev
    sig = 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2)
    lim = np.percentile(np.abs(np.concatenate([fwd, rev])), 99)
    ax.plot([-lim, lim], [-lim, lim], color="#c3c2b7", lw=1)
    ax.scatter(fwd, rev, s=8, alpha=0.5, color=COL[band], edgecolors="none")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(f"{band.replace('_','-')} s   n={len(fwd)} reciprocal pairs\n"
                 f"median diff={np.median(diff):+.3f}s   sigma_dt={sig:.3f}s")
    ax.set_xlabel("dt  A->B (s)"); ax.set_ylabel("dt  B->A (s)")
    r = np.corrcoef(fwd, rev)[0, 1] if len(fwd) > 2 else np.nan
    print(f"  {band}: n={len(fwd)}  median(dt_AB-dt_BA)={np.median(diff):+.3f}s  "
          f"sigma_dt={sig:.3f}s  corr={r:.3f}  median|dt|={np.median(np.abs(fwd)):.3f}s")
fig.suptitle("Reciprocal-pair dt consistency — per-measurement error estimate", fontsize=13)
out = f"{ROOT}/figures/reciprocity_dt.png"
fig.savefig(out, dpi=120)
print("wrote", out)
