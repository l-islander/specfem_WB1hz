"""Cross-band traveltime-shift consistency check on the measured basis.

For every pair measured in more than one band, compare the cc dt across bands:
sign agreement, correlation, and systematic offsets (dispersion signal).
Convention (pyadjoint xcorr_shift): dt > 0 = observed later than synthetic
= model too FAST along the path; dt < 0 = model too slow.

Writes figures/band_dt_consistency.png and prints the stats.
"""
import glob
import os

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATADIR = os.path.join(ROOT, "work/forward/result_M0")
BANDS = ["2_6", "3_10", "6_18"]
COL = {"2_6": "#4477aa", "3_10": "#44aa77", "6_18": "#cc6677"}

# ---- collect dt (and distance from the .win header) per pair per band ------
dt = {}     # (src, rec) -> {band: dt}
dist = {}   # (src, rec) -> km
for f in glob.glob(f"{DATADIR}/*_Z/OPT_*/*.adjinfo"):
    parts = f.split(os.sep)
    src, band = parts[-3][:-2], parts[-2].split("OPT_")[1]
    rec = ".".join(os.path.basename(f).split(".")[:2])
    val = float(open(f).read().split()[0])
    dt.setdefault((src, rec), {})[band] = val
    if (src, rec) not in dist:
        rn, rs = rec.split(".")
        wp = os.path.join(DATADIR, f"{src}_Z", f"WIN_{band}",
                          f"{rs}.{rn}..BXZ.win")
        if os.path.exists(wp):
            try:
                dist[(src, rec)] = float(open(wp).readline().split()[1])
            except Exception:
                pass

print(f"pairs with >=1 band: {len(dt)}")
print("\n== per-band dt statistics (dt>0: model too fast; dt<0: too slow) ==")
for b in BANDS:
    v = np.array([d[b] for d in dt.values() if b in d])
    print(f"  {b:5s}: n={len(v):5d}  mean={v.mean():+.3f}s  median={np.median(v):+.3f}s"
          f"  |dt| med={np.median(np.abs(v)):.2f}s  frac dt>0: {np.mean(v > 0)*100:.0f}%")

combos = [("2_6", "3_10"), ("3_10", "6_18"), ("2_6", "6_18")]
print("\n== same-pair cross-band consistency ==")
stats = {}
for a, b in combos:
    both = [(d[a], d[b]) for d in dt.values() if a in d and b in d]
    if len(both) < 10:
        print(f"  {a} vs {b}: only {len(both)} shared pairs -- skipped")
        continue
    va, vb = map(np.array, zip(*both))
    nz = (va != 0) & (vb != 0)
    agree = np.mean(np.sign(va[nz]) == np.sign(vb[nz])) * 100
    r = np.corrcoef(va, vb)[0, 1]
    off = np.mean(va - vb)
    stats[(a, b)] = (va, vb)
    print(f"  {a} vs {b}: {len(both):4d} shared pairs | sign agreement {agree:.0f}% "
          f"| corr r={r:.2f} | mean(dt_{a} - dt_{b}) = {off:+.3f}s")

n3 = sum(1 for d in dt.values() if len(d) == 3)
print(f"  pairs measured in all 3 bands: {n3}")

# ---- figure -----------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13.5, 11), constrained_layout=True)
ax = axs[0, 0]
for b in BANDS:
    v = np.array([d[b] for d in dt.values() if b in d])
    ax.hist(v, bins=np.linspace(-3, 3, 61), histtype="step", lw=1.8,
            color=COL[b], label=f"{b.replace('_','-')}s (n={len(v)}, "
                                f"med {np.median(v):+.2f}s)")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("dt (s)   [>0 : model too fast]"); ax.set_ylabel("pairs")
ax.set_title("measured cc dt per band"); ax.legend(fontsize=9)

for k, (a, b) in enumerate(combos[:2]):
    axx = axs[0, 1] if k == 0 else axs[1, 0]
    if (a, b) not in stats:
        axx.set_visible(False); continue
    va, vb = stats[(a, b)]
    lim = max(np.percentile(np.abs(np.r_[va, vb]), 99), 1.0)
    axx.plot([-lim, lim], [-lim, lim], "k-", lw=0.7)
    axx.axhline(0, color="gray", lw=0.5); axx.axvline(0, color="gray", lw=0.5)
    axx.plot(va, vb, ".", ms=4, alpha=0.5, color="#4477aa")
    r = np.corrcoef(va, vb)[0, 1]
    axx.set_xlim(-lim, lim); axx.set_ylim(-lim, lim); axx.set_aspect("equal")
    axx.set_xlabel(f"dt {a.replace('_','-')}s (s)")
    axx.set_ylabel(f"dt {b.replace('_','-')}s (s)")
    axx.set_title(f"same pair: {a.replace('_','-')}s vs {b.replace('_','-')}s "
                  f"(n={len(va)}, r={r:.2f})")

ax = axs[1, 1]
for b in BANDS:
    pts = [(dist[k], d[b]) for k, d in dt.items() if b in d and k in dist]
    if not pts:
        continue
    x, y = map(np.array, zip(*pts))
    bins = np.arange(0, x.max() + 20, 20)
    idx = np.digitize(x, bins)
    bx, by, be = [], [], []
    for i in range(1, len(bins)):
        m = idx == i
        if m.sum() >= 10:
            bx.append(bins[i] - 10); by.append(np.mean(y[m]))
            be.append(np.std(y[m]) / np.sqrt(m.sum()))
    ax.errorbar(bx, by, yerr=be, color=COL[b], lw=1.8, capsize=2,
                label=f"{b.replace('_','-')}s")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("distance (km)"); ax.set_ylabel("mean dt (s)")
ax.set_title("mean dt vs distance (20 km bins, ±SE)")
ax.legend(fontsize=9)

fig.suptitle("Cross-band dt sanity check — band-2 M0 measurements "
             "(dt>0: model too fast on path)", fontsize=13)
out = os.path.join(ROOT, "figures", "band_dt_consistency.png")
fig.savefig(out, dpi=110)
print(f"\nwrote {out}")
