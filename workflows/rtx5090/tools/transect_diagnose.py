"""Is the N~81 km transect anomaly data-driven or a coverage artifact?

Classifies every measured pair by its relationship to the ZO nodal line
(both endpoints on it / one / none) and tests whether the classes AGREE about
structure along the line:

  * mean dt vs midpoint easting, same-line vs crossing pairs, at M0 and M1.
    If crossing pairs (independent geometry) show the same dt signature where
    they sample the same volume, the feature is data-driven. If only the
    same-line pairs demand it, it is a line artifact.
  * per-class dt statistics and counts (how much of the basis is same-line).

Writes figures/transect_diagnosis.png.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "figures"))
from geo2utm import geo2utm

SP = os.environ.get("TRANSECT_OUT", "./transect_out")
LINE = set(json.load(open(f"{SP}/transect_stations.json")))

sta = {}
for p in (l.split() for l in open(os.path.join(ROOT, "DATA/STATIONS"))):
    if len(p) >= 4:
        e, n = geo2utm(float(p[3]), float(p[2]), 10)
        sta[f"{p[1]}.{p[0]}"] = (e / 1e3 - 449.321, n / 1e3 - 4837.704)


def collect(resdir):
    out = []                      # (src, rec, band, dt)
    for f in glob.glob(f"{resdir}/*_Z/OPT_*/*.adjinfo"):
        parts = f.split(os.sep)
        src, band = parts[-3][:-2], parts[-2].split("OPT_")[1]
        rec = ".".join(os.path.basename(f).split(".")[:2])
        out.append((src, rec, band, float(open(f).read().split()[0])))
    return out


def classify(pairs):
    cls = {"same-line": [], "crossing": [], "unrelated": []}
    for src, rec, band, dt in pairs:
        if src not in sta or rec not in sta:
            continue
        k = (src in LINE) + (rec in LINE)
        name = ("unrelated", "crossing", "same-line")[k]
        mx = (sta[src][0] + sta[rec][0]) / 2
        my = (sta[src][1] + sta[rec][1]) / 2
        cls[name].append((mx, my, dt))
    return {k: np.array(v) for k, v in cls.items() if v}


M0 = classify(collect(os.path.join(ROOT, "work/forward/result_M0")))
M1 = classify(collect(os.path.join(ROOT, "work/forward/result_M1")))

print("class            n(M1)   dt mean   dt med   |dt| med   frac>0")
for k in ("same-line", "crossing", "unrelated"):
    v = M1[k][:, 2]
    print(f"{k:12s} {len(v):8d}   {v.mean():+.3f}   {np.median(v):+.3f}"
          f"    {np.median(np.abs(v)):.2f}      {np.mean(v > 0)*100:.0f}%")

# ---- figure -----------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10.5), constrained_layout=True)

# (a) map: stations, line highlighted
ax = axs[0, 0]
E = [v[0] for v in sta.values()]; N = [v[1] for v in sta.values()]
ax.scatter(E, N, s=4, c="lightgray")
le = [sta[s][0] for s in LINE if s in sta]; ln = [sta[s][1] for s in LINE if s in sta]
ax.scatter(le, ln, s=6, c="crimson", label=f"ZO line (n={len(le)})")
ax.set_aspect("equal"); ax.legend(fontsize=9)
ax.set_title("the transect under test"); ax.set_xlabel("E (km)"); ax.set_ylabel("N (km)")

# (b) THE TEST: mean dt vs midpoint easting near the line, per class, M1
ax = axs[0, 1]
bins = np.arange(10, 100, 6)
for cls_name, color in (("same-line", "#c1272d"), ("crossing", "#4477aa")):
    v = M1[cls_name]
    nearline = np.abs(v[:, 1] - 81) < 12          # midpoint within 12 km of line
    x, dt = v[nearline, 0], v[nearline, 2]
    idx = np.digitize(x, bins)
    bx, by, be = [], [], []
    for i in range(1, len(bins)):
        m = idx == i
        if m.sum() >= 5:
            bx.append(bins[i] - 3); by.append(dt[m].mean())
            be.append(dt[m].std() / np.sqrt(m.sum()))
    ax.errorbar(bx, by, yerr=be, color=color, lw=1.8, capsize=2,
                label=f"{cls_name} (n={nearline.sum()})")
ax.axhline(0, color="k", lw=0.7)
ax.set_title("M1 residual dt along the line — do independent geometries agree?")
ax.set_xlabel("pair midpoint easting (km)"); ax.set_ylabel("mean dt (s)  [>0: model fast]")
ax.legend(fontsize=9)

# (c) same but at M0 (before the update fed the feature)
ax = axs[1, 0]
for cls_name, color in (("same-line", "#c1272d"), ("crossing", "#4477aa")):
    v = M0[cls_name]
    nearline = np.abs(v[:, 1] - 81) < 12
    x, dt = v[nearline, 0], v[nearline, 2]
    idx = np.digitize(x, bins)
    bx, by, be = [], [], []
    for i in range(1, len(bins)):
        m = idx == i
        if m.sum() >= 5:
            bx.append(bins[i] - 3); by.append(dt[m].mean())
            be.append(dt[m].std() / np.sqrt(m.sum()))
    ax.errorbar(bx, by, yerr=be, color=color, lw=1.8, capsize=2, label=cls_name)
ax.axhline(0, color="k", lw=0.7)
ax.set_title("same test at M0 (pre-update)")
ax.set_xlabel("pair midpoint easting (km)"); ax.set_ylabel("mean dt (s)")
ax.legend(fontsize=9)

# (d) dt histograms per class, M1
ax = axs[1, 1]
for cls_name, color in (("same-line", "#c1272d"), ("crossing", "#4477aa"),
                        ("unrelated", "gray")):
    ax.hist(M1[cls_name][:, 2], bins=np.linspace(-2.5, 2.5, 60), histtype="step",
            lw=1.7, color=color, density=True, label=cls_name)
ax.axvline(0, color="k", lw=0.7)
ax.set_title("M1 dt distributions per class"); ax.set_xlabel("dt (s)")
ax.legend(fontsize=9)

fig.suptitle("Transect diagnosis — ZO nodal line at N≈81 km", fontsize=14)
out = os.path.join(ROOT, "figures", "transect_diagnosis.png")
fig.savefig(out, dpi=110)
print(f"wrote {out}")
