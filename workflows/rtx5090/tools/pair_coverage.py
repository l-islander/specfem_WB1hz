"""Per-source measured-pair coverage -> source-count decision support.

    python tools/pair_coverage.py [--datadir work/forward/result_M0]

Run AFTER cal_adjoint has measured a result dir. Reads every
<src>_Z/OPT_<band>/*.adjinfo, counts measured pairs per source, and answers:
if I keep only the top-N sources, what fraction of all measured pairs do I
retain, and does the adjoint stage fit on the internal disk?

Disk model (band-2 mesh, measured): ~2.6 GB kernels per event held until the
sum (raw 1.8 + clipped 0.8), + ~75 GB transient wavefield high-water mark.
Writes figures/pair_coverage.png and prints the decision table.
"""
import argparse
import glob
import os
import shutil

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--datadir", default=os.path.join(ROOT, "work/forward/result_M0"))
args = ap.parse_args()

KGB = 2.6          # GB of kernels per event, held simultaneously
TRANSIENT = 75.0   # GB wavefield high-water mark
SAFETY = 40.0      # GB slack to leave free

per_src = {}       # src -> {band: npairs}
bands = set()
for f in glob.glob(f"{args.datadir}/*_Z/OPT_*/*.adjinfo"):
    parts = f.split(os.sep)
    src, band = parts[-3][:-2], parts[-2].split("OPT_")[1]
    bands.add(band)
    per_src.setdefault(src, {}).setdefault(band, 0)
    per_src[src][band] += 1
bands = sorted(bands)
if not per_src:
    raise SystemExit(f"no .adjinfo under {args.datadir} -- run cal_adjoint first")

srcs = sorted(per_src, key=lambda s: -sum(per_src[s].values()))
totals = np.array([sum(per_src[s].values()) for s in srcs])
grand = totals.sum()
free_gb = shutil.disk_usage(ROOT).free / 1e9

print(f"sources measured : {len(srcs)}")
for b in bands:
    print(f"  band {b:5s}: {sum(v.get(b, 0) for v in per_src.values()):7d} pairs")
print(f"  TOTAL      : {grand:7d} pairs")
print(f"pairs/source: median={int(np.median(totals))}  mean={totals.mean():.0f}  "
      f"min={totals.min()}  max={totals.max()}")
print(f"disk free now: {free_gb:.0f} GB   (model: {KGB} GB/event kernels "
      f"+ {TRANSIENT:.0f} GB transient + {SAFETY:.0f} GB slack)")

print(f"\n{'keep top-N':>10} | {'pairs kept':>10} | {'% of all':>8} | "
      f"{'kernel+transient GB':>19} | fits internal?")
cands = [n for n in (147, 200, 250, 300, 350, 400, 450, len(srcs)) if n <= len(srcs)]
for n in cands:
    kept = totals[:n].sum()
    need = n * KGB + TRANSIENT
    fits = "YES" if need + SAFETY <= free_gb else "no -> 2TB drive"
    print(f"{n:>10} | {kept:>10} | {kept/grand*100:>7.1f}% | {need:>19.0f} | {fits}")

print("\ntop 10 sources:")
for s in srcs[:10]:
    print(f"  {s:12s} {sum(per_src[s].values()):5d}  "
          + "  ".join(f"{b}:{per_src[s].get(b,0)}" for b in bands))
print("bottom 10 sources:")
for s in srcs[-10:]:
    print(f"  {s:12s} {sum(per_src[s].values()):5d}  "
          + "  ".join(f"{b}:{per_src[s].get(b,0)}" for b in bands))

# ---- figure: histogram + cumulative coverage curve -------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
a1.hist(totals, bins=50, color="#4477aa")
a1.set_xlabel("measured pairs per source"); a1.set_ylabel("sources")
a1.set_title(f"pairs/source (n={len(srcs)}, total {grand:,})")

cum = np.cumsum(totals) / grand * 100
a2.plot(np.arange(1, len(srcs) + 1), cum, lw=2, color="#c1272d")
for n in (147, 250, 350, len(srcs)):
    if n <= len(srcs):
        a2.axvline(n, color="gray", ls=":", lw=1)
        a2.annotate(f"N={n}\n{cum[n-1]:.0f}% pairs\n{n*KGB+TRANSIENT:.0f} GB",
                    (n, cum[n - 1]), textcoords="offset points", xytext=(6, -42),
                    fontsize=8)
a2.axhline(100, color="k", lw=0.5)
a2.set_xlabel("sources kept (ranked by pair count)")
a2.set_ylabel("% of measured pairs retained")
a2.set_title("cumulative coverage vs source count")
out = os.path.join(ROOT, "figures", "pair_coverage.png")
fig.savefig(out, dpi=110)
print(f"\nwrote {out}")
