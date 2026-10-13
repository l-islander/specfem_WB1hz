"""Uncertainty proxy #3: sampling maps — midpoint density + azimuthal coverage.

Where resolution CAN exist: per 10-km cell, (a) count of pair midpoints,
(b) azimuthal diversity = number of 30-degree path-azimuth bins hit (folded to
0-180, so 6 bins max). Reads the current measured result (read-only), per band.
"""
import glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "figures"))
from geo2utm import geo2utm

RESULT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/work/forward/result_M3"
BANDS = ["2_6", "3_10", "6_18"]; CELL = 10.0

sta = {}
for p in (l.split() for l in open(f"{ROOT}/DATA/STATIONS")):
    if len(p) >= 4:
        e, n = geo2utm(float(p[3]), float(p[2]), 10)
        sta[f"{p[1]}.{p[0]}"] = ((e - 449321.0) / 1e3, (n - 4837704.1) / 1e3)

xs = [v[0] for v in sta.values()]; ys = [v[1] for v in sta.values()]
nx = int(max(xs) // CELL) + 2; ny = int(max(ys) // CELL) + 2

fig, axs = plt.subplots(2, 3, figsize=(13.5, 15), constrained_layout=True)
for c, band in enumerate(BANDS):
    dens = np.zeros((ny, nx)); azbins = np.zeros((ny, nx, 6), bool)
    npairs = 0
    for f in glob.glob(f"{RESULT}/*_Z/OPT_{band}/*.adjinfo"):
        src = f.split(os.sep)[-3][:-2]
        rec = ".".join(os.path.basename(f).split(".")[:2])
        if src not in sta or rec not in sta:
            continue
        (x1, y1), (x2, y2) = sta[src], sta[rec]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ix, iy = int(mx // CELL), int(my // CELL)
        if 0 <= ix < nx and 0 <= iy < ny:
            dens[iy, ix] += 1
            az = np.degrees(np.arctan2(x2 - x1, y2 - y1)) % 180.0
            azbins[iy, ix, int(az // 30)] = True
            npairs += 1
    div = azbins.sum(2).astype(float); div[dens == 0] = np.nan
    d_masked = np.ma.array(dens, mask=dens == 0)
    ext = [0, nx * CELL, 0, ny * CELL]
    im0 = axs[0, c].imshow(d_masked, origin="lower", extent=ext, cmap="Blues",
                           vmax=np.percentile(dens[dens > 0], 98) if (dens > 0).any() else 1)
    axs[0, c].set_title(f"{band.replace('_','-')} s — midpoint density ({npairs} pairs)")
    im1 = axs[1, c].imshow(div, origin="lower", extent=ext, cmap="RdYlBu", vmin=1, vmax=6)
    axs[1, c].set_title("azimuth bins hit (of 6)")
    for r in (0, 1):
        axs[r, c].set_aspect("equal"); axs[r, c].set_xlabel("Easting (km)")
    fig.colorbar(im0, ax=axs[0, c], shrink=0.75)
    fig.colorbar(im1, ax=axs[1, c], shrink=0.75)
    good = np.nansum(div >= 4)
    print(f"{band}: {npairs} pairs | cells sampled: {(dens>0).sum()} | "
          f"cells with >=4/6 azimuth bins: {good}")
axs[0, 0].set_ylabel("Northing (km)"); axs[1, 0].set_ylabel("Northing (km)")
fig.suptitle(f"Sampling coverage per band — {os.path.basename(RESULT)}, {CELL:.0f} km cells",
             fontsize=13)
out = f"{ROOT}/figures/coverage_maps.png"
fig.savefig(out, dpi=110)
print("wrote", out)
