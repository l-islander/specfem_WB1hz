#!/usr/bin/env python3
"""gll_to_tomo.py -- export a GLL model (proc*_{vp,vs,rho}.bin) back to a
SPECFEM tomography_model.xyz, so the model survives a REMESH.

WHY THIS EXISTS
---------------
A GLL model is welded to the mesh it was built on: one value per GLL point,
in that mesh's (i,j,k,ispec) order. Change ANYTHING about the mesh -- NEX,
NPROC, depth, interfaces -- and every M* becomes unreadable. The tomography
.xyz format is the mesh-free escape hatch: xgenerate_databases (MODEL=tomo)
trilinearly interpolates it onto whatever mesh exists. So the remesh flow is:

    1.  python tools/gll_to_tomo.py --model models/M5          # -> M5 as .xyz
    2.  edit DATA/meshfem3D_files/Mesh_Par_file (new NEX etc.)
    3.  cp DATA/tomo_files/tomography_model.xyz DATA/tomo_files/tomography_model.xyz.bak
        cp DATA/tomo_files/tomography_model_M5.xyz DATA/tomo_files/tomography_model.xyz
        (the Mesh_Par_file material line names tomography_model.xyz literally,
         so the export must REPLACE it -- keep the backup)
    4.  ./2_make_mesh.sh --force  &&  ./3_make_M0.sh --force   # M5 baked onto new mesh
    5.  rm -rf models/LBFGS_STATE; set ITER=0 in config.sh     # curvature pairs are
                                                               # partition-dependent
                                                               # (same reason ite3's
                                                               # Anvil state was dropped)

GRID -- defaults to the template's own grid (the existing tomography_model.xyz
header), which is proven to work with this Mesh_Par_file. Its 1.32 x 1.25 x
1.01 km spacing comfortably resolves the gradient's 2.5 km smoothing scale.

VALUES -- inverse-distance-weighted mean of the k nearest GLL points (default
k=8; GLL spacing is ~440 m, so k=8 averages within roughly one element).
Tomo nodes OUTSIDE the mesh (below the 30 km bottom -- the template grid
reaches 100 km -- or above topography) clamp to the nearest GLL value, which
mirrors what SPECFEM's tomography reader does at its own grid edges anyway.

NPROC=1 only (this port's defining constraint). The mesh coordinates come
from a DATABASES_MPI/proc000000_external_mesh.bin of the CURRENT (pre-remesh)
mesh -- any solver workdir has one; auto-discovered, or pass --topo.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "adjoint", "lib"))
from sem_io import read_sem_bin            # noqa: E402
from mesh_coords import read_proc_coords   # noqa: E402

TOPO_CANDIDATES = [
    "work/adjoint/solver/DATABASES_MPI",
    "work/forward/solver/DATABASES_MPI",
    "work/genmesh_M0/DATABASES_MPI",
    "work/bench/solver/DATABASES_MPI",
]


def find_topo():
    for c in TOPO_CANDIDATES:
        p = os.path.join(ROOT, c)
        if os.path.exists(os.path.join(p, "proc000000_external_mesh.bin")) or \
           os.path.exists(os.path.join(p, "proc000000_xyz.npy")):
            return p
    return None


def read_template_header(path):
    with open(path) as f:
        l1, l2, l3 = f.readline(), f.readline(), f.readline()
    ox, oy, oz, ex, ey, ez = map(float, l1.split())
    dx, dy, dz = map(float, l2.split())
    nx, ny, nz = map(int, l3.split())
    return (ox, oy, oz, ex, ey, ez), (dx, dy, dz), (nx, ny, nz), (l1, l2, l3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help="model dir holding proc000000_{vp,vs,rho}.bin (e.g. models/M5)")
    ap.add_argument("--template",
                    default=os.path.join(ROOT, "DATA/tomo_files/tomography_model.xyz"),
                    help="existing tomo file whose grid+header to mirror")
    ap.add_argument("--topo", default=None,
                    help="DATABASES_MPI dir with proc000000_external_mesh.bin "
                         "(default: auto-discover a solver workdir)")
    ap.add_argument("--out", default=None,
                    help="output path (default: DATA/tomo_files/"
                         "tomography_model_<modelname>.xyz)")
    ap.add_argument("--k", type=int, default=8,
                    help="nearest GLL points per tomo node for the IDW mean")
    ap.add_argument("--workers", type=int, default=-1,
                    help="KDTree query threads (-1 = all; lower it while the "
                         "inversion loop is running)")
    ap.add_argument("--nx", type=int, default=None,
                    help="override grid node count in x (extents stay the "
                         "template's). Use a FINE grid (~GLL spacing, e.g. "
                         "--nx 300 --ny 570) when the export seeds a remesh, "
                         "so the intermediate grid loses nothing.")
    ap.add_argument("--ny", type=int, default=None, help="override node count in y")
    ap.add_argument("--nz", type=int, default=None,
                    help="override node count in z. NB the M1+ updates carry "
                         "SMOOTH_SIGMA_V=200 m vertical structure -- the "
                         "template's ~1 km vertical spacing smears it; use "
                         "--nz 300+ (<=333 m) when exporting an updated model.")
    args = ap.parse_args()

    from scipy.spatial import cKDTree

    model = os.path.abspath(args.model)
    name = os.path.basename(model.rstrip("/"))
    out = args.out or os.path.join(ROOT, "DATA/tomo_files",
                                   f"tomography_model_{name}.xyz")
    topo = args.topo or find_topo()
    if topo is None:
        sys.exit("ERROR: no DATABASES_MPI with proc000000_external_mesh.bin found "
                 f"under {ROOT}/work -- pass --topo. (After wiping work/, re-run "
                 "any solver stage once, or ./3_make_M0.sh --force.)")
    if os.path.abspath(out) == os.path.abspath(args.template):
        sys.exit("ERROR: refusing to overwrite the template in place -- write to a "
                 "new file, then copy over the original yourself (keeping a .bak).")

    # ---- inputs -----------------------------------------------------------
    fields = {}
    for p in ("vp", "vs", "rho"):
        path = os.path.join(model, f"proc000000_{p}.bin")
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found -- is {name} a complete GLL model?")
        fields[p] = read_sem_bin(path)
    xg, yg, zg = read_proc_coords(topo, 0)
    n = xg.size
    for p, v in fields.items():
        if v.size != n:
            sys.exit(f"ERROR: {p} has {v.size} values but the mesh in {topo} has "
                     f"{n} GLL points -- model and mesh do not match. This tool "
                     "must run against the mesh the model was BUILT on (i.e. "
                     "before the remesh).")
    print(f"model : {model}  ({n:,} GLL points/field)")
    print(f"coords: {topo}")

    # ---- target grid: template extents, template or overridden counts -----
    (ox, oy, oz, ex, ey, ez), (dx, dy, dz), (tnx, tny, tnz), hdr = \
        read_template_header(args.template)
    nx = args.nx or tnx
    ny = args.ny or tny
    nz = args.nz or tnz
    custom = (nx, ny, nz) != (tnx, tny, tnz)
    if custom:                            # regenerate header lines for the new grid
        dx = (ex - ox) / (nx - 1); dy = (ey - oy) / (ny - 1); dz = (ez - oz) / (nz - 1)
        hdr = (f"{ox} {oy} {oz}  {ex} {ey} {ez}\n",
               f"{dx} {dy} {dz}\n",
               f"{nx} {ny} {nz}\n")
    x = np.linspace(ox, ex, nx)
    y = np.linspace(oy, ey, ny)
    z = np.linspace(oz, ez, nz)          # template runs 0 -> -100000 (top->down)
    print(f"grid  : {nx} x {ny} x {nz} = {nx*ny*nz:,} nodes, "
          f"spacing {abs(dx):.0f} x {abs(dy):.0f} x {abs(dz):.0f} m "
          f"({'custom counts, template extents' if custom else 'mirroring template'})")

    # record order: x fastest, then y, then z -- same as the template
    Xf = np.tile(x, ny * nz)
    Yf = np.tile(np.repeat(y, nx), nz)
    Zf = np.repeat(z, nx * ny)

    # ---- k-NN inverse-distance interpolation, chunked ---------------------
    t0 = time.time()
    tree = cKDTree(np.column_stack([xg, yg, zg]).astype(np.float64))
    print(f"kdtree: built in {time.time()-t0:.0f}s")

    ntot = nx * ny * nz
    CHUNK = 4_000_000                    # nodes per block: keeps query+IDW ~flat in RAM
    interp = {p: np.empty(ntot, dtype=np.float64) for p in fields}
    t0 = time.time()
    for lo in range(0, ntot, CHUNK):
        hi = min(lo + CHUNK, ntot)
        d, idx = tree.query(np.column_stack([Xf[lo:hi], Yf[lo:hi], Zf[lo:hi]]),
                            k=args.k, workers=args.workers)
        w = 1.0 / np.maximum(d, 1e-3) ** 2   # 1 mm floor: exact hits dominate
        wsum = w.sum(axis=1)
        for p, v in fields.items():
            interp[p][lo:hi] = (w * v[idx]).sum(axis=1) / wsum
        if ntot > CHUNK:
            print(f"        {hi:,}/{ntot:,} nodes ({time.time()-t0:.0f}s)")
    print(f"query : {ntot:,} nodes x k={args.k} in {time.time()-t0:.0f}s")

    # ---- sanity: the same guards 3_make_M0.sh will apply ------------------
    inmesh = Zf >= zg.min()              # nodes below the mesh bottom are clamped
    med_vs = float(np.median(interp["vs"][inmesh]))
    print(f"check : z_min={z.min():.0f} m (metres: ok), "
          f"median in-mesh Vs={med_vs:.0f} m/s "
          f"({'crustal: ok' if med_vs < 4100 else 'MANTLE-LIKE -- something is wrong'})")
    if med_vs >= 4100:
        sys.exit("ERROR: refusing to write a mantle-looking model (this is the "
                 "check that catches unit/order mistakes).")

    # ---- write; header lines 1-3 verbatim unless the grid was overridden --
    t0 = time.time()
    with open(out, "w") as f:
        f.write(hdr[0]); f.write(hdr[1]); f.write(hdr[2])
        f.write(f"{interp['vp'].min()} {interp['vp'].max()} "
                f"{interp['vs'].min()} {interp['vs'].max()} "
                f"{interp['rho'].min()} {interp['rho'].max()}\n")
        for lo in range(0, ntot, CHUNK):
            hi = min(lo + CHUNK, ntot)
            np.savetxt(f, np.column_stack([Xf[lo:hi], Yf[lo:hi], Zf[lo:hi],
                                           interp["vp"][lo:hi],
                                           interp["vs"][lo:hi],
                                           interp["rho"][lo:hi]]),
                       fmt="%.6f")
    print(f"wrote : {out}  ({time.time()-t0:.0f}s)")
    print(f"\nnext, to seed a remeshed run with {name}:")
    print(f"  cp {args.template} {args.template}.bak")
    print(f"  cp {out} {args.template}")
    print("  # edit Mesh_Par_file, then: ./2_make_mesh.sh --force && ./3_make_M0.sh --force")
    print("  # and reset: rm -rf models/LBFGS_STATE, ITER=0 in config.sh")


if __name__ == "__main__":
    main()
