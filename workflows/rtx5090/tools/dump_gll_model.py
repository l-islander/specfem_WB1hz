#!/usr/bin/env python3
"""dump_gll_model.py -- write the EXACT GLL model: every unique GLL node's
(x, y, z, vp, vs, rho), no interpolation, no resampling. This is the true
model as the solver sees it; use it for archive, analysis, or as the input
truth when you build a transfer grid for a remesh.

WHAT "EXACT" MEANS HERE
-----------------------
proc000000_{vp,vs,rho}.bin hold NGLL3*NSPEC = 125*nspec values -- one per
element-local GLL slot. Slots on shared element faces/edges/corners REPEAT
the same physical node: the mesh has only nglob unique points (ibool maps
slot -> unique node, and external_mesh.bin stores the unique coordinate
arrays directly). This tool dumps the nglob unique nodes, and VERIFIES that
every duplicated slot carries an identical value (max spread is printed;
anything but ~0 would mean the field is discontinuous, i.e. not a model).

Output: .npz (compact, float32 -- the native precision of the .bin files,
so nothing is lost) with arrays x, y, z, vp, vs, rho, each length nglob.
--text additionally writes an ascii "x y z vp vs rho" table (~1 GB; NOT a
valid MODEL=tomo file -- that format requires a regular grid; for the
remesh transfer grid use gll_to_tomo.py, ideally with a fine --nx/ny/nz).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "adjoint", "lib"))
from sem_io import read_sem_bin                      # noqa: E402
from mesh_coords import _read_record, _M, _R, NGLL3  # noqa: E402
from gll_to_tomo import find_topo                    # noqa: E402


def read_mesh_unique(topo_dir, p=0):
    """Return (ibool0, xu, yu, zu): slot->unique map (0-based) and the unique
    coordinate arrays, straight from external_mesh.bin -- no reconstruction."""
    path = os.path.join(topo_dir, f"proc{p:06d}_external_mesh.bin")
    with open(path, "rb") as f:
        nspec = int(_read_record(f, _M)[0])
        nglob = int(_read_record(f, _M)[0])
        _ = _read_record(f, _M)                      # nspec_irregular
        ibool = _read_record(f, _M)                  # NGLL3*nspec, 1-based
        xu = _read_record(f, _R)
        yu = _read_record(f, _R)
        zu = _read_record(f, _R)
    if ibool.size != NGLL3 * nspec or xu.size != nglob:
        raise IOError(f"{path}: unexpected record sizes")
    return ibool.astype(np.int64) - 1, xu, yu, zu


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help="model dir with proc000000_{vp,vs,rho}.bin (e.g. models/M1)")
    ap.add_argument("--topo", default=None,
                    help="DATABASES_MPI dir with proc000000_external_mesh.bin "
                         "(default: auto-discover)")
    ap.add_argument("--out", default=None,
                    help="output .npz (default: <model>/<name>_gll_points.npz)")
    ap.add_argument("--text", action="store_true",
                    help="also write <out>.xyz ascii table (~1 GB)")
    ap.add_argument("--stats", action="store_true",
                    help="also report nearest-neighbour GLL spacing (KDTree sample)")
    args = ap.parse_args()

    model = os.path.abspath(args.model)
    name = os.path.basename(model.rstrip("/"))
    out = args.out or os.path.join(model, f"{name}_gll_points.npz")
    topo = args.topo or find_topo()
    if topo is None:
        sys.exit("ERROR: no DATABASES_MPI with external_mesh.bin found -- pass --topo")

    ib, xu, yu, zu = read_mesh_unique(topo)
    nglob, nslot = xu.size, ib.size
    print(f"mesh  : {topo}")
    print(f"        {nslot:,} GLL slots -> {nglob:,} unique nodes "
          f"({nslot/nglob:.2f}x duplication on shared faces)")

    cols = {"x": xu, "y": yu, "z": zu}
    for pnm in ("vp", "vs", "rho"):
        v = read_sem_bin(os.path.join(model, f"proc000000_{pnm}.bin"))
        if v.size != nslot:
            sys.exit(f"ERROR: {pnm} has {v.size} slots, mesh has {nslot} -- "
                     "model was built on a different mesh")
        vu = np.empty(nglob, dtype=v.dtype)
        vu[ib] = v                                   # scatter, last write wins
        spread = np.abs(v - vu[ib]).max()            # every slot vs stored value
        print(f"  {pnm:3s}: min={v.min():.1f} max={v.max():.1f}  "
              f"duplicate-slot spread={spread:.3g} "
              f"({'exact' if spread == 0 else 'CHECK: field discontinuous?'})")
        cols[pnm] = vu

    np.savez_compressed(out, **cols)
    print(f"wrote : {out}  ({os.path.getsize(out)/1e6:.0f} MB, "
          f"{nglob:,} nodes x 6 float32 -- native .bin precision, lossless)")

    if args.text:
        txt = os.path.splitext(out)[0] + ".xyz"
        t0 = time.time()
        np.savetxt(txt, np.column_stack([cols[c] for c in
                                         ("x", "y", "z", "vp", "vs", "rho")]),
                   fmt="%.6f")
        print(f"wrote : {txt}  ({os.path.getsize(txt)/1e9:.1f} GB, "
              f"{time.time()-t0:.0f}s)  [scattered points -- NOT a MODEL=tomo file]")

    if args.stats:
        from scipy.spatial import cKDTree
        sub = np.random.default_rng(0).choice(nglob, size=min(200_000, nglob),
                                              replace=False)
        tree = cKDTree(np.column_stack([xu, yu, zu]))
        d, _ = tree.query(np.column_stack([xu[sub], yu[sub], zu[sub]]),
                          k=2, workers=8)
        nn = d[:, 1]
        print(f"GLL nearest-neighbour spacing (n={len(sub):,} sample): "
              f"min={nn.min():.0f} m  median={np.median(nn):.0f} m  "
              f"max={nn.max():.0f} m")


if __name__ == "__main__":
    main()
