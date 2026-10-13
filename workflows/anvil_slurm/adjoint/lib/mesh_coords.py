#!/usr/bin/env python3
"""Per-GLL coordinates from SPECFEM3D Cartesian external_mesh.bin.

The kernel files proc*_<name>.bin store one value per GLL point in Fortran
(i,j,k,ispec) order, length NGLLX*NGLLY*NGLLZ*NSPEC. external_mesh.bin
(save_arrays_solver_mesh) begins with the records:

    nspec  nglob  nspec_irregular           (3 single ints)
    ibool(NGLLX,NGLLY,NGLLZ,NSPEC)           (int, Fortran order)
    xstore_unique(nglob)                     (real4)
    ystore_unique(nglob)
    zstore_unique(nglob)
    ... (the rest is not read)

So the coordinate of GLL point n (in kernel order) is xstore_unique[ibool[n]-1].
read_proc_coords returns (xg, yg, zg), each float32 length NSPEC*125, aligned
1:1 with read_sem_bin(proc*_<name>.bin). Validated against output_list_stations
positions and against the kernel record length.

A .npy cache (proc*_xyz.npy, shape (3, ngll)) is written next to the mesh on
first read so repeated callers (one per event) don't re-parse the big binary.
"""
import os

import numpy as np

_M = np.dtype("<i4")
_R = np.dtype("<f4")
NGLL3 = 125  # NGLLX*NGLLY*NGLLZ = 5^3


def _read_record(f, dtype):
    head = np.fromfile(f, _M, 1)
    if head.size != 1:
        raise IOError("truncated record (no leading marker)")
    nbytes = int(head[0])
    data = np.fromfile(f, dtype, nbytes // dtype.itemsize)
    tail = np.fromfile(f, _M, 1)
    if tail.size != 1 or int(tail[0]) != nbytes:
        raise IOError(f"record marker mismatch (lead={nbytes}, "
                      f"trail={tail.tolist()})")
    return data


def read_proc_coords(topo_dir, p, cache=True):
    """Return (xg, yg, zg) float32 arrays of per-GLL coords for proc p,
    aligned with the kernel bin layout."""
    cache_path = os.path.join(topo_dir, f"proc{p:06d}_xyz.npy")
    if cache and os.path.exists(cache_path):
        a = np.load(cache_path)
        return a[0], a[1], a[2]

    mesh_path = os.path.join(topo_dir, f"proc{p:06d}_external_mesh.bin")
    with open(mesh_path, "rb") as f:
        nspec = int(_read_record(f, _M)[0])
        _nglob = int(_read_record(f, _M)[0])
        _nspec_irr = int(_read_record(f, _M)[0])
        ibool = _read_record(f, _M)            # NSPEC*125, Fortran order
        x = _read_record(f, _R)
        y = _read_record(f, _R)
        z = _read_record(f, _R)
    if ibool.size != nspec * NGLL3:
        raise IOError(f"{mesh_path}: ibool size {ibool.size} != "
                      f"nspec*125 ({nspec * NGLL3})")
    idx = ibool - 1
    xg = x[idx].astype(_R, copy=False)
    yg = y[idx].astype(_R, copy=False)
    zg = z[idx].astype(_R, copy=False)
    if cache:
        try:
            np.save(cache_path, np.stack([xg, yg, zg]))
        except OSError:
            pass
    return xg, yg, zg


if __name__ == "__main__":
    import sys
    topo = sys.argv[1] if len(sys.argv) > 1 else "DATABASES_MPI"
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    xg, yg, zg = read_proc_coords(topo, p, cache=False)
    print(f"proc {p}: ngll={xg.size}  "
          f"X[{xg.min():.0f},{xg.max():.0f}] "
          f"Y[{yg.min():.0f},{yg.max():.0f}] "
          f"Z[{zg.min():.0f},{zg.max():.0f}]")
