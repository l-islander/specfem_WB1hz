#!/usr/bin/env python3
"""adj_to_asdf.py -- convert SPECFEM ASCII adjoint sources (SEM/*.adj) into the
ASDF file xspecfem3D reads with READ_ADJSRC_ASDF = .true. (SEM/adjoint.h5).

The solver (src/shared/asdf_manager.f90 + read_adjoint_sources_ASDF.f90) opens
<rundir>/SEM/adjoint.h5 and, for every station in STATIONS_ADJOINT and every
component, reads AuxiliaryData/AdjointSources/<NET>_<STA>_<BAND><ORIENT>
(e.g. 3A_20111_BXZ) as a plain float array of NSTEP samples in natural time
order -- exactly the second column of the ASCII <NET>.<STA>.<BAND><ORIENT>.adj.

Usage: adj_to_asdf.py <SEM dir with *.adj> [--out SEM/adjoint.h5] [--nstep 6900]
Needs pyasdf (pip/conda). Writes uncompressed datasets.
"""
import argparse, glob, os, re, sys
import numpy as np
import pyasdf

ap = argparse.ArgumentParser()
ap.add_argument("semdir"); ap.add_argument("--out"); ap.add_argument("--nstep", type=int, default=0)
ap.add_argument("--dt", type=float, default=0.0)
a = ap.parse_args()
out = a.out or os.path.join(a.semdir, "adjoint.h5")
files = sorted(glob.glob(os.path.join(a.semdir, "*.adj")))
if not files:
    sys.exit(f"no *.adj under {a.semdir}")
if os.path.exists(out):
    os.remove(out)
ds = pyasdf.ASDFDataSet(out, compression=None, mode="w")
n = 0
for f in files:
    m = re.match(r"([A-Za-z0-9]+)\.([A-Za-z0-9]+)\.([A-Z]{3})\.adj$", os.path.basename(f))
    if not m:
        print("skip (name pattern):", f); continue
    net, sta, cha = m.groups()
    arr = np.loadtxt(f)
    t, u = arr[:, 0], arr[:, 1].astype(np.float32)
    if a.nstep and u.size != a.nstep:
        sys.exit(f"{f}: {u.size} samples, expected {a.nstep}")
    dt = float(np.median(np.diff(t))) if t.size > 1 else a.dt
    ds.add_auxiliary_data(data=u, data_type="AdjointSources", path=f"{net}_{sta}_{cha}",
                          parameters={"dt": dt, "starttime": float(t[0]), "npts": int(u.size),
                                      "station_id": f"{net}.{sta}..{cha}", "source": os.path.basename(f)})
    n += 1
del ds
print(f"wrote {out}: {n} adjoint sources")
