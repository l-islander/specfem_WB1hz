#!/usr/bin/env python3
"""asdf_to_semd.py -- unpack a SPECFEM ASDF seismogram file (OUTPUT_FILES/synthetic.h5,
written with ASDF_FORMAT = .true.) into one binary trace per station plus the
seismo_meta.txt sidecar, i.e. exactly what a SEISMO_FORMAT=bin forward leaves in
an event result dir. That lets validate_forward.py (and cal_adjoint's
read_semd) consume ASDF output unchanged while the workflow is still per-trace.

Usage: asdf_to_semd.py synthetic.h5 OUTDIR [--component Z] [--start-time -0.10875] [--dt 0.0145]
  --start-time/--dt : written to seismo_meta.txt; default: taken from the ASDF
                      trace (dt = 1/sampling_rate) and the event origin in the
                      file's QuakeML (start = trace.starttime - origin).
Needs pyasdf.
"""
import argparse, os, sys, warnings
import numpy as np
import pyasdf
# SPECFEM writes no AuxiliaryData group unless ASDF_OUTPUT_PROVENANCE is on, and
# pyasdf 0.8.2 + h5py 3.16 warn once at close; both are harmless here.
warnings.filterwarnings("ignore", category=pyasdf.ASDFWarning)

ap = argparse.ArgumentParser()
ap.add_argument("h5"); ap.add_argument("outdir")
ap.add_argument("--component", default="Z", help="orientation letter(s) to keep (default Z; '' = all)")
ap.add_argument("--start-time", type=float, default=None); ap.add_argument("--dt", type=float, default=None)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
ds = pyasdf.ASDFDataSet(a.h5, mode="r")
origin = None
try:
    ev = ds.events[0]
    origin = (ev.preferred_origin() or ev.origins[0]).time
except Exception as e:  # noqa
    print("no usable QuakeML origin in file:", e)
n = 0; dt = a.dt; st0 = a.start_time; tags = set()
for sta in ds.waveforms.list():
    acc = ds.waveforms[sta]
    for tag in acc.get_waveform_tags():
        tags.add(tag)
        for tr in acc[tag]:
            cha = tr.stats.channel
            if a.component and cha[-1] not in a.component:
                continue
            name = f"{tr.stats.network}.{tr.stats.station}.{cha}.semd"
            tr.data.astype(np.float32).tofile(os.path.join(a.outdir, name))
            if dt is None: dt = 1.0 / tr.stats.sampling_rate
            if st0 is None and origin is not None: st0 = float(tr.stats.starttime - origin)
            n += 1
with open(os.path.join(a.outdir, "seismo_meta.txt"), "w") as f:
    f.write(f"format=bin\nnstep={tr.stats.npts}\ndt={dt}\nstart_time={st0 if st0 is not None else 'NA'}\n"
            f"band={cha[:2]}\nsource=asdf:{os.path.abspath(a.h5)}\ntags={','.join(sorted(tags))}\n")
print(f"wrote {n} traces -> {a.outdir} (dt={dt}, start_time={st0}, tags={sorted(tags)})")
