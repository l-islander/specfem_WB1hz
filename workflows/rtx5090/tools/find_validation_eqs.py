"""Find local earthquakes for waveform validation of the FWI model chain.

Ranks catalog events by how many of OUR stations (DATA/STATIONS, matched by
net.sta against FDSN epochs) were operating at origin time. Constraints keep
the source well inside the mesh volume:
  lat 43.692..45.931, lon -123.629..-121.963, depth 30 km block
  -> events >= EDGE_KM from lateral edges, depth <= 25 km.

Usage: .venv python tools/find_validation_eqs.py [minmag]  (default 2.5)
"""
import os, sys
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAT0, LAT1 = 43.692567, 45.931036
LON0, LON1 = -123.628855, -121.962818
EDGE_KM = 8.0          # min distance from lateral mesh edge
MAXDEP = 25.0          # km, mesh block is 30 km
MINMAG = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
T0, T1 = UTCDateTime(2006, 1, 1), UTCDateTime(2026, 8, 1)

# our stations
sta = {}
for l in open(f"{ROOT}/DATA/STATIONS"):
    p = l.split()
    if len(p) >= 4:
        sta[(p[1], p[0])] = (float(p[2]), float(p[3]))
nets = sorted({n for n, s in sta})
print(f"{len(sta)} stations, networks: {' '.join(nets)}", flush=True)

cli = Client("IRIS", timeout=120)

# per-station operating epochs (station level, matched to our list)
epochs = {}
inv = cli.get_stations(network=",".join(nets), level="station",
                       minlatitude=LAT0 - 0.1, maxlatitude=LAT1 + 0.1,
                       minlongitude=LON0 - 0.1, maxlongitude=LON1 + 0.1,
                       starttime=T0, endtime=T1)
for net in inv:
    for s in net:
        k = (net.code, s.code)
        if k in sta:
            epochs.setdefault(k, []).append(
                (s.start_date, s.end_date or UTCDateTime(2100, 1, 1)))
print(f"epochs matched for {len(epochs)} of {len(sta)} stations", flush=True)

# events (USGS catalog via IRIS event service falls back fine; use USGS)
try:
    ev_cli = Client("USGS", timeout=120)
except Exception:
    ev_cli = cli
cat = ev_cli.get_events(minlatitude=LAT0, maxlatitude=LAT1,
                        minlongitude=LON0, maxlongitude=LON1,
                        minmagnitude=MINMAG, maxdepth=MAXDEP,
                        starttime=T0, endtime=T1, orderby="magnitude")
print(f"{len(cat)} catalog events M>={MINMAG}", flush=True)

# lateral edge margin in degrees (approx)
dlat = EDGE_KM / 111.0
rows = []
for ev in cat:
    o = ev.preferred_origin() or ev.origins[0]
    m = ev.preferred_magnitude() or ev.magnitudes[0]
    la, lo, dep = o.latitude, o.longitude, (o.depth or 0) / 1e3
    dlon = EDGE_KM / (111.0 * np.cos(np.radians(la)))
    if not (LAT0 + dlat < la < LAT1 - dlat and LON0 + dlon < lo < LON1 - dlon):
        continue
    live = [k for k, eps in epochs.items()
            if any(a <= o.time <= b for a, b in eps)]
    if not live:
        continue
    dists = sorted(gps2dist_azimuth(la, lo, *sta[k])[0] / 1e3 for k in live)
    azs = sorted(gps2dist_azimuth(la, lo, *sta[k])[2] for k in live)
    gaps = [azs[i + 1] - azs[i] for i in range(len(azs) - 1)] + \
           [azs[0] + 360 - azs[-1]]
    rows.append((len(live), m.mag, o.time, la, lo, dep,
                 dists[0], dists[len(dists) // 2], max(gaps),
                 (ev.resource_id.id.split("eventid=")[1].split("&")[0]
                  if "eventid=" in ev.resource_id.id
                  else ev.resource_id.id.split("/")[-1])))

rows.sort(key=lambda r: (-r[0], -r[1]))
print(f"\n{len(rows)} events inside mesh with live stations; top 25 by coverage:")
print("nsta  mag   origin_time           lat      lon       dep  d_min d_med "
      "azgap  id")
for r in rows[:25]:
    print(f"{r[0]:4d}  M{r[1]:3.1f}  {str(r[2])[:19]}  {r[3]:.3f}  {r[4]:.3f}  "
          f"{r[5]:5.1f}  {r[6]:5.1f} {r[7]:5.1f}  {r[8]:4.0f}  {r[9]}")
