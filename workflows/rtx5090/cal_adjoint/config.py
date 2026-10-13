"""Shared processing parameters for the specfem_process pipeline (LOCAL GPU port).

DIFFERENCE FROM THE ANVIL VERSION
---------------------------------
Anvil's config.py hardcodes absolute /anvil/projects and /anvil/scratch paths.
Here every path is read from an environment variable, with a sensible default
relative to this file. `run_caladj_local.sh` exports them all from config.sh, so
`ite3_gpu/config.sh` stays the single source of truth and this file never needs
editing when the tree moves.

The science parameters (BANDS, SNR_CUTOFF, trim window, pyflex inputs) are
carried over VERBATIM from ite3 -- they define the measurement, and changing
them would make the local misfit incomparable with the Anvil runs.
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent          # ite3_gpu/


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


# ---------------------------------------------------------------------------
# measurement parameters -- IDENTICAL to ite3. Do not drift these casually:
# the misfit series is only comparable within a fixed band/window definition.
# ---------------------------------------------------------------------------
# Period bands as (label, minf, maxf, vg, vsmin, vsmax). Each band gets its
# own per-band output subdirectory in both the cc dir and the per-source
# SYNP/DATAP dirs.
#   `vg`              -- rough Rayleigh group velocity (km/s) at the band's
#                        shortest period. Sets the minimum source-receiver
#                        distance via min_dist = vg * MINPERIODNUM / maxf.
#   `vsmin`, `vsmax`  -- group-velocity bounds (km/s) used to define the
#                        signal window for the SNR calculation.
BANDS: list[tuple[str, float, float, float, float, float]] = [
    ("2_6",  1/6, 1/2,  1.0, 0.8, 3.0),    # stage-2 (fine 2s mesh, re-enabled 2026-08-10)
    ("3_10",  1/10, 1/3,  1.0, 0.8, 3.0),
    ("6_18",  1/18, 1/6,  2.2, 1.8, 3.5),
    #("12_30", 1/30, 1/12, 2.5, 2.2, 3.8),
]

SAMPLING_RATE = 20      # resample rate (Hz), stage 0
T1_B = -1               # trim start relative to trace starttime (s)
T2_A = 70               # trim end (s)
MINPERIODNUM = 3        # periods in the minimum-distance gate
SNR_CUTOFF = 3          # a pair survives a band when max(snr_peak, snr_energy) >= this
EPS = 1e-10
NOISE_OFFSET_SECONDS = 10.0

# ---------------------------------------------------------------------------
# paths -- all overridable from the environment
# ---------------------------------------------------------------------------
# CALADJ_DATA : observed data root. Must contain the per-band correlation dirs
#               (3_10/, 6_18/) and STATIONS. ~12 GB staged from Anvil; the 11 GB
#               `cc/` broadband tree is stage-0 output and is NOT needed here.
BASEDIR = _env_path('CALADJ_DATA', _ROOT / 'obs_data')

BASESACDIR = _env_path('CALADJ_SACDIR', BASEDIR / 'final_sac')
CCDIR      = _env_path('CALADJ_CCDIR',  BASEDIR / 'final_sac_wb_new')
STATIONFILE = _env_path('CALADJ_STATIONS', BASEDIR / 'STATIONS')

# DATADIR : the per-source SPECFEM workdirs (<source>_Z/*.semd) to measure against.
#           The loop points this at the current model's forward result; the line
#           search points it at a trial candidate's synthetics.
DATADIR = _env_path('CALADJ_DATADIR', _ROOT / 'work' / 'forward' / 'result')

# ADJDIR  : where 4_prep_adj.py writes the SPECFEM-ready adjoint sources.
#           Must equal ADJSRC in config.sh.
ADJDIR = _env_path('CALADJ_ADJDIR', _ROOT / 'work' / 'adjoint_sources')

# WINDIR  : the frozen pyflex window cache.
#           IMPORTANT: ite2's windows do NOT transfer to ite3's model (the
#           corrected M0 has completely different arrival times), and neither do
#           Anvil-ite3's windows transfer automatically here -- but they SHOULD,
#           because the physics is identical; only the mesh partitioning changed.
#           If you staged windows/ from Anvil, point this at them and use
#           --reuse. If the cache is empty, the first run MUST pick windows
#           (run_caladj_local.sh detects this and drops --reuse automatically).
WINDIR = _env_path('CALADJ_WINDIR', BASEDIR / 'windows' / 'c1_r1')

# SEMD_BAND : the channel band code SPECFEM writes, derived from DT
#             (write_seismograms.f90:band_instrument_code).
#               DT=0.02  -> 'BX'     DT=0.005 -> 'HX'
#             ite2 ran DT=0.005 and produced *.HXZ.semd, which is why the stock
#             1_process_data.py hardcodes an HXZ glob. ite3 runs DT=0.02 and
#             produces *.BXZ.semd, so that glob silently matches NOTHING and the
#             pipeline reports a misfit of 0 over 0 measurements. This port makes
#             the code accept whichever is present; the value here is only used
#             as a hint in log messages.
SEMD_BAND = os.environ.get('CALADJ_SEMD_BAND', 'BX')

# Directory for per-rank and combined log files (relative to CWD).
LOG_DIR = 'log'
