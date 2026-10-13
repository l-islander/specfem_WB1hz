"""Shared helpers used across the specfem_process workflow scripts.

Functions / constants needed by the per-stage workflow (0_newprocess,
1_process_data, 2_measure_windows, 3_calculateadj, compare_syn, window_stats)
live here so those scripts don't have to import the much larger sac_reorder
module (which is a standalone reorder tool). Spectral whitening
(old_2_spectrum_whitening.py) is no longer in the active workflow but still
uses signal_noise_indices if revived.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np


Pair = tuple[str, str]


def detect_channel(workdir, default: str = "HX") -> str:
    """Band+instrument code (e.g. ``"BX"``, ``"HX"``) taken from the FORWARD
    synthetics in ``workdir`` -- so the whole pipeline follows the forward run.

    SPECFEM3D names each seismogram channel from the time step (SEED convention,
    ``band_instrument_code`` in write_seismograms.f90): 0.0125 < DT <= 0.1 -> "BX",
    0.004 < DT <= 0.0125 -> "HX", etc. The raw synthetics (``<net>.<sta>.BXZ.semd``)
    therefore already encode the correct code, so we read it straight off a filename
    instead of re-deriving it from DT (which would risk disagreeing with what the
    solver actually wrote). The adjoint sources handed back to xspecfem3D MUST carry
    this same code, and windows/measurements follow it for internal consistency.

    Looks in ``workdir/SYN/`` first (where 1_process_data moves the .semd), then the
    workdir top level (a fresh run before that move). Falls back to ``default`` if no
    synthetic is present (e.g. a stage run on a workdir whose SYN was archived), which
    preserves the historical "HX" behaviour.
    """
    workdir = Path(workdir)
    for pat in ("SYN/*.semd", "*.semd"):
        for f in sorted(workdir.glob(pat)):
            m = re.search(r"\.([A-Z]{2})[ENZ]\.semd$", f.name)
            if m:
                return m.group(1)
    logging.warning("detect_channel: no synthetics under %s; falling back to %r",
                    workdir, default)
    return default


_SEMD_TEXT_BYTES = frozenset(b" \t\r\n+-.0123456789EeDd")


def find_seismo_meta(semd_path):
    """Locate the time-axis metadata for a trace: the runner's ``seismo_meta.txt``
    next to the traces (or one level up, for traces moved into ``SYN/``), else the
    solver log in the same places. Symlinks are resolved so a farm's ``SYN`` link
    finds the sidecar in the real result dir. Returns a dict with ``dt``,
    ``start_time`` (floats) and ``nstep`` (int), or None."""
    p = Path(semd_path).resolve()
    dirs = (p.parent, p.parent.parent)
    for d in dirs:
        f = d / "seismo_meta.txt"
        if f.is_file():
            kv = dict(l.strip().split("=", 1) for l in f.read_text().splitlines() if "=" in l)
            if kv.get("start_time", "NA") != "NA":
                return {"dt": float(kv["dt"]), "start_time": float(kv["start_time"]),
                        "nstep": int(kv.get("nstep", 0)), "source": str(f)}
    for d in dirs:
        f = d / "output_solver.txt"
        if f.is_file():
            dt = st = ns = None
            for line in f.read_text(errors="replace").splitlines():
                s = line.strip()
                if s.startswith("time step:"):            dt = float(s.split(":")[1].split()[0])
                elif s.startswith("start time:"):         st = float(s.split(":")[1].split()[0])
                elif s.startswith("number of time steps:"): ns = int(s.split(":")[1].split()[0])
                if dt is not None and st is not None and ns is not None:
                    return {"dt": dt, "start_time": st, "nstep": ns, "source": str(f)}
    return None


def read_semd(path):
    """Read one SPECFEM seismogram, whatever format it was written in.

    * ascii  (USE_BINARY_FOR_SEISMOGRAMS=.false.): two columns ``t u``.
    * binary (USE_BINARY_FOR_SEISMOGRAMS=.true., config.sh SEISMO_FORMAT=bin):
      the bare float32 trace, same file name; the time axis is rebuilt from
      :func:`find_seismo_meta` (dt + solver start time).

    Returns ``(times, data)`` as float64 arrays, bitwise identical to the
    historical ``np.loadtxt`` result for ascii files (verified 2026-09-11).
    """
    p = Path(path)
    with open(p, "rb") as fh:
        raw = fh.read()
    head = raw[:64]
    is_text = len(head) > 0 and all(c in _SEMD_TEXT_BYTES for c in head) and (b"\n" in raw[:4096])
    if is_text:
        a = np.fromstring(raw.replace(b"D", b"E"), sep=" ", dtype=np.float64)
        if a.size % 2:
            raise ValueError(f"{p}: ascii seismogram has an odd number of tokens ({a.size})")
        a = a.reshape(-1, 2)
        return a[:, 0], a[:, 1]
    data = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    meta = find_seismo_meta(p)
    if meta is None:
        raise RuntimeError(f"{p}: binary seismogram but no seismo_meta.txt / output_solver.txt "
                           "found beside it to rebuild the time axis")
    times = meta["start_time"] + np.arange(data.size, dtype=np.float64) * meta["dt"]
    return times, data


def signal_noise_indices(
    n: int, delta: float, dist: float, t1_b: float,
    vsmin: float, vsmax: float,
) -> tuple[int, int, int, int] | None:
    """Return (i0, i1, ns0, ns1) for signal/noise time-domain windows.

    Signal window: surface-wave arrivals between vsmax (earliest, at
    ``dist/vsmax + t1_b``) and vsmin (latest, at ``dist/vsmin + t1_b``).
    Noise window matches signal length and is placed before, after, or
    whichever side has the most room. Returns None if window geometry
    is invalid (signal window degenerates or no room for noise).
    """
    tmin = dist / vsmax + t1_b
    tmax = dist / vsmin + t1_b
    i0 = max(0, int(np.floor(tmin / delta)))
    i1 = min(n, int(np.ceil(tmax / delta)))
    if i1 <= i0:
        return None
    n_sig = i1 - i0
    pre, post = i0, n - i1
    if pre >= n_sig:
        ns0, ns1 = i0 - n_sig, i0
    elif post >= n_sig:
        ns0, ns1 = i1, i1 + n_sig
    elif pre >= post and pre > 0:
        ns0, ns1 = 0, i0
    elif post > 0:
        ns0, ns1 = i1, n
    else:
        return None
    return i0, i1, ns0, ns1


def compute_time_snr(
    data: np.ndarray, delta: float, dist: float, t1_b: float,
    vsmin: float, vsmax: float, eps: float = 1e-10,
) -> tuple[float, float]:
    """Wide-band SNR on a time-domain trace: (peak ratio, energy ratio).

    Uses :func:`signal_noise_indices` to pick matched signal/noise windows
    given the band's per-band group-velocity bounds, then returns
        peak   = max(|signal|) / max(|noise|)
        energy = ||signal||_2 / ||noise||_2
    Returns (0, 0) if the signal/noise geometry is invalid.
    """
    idx = signal_noise_indices(data.size, delta, dist, t1_b, vsmin, vsmax)
    if idx is None:
        return 0.0, 0.0
    i0, i1, ns0, ns1 = idx
    sig = np.asarray(data[i0:i1])
    noise = np.asarray(data[ns0:ns1])
    if sig.size == 0 or noise.size == 0:
        return 0.0, 0.0
    snr_peak = float(np.max(np.abs(sig)) / (np.max(np.abs(noise)) + eps))
    snr_energy = float(np.linalg.norm(sig) / (np.linalg.norm(noise) + eps))
    return snr_peak, snr_energy


# Networks whose stations are nodal / short-period instruments. Used to
# classify a (src, rec) pair as BB / NB / NN in window_stats and to bias
# the greedy set-cover ordering toward broadband stations first.
NODAL_NETWORKS = {"3A", "Z4", "YM", "XD", "ZO", "WB"}


def is_nodal(net_dot_sta: str) -> bool:
    """True iff the station's network code is in NODAL_NETWORKS."""
    return net_dot_sta.split(".")[0] in NODAL_NETWORKS


def is_bb(station: str) -> bool:
    """True iff the station's network code is NOT in NODAL_NETWORKS."""
    return station.split(".")[0] not in NODAL_NETWORKS


def canonical_pair(sta1: str, sta2: str) -> Pair:
    """Stable unordered key used for station pairs."""
    return tuple(sorted((sta1, sta2)))


def load_stations(stationfile) -> dict[str, tuple[float, float]]:
    """Parse a SPECFEM STATIONS file into {NET.STA: (lat, lon)}.

    Skips lines that do not have the four expected columns
    (sta_id, network, lat, lon, [...]).
    """
    from pathlib import Path
    info: dict[str, tuple[float, float]] = {}
    for line in Path(stationfile).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            info[f"{parts[1]}.{parts[0]}"] = (float(parts[2]), float(parts[3]))
    return info


def collect_window_pairs(
    workdirs, band_labels: list[str], win_prefix: str = "WIN",
) -> dict[str, set[Pair]]:
    """Walk WIN_<band>/*.win files and return {band: {(sta_a, sta_b), ...}}.

    Each entry is a station pair (NET.STA, NET.STA), sorted lexicographically.
    Only pairs with >=1 measured window are included.
    """
    from collections import defaultdict
    from pathlib import Path
    pairs: dict[str, set[Pair]] = defaultdict(set)
    for wd in workdirs:
        wd = Path(wd)
        src = wd.name.split("_")[0]
        for band in band_labels:
            windir = wd / f"{win_prefix}_{band}"
            if not windir.is_dir():
                continue
            for wf in windir.glob("*.win"):
                try:
                    toks = wf.read_text().splitlines()[0].split()
                    nw = int(toks[3]) if len(toks) >= 4 and toks[2] == "win_num" else 0
                except Exception:
                    nw = 0
                if nw <= 0:
                    continue
                parts = wf.name.split(".")
                rec = f"{parts[1]}.{parts[0]}"
                pairs[band].add(tuple(sorted((src, rec))))
    return pairs


def decimate_pairs_by_cell(
    pairs: set[Pair],
    stations: dict[str, tuple[float, float]],
    cluster_deg: float,
    *,
    seed: int = 42,
) -> set[Pair]:
    """Drop redundant station pairs that follow nearly-identical paths.

    Groups stations into ``cluster_deg`` x ``cluster_deg`` lat/lon cells, then
    keeps one representative pair per unordered (cell_A, cell_B) combination.
    Pairs where either station is missing from ``stations`` are dropped.

    Deterministic for a given seed: the representative is the lexicographically
    smallest pair in the group (no RNG, no float comparison surprises).
    """
    from collections import defaultdict

    def cell_of(sta: str) -> tuple[int, int]:
        lat, lon = stations[sta]
        return (
            int(round(lat / cluster_deg)),
            int(round(lon / cluster_deg)),
        )

    groups: dict[tuple, list[Pair]] = defaultdict(list)
    for a, b in pairs:
        if a not in stations or b not in stations:
            continue
        key = tuple(sorted((cell_of(a), cell_of(b))))
        groups[key].append((a, b))

    kept: set[Pair] = set()
    for grp in groups.values():
        kept.add(min(grp))  # lexicographic — deterministic without an RNG
    _ = seed  # kept for API stability if a future change wants randomization
    return kept


def parse_receiver(obs_name: str, source: str) -> tuple[str, str]:
    """Return (rec_net, rec_sta) for an obs cc filename given the workdir source.

    Filenames follow ``<net1>_<sta1>_<net2>_<sta2>_mean.sac[.nor]``. After
    sac_reorder regroups files by greedy set cover, the workdir's source can
    sit at *either* (net1, sta1) or (net2, sta2). The receiver is whichever
    side is not the source. ``source`` is "NET.STA" — i.e.
    ``workdir.name.split("_")[0]``.
    """
    base = obs_name.removesuffix(".sac.nor").removesuffix(".sac")
    parts = base.split("_")
    src_net, src_sta = source.split(".", 1)
    if (parts[0], parts[1]) == (src_net, src_sta):
        return parts[2], parts[3]
    return parts[0], parts[1]


def station_cover_order(pairs: list[Pair]) -> list[tuple[str, int]]:
    """Greedy set-cover ordering of stations over (unweighted) pairs.

    Broadband stations (network not in :data:`NODAL_NETWORKS`) are selected
    first; nodal stations only cover the pairs that remain. Each entry is
    ``(station, marginal_pair_count)``.
    """
    return station_cover_order_weighted({canonical_pair(*pair): 1 for pair in pairs})


def station_cover_order_weighted(
    pair_weights: dict[Pair, int],
) -> list[tuple[str, int]]:
    """Greedy station-cover ordering weighted by per-pair counts."""

    def greedy(pool: dict[Pair, int], candidates: set[str]) -> list[tuple[str, int]]:
        remaining = dict(pool)
        result: list[tuple[str, int]] = []
        while remaining:
            counts: dict[str, int] = {}
            for (a, b), weight in remaining.items():
                if a in candidates:
                    counts[a] = counts.get(a, 0) + weight
                if b in candidates and b != a:
                    counts[b] = counts.get(b, 0) + weight
            if not counts:
                break
            station, claimed = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
            if claimed == 0:
                break
            result.append((station, claimed))
            remaining = {p: w for p, w in remaining.items() if station not in p}
        return result

    clean_weights = {
        canonical_pair(*pair): weight
        for pair, weight in pair_weights.items()
        if weight > 0
    }
    bb_candidates = {s for pair in clean_weights for s in pair if is_bb(s)}
    nodal_candidates = {s for pair in clean_weights for s in pair if not is_bb(s)}

    bb_pool = {
        p: w for p, w in clean_weights.items() if is_bb(p[0]) or is_bb(p[1])
    }
    nodal_pool = {
        p: w for p, w in clean_weights.items() if not (is_bb(p[0]) or is_bb(p[1]))
    }

    return greedy(bb_pool, bb_candidates) + greedy(nodal_pool, nodal_candidates)
