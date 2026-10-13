"""Single-command pipeline: ZZ -> reordered cc/ + reordered band folders.

Replaces the 3-step flow (0_preocesscc.py -> sac_reorder.py -> 0_preocesscc.py --bands-only)
with one MPI run that:
  1. Processes each ZZ cc file (integrate, sign-flip, headers) and filters per band,
     keeping survivors in memory.
  2. Gathers the survivor pair set on rank 0, computes greedy station cover, and
     broadcasts the pair->station mapping.
  3. Each rank writes its in-memory traces directly to the reordered layout:
        <out_dir>/cc/<station>/<pair>_mean.sac
        <out_dir>/<band>/<station>/<pair>_mean.sac

Nothing is written to disk before the reorder mapping is known, so there are no
intermediate files to clean up.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import obspy as obp
from mpi4py import MPI

import config
from mpi_progress import mpi_tqdm
from utility import compute_time_snr, station_cover_order

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

LOG_DIR = Path(config.LOG_DIR)
if rank == 0:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
comm.Barrier()

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(LOG_DIR / f"0_newprocess.rank{rank:03d}.log")],
    format=f"%(asctime)s - rank{rank:03d} - %(levelname)s - %(message)s",
)


def parse_pair_stem(stem: str) -> tuple[str, str, str, str]:
    parts = stem.split("_")
    return parts[0], parts[1], parts[2], parts[3]


def make_base_trace(ccfile: Path, t1_b: float, t2_a: float, sampling_rate: int) -> obp.Trace:
    """Read GF-form SAC produced by h5_to_sac.py: set headers, resample, trim.

    h5_to_sac.py now integrates and antisymmetric-folds the two-sided velocity-
    velocity CC upstream, so the trace on disk already approximates the
    displacement Green's function. No further integration or sign flip here.
    """
    net1, sta1, net2, sta2 = parse_pair_stem(ccfile.stem)
    base = obp.read(str(ccfile))[0]
    base.detrend("demean")
    base.detrend("linear")
    base.taper(max_percentage=0.05, type="cosine", side="right")

    base.stats.network = net1
    base.stats.station = sta1
    base.stats.channel = "Z"
    base.stats.sac.evdp = 0.0
    base.stats.sac.b = t1_b
    base.stats.sac.e = t2_a
    base.stats.sac.kevnm = f"{net2}.{sta2}"[:16]
    base.stats.sac.kuser0 = net2[:8]
    base.stats.sac.kuser1 = sta2[:8]

    base.resample(sampling_rate)
    base.trim(base.stats.starttime + t1_b, base.stats.starttime + t2_a, pad=True, fill_value=0)
    return base


def filter_band(
    base: obp.Trace, minf: float, maxf: float, sampling_rate: int, t1_b: float, t2_a: float
) -> obp.Trace:
    tr = base.copy()
    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(max_percentage=0.05, type="cosine", side="right")
    tr.filter("bandpass", freqmin=minf, freqmax=maxf, corners=4, zerophase=True)
    tr.trim(tr.stats.starttime, tr.stats.starttime + (t2_a - t1_b), pad=True, fill_value=0)
    tr.normalize()
    return tr


def process_one_file(
    ccfile: Path,
    bands: list[tuple[str, float, float, float, float, float]],
    *,
    sampling_rate: int,
    t1_b: float,
    t2_a: float,
    minperiodnum: int,
    skip_snr: bool,
    skip_dist: bool,
) -> tuple[tuple[str, str], obp.Trace, dict[str, obp.Trace]] | None:
    """Return (pair_key, raw_base_trace, {band_label: filtered_trace}) for survivors.

    pair_key is the sorted (sta1, sta2) tuple used for set-cover.
    Bands that fail distance or SNR gates are omitted from the dict.
    A pair survives if at least one band passes; otherwise None is returned.
    """
    net1, sta1, net2, sta2 = parse_pair_stem(ccfile.stem)
    sta_a, sta_b = f"{net1}.{sta1}", f"{net2}.{sta2}"
    pair_key = tuple(sorted((sta_a, sta_b)))

    base = make_base_trace(ccfile, t1_b, t2_a, sampling_rate)
    distance = float(base.stats.sac.dist)

    band_traces: dict[str, obp.Trace] = {}
    for label, minf, maxf, vg, vsmin, vsmax in bands:
        if not skip_dist and distance < vg * minperiodnum / maxf:
            logging.info("dist too short for %s.%s %s.%s [%s]: %.1f km", net1, sta1, net2, sta2, label, distance)
            continue
        if maxf >= sampling_rate / 2:
            logging.warning("band [%s] freqmax %.3f >= Nyquist %.3f; aliasing possible", label, maxf, sampling_rate / 2)
        tr = filter_band(base, minf, maxf, sampling_rate, t1_b, t2_a)
        snr_peak, snr_energy = compute_time_snr(
            tr.data, 1.0 / tr.stats.sampling_rate, distance, t1_b,
            vsmin, vsmax, eps=config.EPS,
        )
        logging.info("SNR %s.%s %s.%s [%s]: %.3f peak %.3f energy", net1, sta1, net2, sta2, label, snr_peak, snr_energy)
        if not skip_snr and snr_peak < config.SNR_CUTOFF and snr_energy < config.SNR_CUTOFF:
            continue
        band_traces[label] = tr

    if not band_traces:
        return None
    return pair_key, base, band_traces


def assign_stations(
    survivor_pairs: list[tuple[str, str]],
) -> tuple[dict[tuple[str, str], str], list[str]]:
    """Greedy station cover -> ({sorted_pair: assigned_station}, count_rows)."""
    order = station_cover_order(survivor_pairs)
    remaining = set(survivor_pairs)
    mapping: dict[tuple[str, str], str] = {}
    rows: list[str] = []
    for station, claimed in order:
        copied = 0
        for pair in list(remaining):
            if station in pair:
                mapping[pair] = station
                remaining.discard(pair)
                copied += 1
        rows.append(f"{station} count={claimed} copied={copied}")
        if not remaining:
            break
    if remaining:
        logging.warning("%d pairs left unassigned by set-cover", len(remaining))
    return mapping, rows


def write_outputs(
    out_dir: Path,
    local_results: list[tuple[tuple[str, str], obp.Trace, dict[str, obp.Trace]]],
    pair_to_station: dict[tuple[str, str], str],
) -> int:
    written = 0
    for pair_key, base, band_traces in local_results:
        station = pair_to_station.get(pair_key)
        if station is None:
            continue
        net1, sta1 = base.stats.network, base.stats.station
        net2, sta2 = base.stats.sac.kuser0, base.stats.sac.kuser1
        savename = f"{net1}_{sta1}_{net2}_{sta2}_mean.sac"

        cc_target = out_dir / "cc" / station / savename
        cc_target.parent.mkdir(parents=True, exist_ok=True)
        base.write(str(cc_target), format="SAC")
        written += 1

        for label, tr in band_traces.items():
            band_target = out_dir / label / station / savename
            band_target.parent.mkdir(parents=True, exist_ok=True)
            tr.write(str(band_target), format="SAC")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=config.BASESACDIR,
        help="Directory containing per-source ZZ folders of raw stacked CC SAC files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.CCDIR,
        help="Output root for the reordered cc/ + per-band station folders.",
    )
    parser.add_argument("--skip-snr", action="store_true", help="Disable SNR filtering.")
    parser.add_argument(
        "--skip-dist", action="store_true",
        help="Disable the per-band distance gate (vg*MINPERIODNUM/maxf). Use to "
             "test whether short-aperture pairs survive downstream processing.",
    )
    args = parser.parse_args()

    cfg = dict(
        sampling_rate=config.SAMPLING_RATE,
        t1_b=config.T1_B,
        t2_a=config.T2_A,
        minperiodnum=config.MINPERIODNUM,
        skip_snr=args.skip_snr,
        skip_dist=args.skip_dist,
    )

    if rank == 0:
        ccfiles = sorted(args.in_dir.glob("*/*.sac"))
        logging.info("Found %d ZZ cc files; distributing across %d ranks.", len(ccfiles), size)
    else:
        ccfiles = None
    ccfiles = comm.bcast(ccfiles, root=0)

    # --- Phase 1: process locally, keep survivors in memory ---
    local_files = ccfiles[rank::size]
    local_results: list[tuple[tuple[str, str], obp.Trace, dict[str, obp.Trace]]] = []
    with mpi_tqdm(comm, total=len(ccfiles), unit="file", desc="filter") as tick:
        for ccfile in local_files:
            try:
                res = process_one_file(ccfile, config.BANDS, **cfg)
                if res is not None:
                    local_results.append(res)
            except Exception:
                logging.exception("Failed processing %s", ccfile)
            tick()

    # --- Phase 2: gather survivor pairs, rank 0 builds station cover ---
    local_pairs = [pair for pair, _b, _bd in local_results]
    all_pairs_nested = comm.gather(local_pairs, root=0)
    if rank == 0:
        survivor_pairs = sorted({p for sub in all_pairs_nested for p in sub})
        logging.info("Total survivor pairs across ranks: %d", len(survivor_pairs))
        pair_to_station, count_rows = assign_stations(survivor_pairs)
        logging.info("Station-cover assigned %d pairs to %d stations.",
                     len(pair_to_station), len(set(pair_to_station.values())))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        count_file = args.out_dir / "sac_reorder_station_counts.txt"
        count_file.write_text("\n".join(count_rows) + "\n", encoding="utf-8")
        logging.info("Station counts written to %s", count_file)
    else:
        pair_to_station = None
    pair_to_station = comm.bcast(pair_to_station, root=0)

    # --- Phase 3: write reordered outputs ---
    n_written = write_outputs(args.out_dir, local_results, pair_to_station)
    logging.info("Rank wrote %d files.", n_written)

    for h in logging.getLogger().handlers:
        h.flush()
    comm.Barrier()

    if rank == 0:
        rank_logs = sorted(LOG_DIR.glob("0_newprocess.rank*.log"))
        lines: list[str] = []
        for p in rank_logs:
            lines.extend(p.read_text().splitlines(keepends=True))
        lines.sort()
        (LOG_DIR / "0_newprocess.log").write_text("".join(lines))
        logging.info("All ranks finished.")


if __name__ == "__main__":
    main()
