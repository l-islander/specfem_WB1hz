"""Compute pyadjoint adjoint sources from pre-measured pyflex windows.

Consumes the `.win` files written by 2_measure_windows.py together with the
matched OBS and SYN traces. Default is the raw bandpassed traces (matches
2_measure_windows.py's default):

    workdir/WIN_<label>/{sta}.{net_sta}..<BAND>Z.win
    workdir/DATAP_<label>/<pair>_mean.sac
    workdir/SYNP_<label>/<net>.<sta>.HXZ.sac

Pass --use-whitened to consume the *.sac.nor / *.HXZ.sac.nor variants and
WINNOR_<label>/ instead.

Usage (MPI):
    mpiexec -n <N> python 3_calculateadj.py [--use-whitened]
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import obspy
import pyadjoint
from mpi4py import MPI

import config
from mpi_progress import mpi_tqdm
from utility import (
    Pair,
    collect_window_pairs,
    decimate_pairs_by_cell,
    detect_channel,
    load_stations,
    parse_receiver,
    read_semd,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def _mode_names(whitened: bool, channel: str = "HX") -> tuple[str, str, str]:
    """Return (obs_glob, syn_suffix, win_prefix) for the run mode.

    Must match _mode_names in 2_measure_windows.py. `channel` = the forward
    band+instrument code (e.g. "BX"), so the SYN suffix follows the forward run.
    """
    if whitened:
        return "*.sac.nor", f".{channel}Z.sac.nor", "WINNOR"
    return "*.sac", f".{channel}Z.sac", "WIN"

LOG_DIR = Path(config.LOG_DIR)
if rank == 0:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
comm.Barrier()

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_DIR / f"3_calculateadj.rank{rank:03d}.log"),
    ],
    format=f"%(asctime)s - rank{rank:03d} - %(levelname)s - %(message)s",
)


def read_winfile(winfile: Path) -> list[tuple[float, float]]:
    """Return [(left_samples, right_samples), ...] from a .win file."""
    windows: list[tuple[float, float]] = []
    lines = winfile.read_text().splitlines()
    for lineno, line in enumerate(lines[1:], start=2):
        items = line.strip().split()
        if not items:
            continue
        if len(items) < 3:
            logging.warning("Malformed window line %s in %s: %r", lineno, winfile, line)
            continue
        windows.append((float(items[1]), float(items[2])))
    return windows


def format_adjinfo(adj_src) -> str:
    """Serialise one pair's measurement as whitespace-separated fields.

        <tshift> <misfit> <nwin> <tshift_1> <misfit_1> ... <tshift_n> <misfit_n>

    `misfit` is pyadjoint's total over every window of the pair, so collapsing
    the per-window shifts to `window_stats[0]["tshift"]` (as this used to do)
    made the two leading fields describe different things whenever a pair
    carried more than one window: field 0 came from window 0 alone while field 1
    summed them all. Field 0 is now the misfit-weighted mean shift, which
    weights each window by how much it actually drives the gradient
    (cc_traveltime's per-window misfit goes as tshift^2, so a cycle-skipped
    window stays visible instead of being averaged away by a well-fit one) and
    reduces exactly to `tshift` for the single-window case that covers most
    pairs. The per-window values follow so nothing is lost.

    Readers that split on whitespace and take fields 0 and 1 -- read_misfit.py
    and plotwindowandadj.py -- are unaffected by the appended fields.
    """
    stats = adj_src.window_stats
    shifts = [s["tshift"] for s in stats]
    misfits = [abs(s.get("misfit_dt", 0.0)) for s in stats]

    wsum = sum(misfits)
    if wsum > 0:
        tshift = sum(t * m for t, m in zip(shifts, misfits)) / wsum
    else:
        # Every window measured a zero shift, so the mean is zero either way.
        tshift = sum(shifts) / len(shifts) if shifts else 0.0

    fields = [f"{tshift}", f"{adj_src.misfit}", f"{len(stats)}"]
    for t, m in zip(shifts, misfits):
        fields += [f"{t}", f"{m}"]
    return " ".join(fields)


def find_window_dir(workdir: Path, label: str, win_prefix: str) -> Path | None:
    windir = workdir / f"{win_prefix}_{label}"
    return windir if windir.exists() else None


def calculate_adj(obs_data, synth_data, windows, pyadjoint_cfg):
    return pyadjoint.calculate_adjoint_source(
        config=pyadjoint_cfg, observed=obs_data, synthetic=synth_data, windows=windows,
    )


def process_pair(
    obsfile: Path,
    syndir: Path,
    windir: Path,
    optdir: Path,
    times: np.ndarray,
    pyadjoint_cfg,
    *,
    label: str,
    source: str,
    t1_b: float,
    channel: str = "HX",
    syn_suffix: str = ".HXZ.sac",
    kept_pairs: set[Pair] | None = None,
) -> None:
    # Receiver = whichever side of the cc-pair filename isn't the workdir source.
    net_sta, sta = parse_receiver(obsfile.name, source)

    if kept_pairs is not None:
        pair_key = tuple(sorted((source, f"{net_sta}.{sta}")))
        if pair_key not in kept_pairs:
            return

    synfile = syndir / f"{net_sta}.{sta}{syn_suffix}"
    if not synfile.exists():
        logging.warning("Synth not found [%s] %s.%s: %s", label, net_sta, sta, synfile)
        return

    # `channel` = forward band+instrument code. The .adj name written here is what
    # 4_prep_adj hands to xspecfem3D, so it MUST match the solver's channel (BXZ at
    # DT=0.02); winfile is read with the same code that 2_measure_windows wrote.
    winfile = windir / f"{sta}.{net_sta}..{channel}Z.win"
    if not winfile.exists():
        logging.info("No .win file [%s] %s.%s: %s", label, net_sta, sta, winfile)
        return

    adjfilename = optdir / f"{net_sta}.{sta}.{channel}Z.adj"
    opt_winfile = optdir / winfile.name
    if adjfilename.exists():
        if not opt_winfile.exists():
            shutil.copy2(winfile, opt_winfile)
        return

    obs_data = obspy.read(str(obsfile), format="SAC")[0]
    synth_data = obspy.read(str(synfile), format="SAC")[0]
    delta = obs_data.stats.sac.delta

    windows_t = read_winfile(winfile)
    if not windows_t:
        logging.info("Empty .win [%s] %s.%s", label, net_sta, sta)
        return
    windows = [(w[0] * delta, w[1] * delta) for w in windows_t]

    adj_src = calculate_adj(obs_data, synth_data, windows, pyadjoint_cfg)

    adjinfofilename = optdir / f"{net_sta}.{sta}.{channel}Z.adjinfo"
    # Trailing newline is REQUIRED: without it, cat-ing many .adjinfo files
    # merges the last line of one into the first of the next, corrupting any
    # whitespace-split reader. (The shell misfit sums read line-by-line and
    # tolerate its absence, but keep the file well-formed so a future
    # cat|awk aggregator can't silently break.)
    adjinfofilename.write_text(format_adjinfo(adj_src) + "\n")

    adjdt = adj_src.dt
    adjdata = adj_src.adjoint_source[::-1]
    adjnpts = len(adjdata)
    adjtime = t1_b + np.arange(adjnpts) * adjdt
    adjdata_interp = np.interp(times, adjtime, adjdata)
    np.savetxt(adjfilename, np.column_stack((times, adjdata_interp)), fmt="%.10f")
    shutil.copy2(winfile, opt_winfile)


def process_band(
    workdir: Path,
    label: str,
    minp: float,
    maxp: float,
    times: np.ndarray,
    *,
    t1_b: float,
    tick=None,
    whitened: bool = False,
    kept_pairs: set[Pair] | None = None,
) -> None:
    channel = detect_channel(workdir)
    obs_glob, syn_suffix, win_prefix = _mode_names(whitened, channel)
    obsdir = workdir / f"DATAP_{label}"
    syndir = workdir / f"SYNP_{label}"
    windir = find_window_dir(workdir, label, win_prefix)
    optdir = workdir / f"OPT_{label}"

    if not obsdir.exists():
        logging.warning("No obs dir [%s]: %s", label, obsdir)
        return
    if not syndir.exists():
        logging.warning("No syn dir [%s]: %s", label, syndir)
        return
    if windir is None:
        logging.info("No window dir [%s]: %s (run 2_measure_windows.py first)",
                     label, workdir / f"{win_prefix}_{label}")
        return

    optdir.mkdir(parents=True, exist_ok=True)

    source = workdir.name.split("_")[0]

    pyadjoint_cfg = pyadjoint.get_config(
        min_period=minp, max_period=maxp, adjsrc_type="cc_traveltime",
    )

    for obsfile in sorted(obsdir.glob(obs_glob)):
        try:
            process_pair(
                obsfile, syndir, windir, optdir, times, pyadjoint_cfg,
                label=label, source=source,
                t1_b=t1_b, channel=channel, syn_suffix=syn_suffix,
                kept_pairs=kept_pairs,
            )
        except Exception:
            logging.exception("Failed adj [%s] %s", label, obsfile)
        finally:
            if tick is not None:
                tick()


def process_workdir(
    workdir: Path,
    bands: list[tuple[str, float, float]],
    *,
    t1_b: float,
    tick=None,
    whitened: bool = False,
    kept_pairs_by_band: dict[str, set[Pair]] | None = None,
) -> None:
    logging.info("Processing %s", workdir.name)

    orisyndir = workdir / "SYN"
    sample_semd = next(iter(orisyndir.glob("*.semd")), None)
    if sample_semd is None:
        logging.warning("No .semd in %s, skipping", orisyndir)
        return
    times, _ = read_semd(sample_semd)          # ascii or binary (SEISMO_FORMAT=bin)

    for label, minf, maxf, *_ in bands:
        minp = 1.0 / maxf
        maxp = 1.0 / minf
        process_band(workdir, label, minp, maxp, times,
                     t1_b=t1_b, tick=tick, whitened=whitened,
                     kept_pairs=kept_pairs_by_band.get(label) if kept_pairs_by_band else None)


def clean_output_dirs(workdirs: list[Path], bands: list[tuple[str, float, float]]) -> None:
    """Remove every OPT_<band>/ under the given workdirs.

    Adjoint output dirs are per-iteration scratch — they must be wiped at the
    start of each run so the skip-if-exists check in `process_pair` doesn't
    silently keep stale .adj files when windows or synthetics have changed.
    """
    for wd in workdirs:
        for label, *_ in bands:
            sub = wd / f"OPT_{label}"
            if sub.exists():
                shutil.rmtree(sub)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-whitened", action="store_true",
        help="Read whitened traces (*.sac.nor / *.HXZ.sac.nor) and the "
             "WINNOR_<label>/ windows. Default is raw (*.sac / *.HXZ.sac, "
             "WIN_<label>/), matching 2_measure_windows.py's default.",
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Skip the OPT_<band> wipe (default: each rank wipes its slice "
             "before running so stale .adj files cannot survive across reruns).",
    )
    parser.add_argument(
        "--exclude-same-group", type=Path, default=None,
        help="Station-list file (one NET.STA per line). Pairs with BOTH "
             "endpoints in the list are excluded from decimation, measurement "
             "and adjoint sources. Used to drop the ZO nodal line's "
             "same-line pairs, whose dt systematically contradicts every "
             "independent geometry crossing the same volume (see "
             "figures/transect_diagnosis.png).",
    )
    parser.add_argument(
        "--cluster-deg", type=float, nargs="+", default=[0.02],
        help="Spatial-decimation cell size in degrees applied to the pair set "
             "before computing adjoint sources. Provide one value (applies to "
             "every band, whatever config.BANDS holds) or one per band in "
             "config.BANDS order. 0 disables decimation. Defaults to 0.02 deg "
             "(~2 km here) for every band, so changing config.BANDS needs no "
             "change here. Matches plot_path_density.py.",
    )
    parser.add_argument(
        "--min-pairs", type=int, default=3,
        help="Skip any workdir that owns fewer than this many kept pairs "
             "(unique across bands, after --cluster-deg). 0 disables. Default "
             "3 drops the long tail of low-yield source workdirs (~37%% fewer "
             "SPECFEM adjoint runs for ~11%% pair loss in this dataset).",
    )
    args = parser.parse_args()
    whitened = args.use_whitened

    syndirlist = config.DATADIR
    obs_glob, _, win_prefix = _mode_names(whitened)

    band_labels = [b[0] for b in config.BANDS]
    if len(args.cluster_deg) == 1:
        cluster_per_band = {b: args.cluster_deg[0] for b in band_labels}
    elif len(args.cluster_deg) == len(band_labels):
        cluster_per_band = dict(zip(band_labels, args.cluster_deg))
    else:
        raise SystemExit(
            f"--cluster-deg expects 1 or {len(band_labels)} values, "
            f"got {len(args.cluster_deg)}"
        )
    decimating = any(v > 0 for v in cluster_per_band.values())

    exclude_group: set[str] = set()
    if args.exclude_same_group:
        exclude_group = {ln.strip() for ln in open(args.exclude_same_group)
                         if ln.strip() and not ln.startswith("#")}
        if rank == 0:
            logging.info("exclude-same-group: %d stations from %s",
                         len(exclude_group), args.exclude_same_group)

    if rank == 0:
        workdirs = sorted(syndirlist.glob("*_Z"))
        logging.info("Found %d workdirs; distributing across %d ranks. mode=%s",
                     len(workdirs), size, "whitened" if whitened else "raw")

        if decimating:
            stations = load_stations(config.STATIONFILE)
            raw_pairs = collect_window_pairs(workdirs, band_labels, win_prefix=win_prefix)
            kept_pairs_by_band: dict[str, set[Pair]] = {}
            for band in band_labels:
                pset = raw_pairs.get(band, set())
                if exclude_group:
                    n0 = len(pset)
                    pset = {pr for pr in pset
                            if not (pr[0] in exclude_group and pr[1] in exclude_group)}
                    if n0 != len(pset):
                        logging.info("exclude-same-group [%s]: %d -> %d pairs",
                                     band, n0, len(pset))
                cluster = cluster_per_band[band]
                if cluster > 0 and pset:
                    kept = decimate_pairs_by_cell(pset, stations, cluster)
                else:
                    kept = pset
                kept_pairs_by_band[band] = kept
                logging.info(
                    "decimation [%s]: %d -> %d pairs (cluster=%.4g deg)",
                    band, len(pset), len(kept), cluster,
                )
        else:
            kept_pairs_by_band = None

        # Drop workdirs that own fewer than --min-pairs kept pairs.
        if args.min_pairs > 0:
            if kept_pairs_by_band is None:
                all_kept = set().union(*collect_window_pairs(
                    workdirs, band_labels, win_prefix=win_prefix
                ).values())
            else:
                all_kept = set().union(*kept_pairs_by_band.values())
            pairs_per_wd: dict[Path, set[Pair]] = {wd: set() for wd in workdirs}
            for wd in workdirs:
                src = wd.name.split("_")[0]
                for band in band_labels:
                    bdir = wd / f"{win_prefix}_{band}"
                    if not bdir.is_dir():
                        continue
                    for wf in bdir.glob("*.win"):
                        parts = wf.name.split(".")
                        pair = tuple(sorted((src, f"{parts[1]}.{parts[0]}")))
                        if pair in all_kept:
                            pairs_per_wd[wd].add(pair)
            before = len(workdirs)
            workdirs = [wd for wd in workdirs if len(pairs_per_wd[wd]) >= args.min_pairs]
            logging.info(
                "min-pairs=%d filter: %d -> %d workdirs (dropped %d low-yield)",
                args.min_pairs, before, len(workdirs), before - len(workdirs),
            )
    else:
        workdirs = None
        kept_pairs_by_band = None
    workdirs = comm.bcast(workdirs, root=0)
    kept_pairs_by_band = comm.bcast(kept_pairs_by_band, root=0)

    my_workdirs = workdirs[rank::size]

    if not args.no_clean:
        logging.info("Cleaning OPT_<band> dirs for %d local workdirs...",
                     len(my_workdirs))
        clean_output_dirs(my_workdirs, config.BANDS)
    comm.Barrier()  # ensure all ranks finish cleanup before any starts writing

    local_pairs = sum(
        len(list((wd / f"DATAP_{label}").glob(obs_glob)))
        for wd in my_workdirs
        for label, *_ in config.BANDS
        if (wd / f"DATAP_{label}").exists()
        and (wd / f"SYNP_{label}").exists()
        and find_window_dir(wd, label, win_prefix) is not None
    )
    total_pairs = comm.allreduce(local_pairs, op=MPI.SUM)

    with mpi_tqdm(comm, total=total_pairs, unit="pair", desc="adj",
                  mininterval=2.0) as tick:
        for workdir in my_workdirs:
            try:
                process_workdir(workdir, config.BANDS,
                                t1_b=config.T1_B, tick=tick, whitened=whitened,
                                kept_pairs_by_band=kept_pairs_by_band)
            except Exception:
                logging.exception("Failed processing workdir %s", workdir)

    for h in logging.getLogger().handlers:
        h.flush()
    comm.Barrier()

    if rank == 0:
        logging.info("All ranks finished. Combining per-rank logs.")
        for h in logging.getLogger().handlers:
            h.flush()

        rank_logs = sorted(LOG_DIR.glob("3_calculateadj.rank*.log"))
        lines: list[str] = []
        for p in rank_logs:
            lines.extend(p.read_text().splitlines(keepends=True))
        lines.sort()
        (LOG_DIR / "3_calculateadj.log").write_text("".join(lines))


if __name__ == "__main__":
    main()
