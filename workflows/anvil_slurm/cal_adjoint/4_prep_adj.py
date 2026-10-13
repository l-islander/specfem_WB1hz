"""Prepare SPECFEM adjoint folders from banded Z adjoint sources.

This replaces the old rotate+prep flow for the current Z-only pipeline.
It consumes the banded outputs from 3_calculateadj.py:

    <workdir>/OPT_<label>/{net}.{sta}.HXZ.adj

Matching adjoint files are summed across the selected period bands and written
to a SPECFEM-ready folder named after each workdir, then copied to config.ADJDIR:

    <workdir>/<workdir.name>/
    <config.ADJDIR>/<workdir.name>/
        SEM/
        STATIONS_ADJOINT
        FORCESOLUTION

Receivers whose merged adjoint source is identically null are dropped from both
SEM/ and STATIONS_ADJOINT (see --min-amp); they would otherwise cost a SPECFEM
adjoint receiver while contributing nothing to the gradient.

Usage:
    python 4_prep_adj.py
    python 4_prep_adj.py F:/wb_data/ite1_30_60/UW.ALVY_Z
    python 4_prep_adj.py --bands 3_10 6_18

MPI is optional:
    mpiexec -n <N> python 4_prep_adj.py
"""
from __future__ import annotations

import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from mpi4py import MPI

import config
from mpi_progress import mpi_tqdm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

LOG_DIR = Path(config.LOG_DIR)
if rank == 0:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
comm.Barrier()

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_DIR / f"4_prep_adj.rank{rank:03d}.log"),
    ],
    format=f"%(asctime)s - rank{rank:03d} - %(levelname)s - %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workdirs",
        nargs="*",
        type=Path,
        help="Specific per-source workdirs. Defaults to config.DATADIR/*_Z.",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=[label for label, *_ in config.BANDS],
        help="Band labels to merge. Defaults to all labels in config.BANDS.",
    )
    parser.add_argument(
        "--adjdir",
        type=Path,
        default=Path(config.ADJDIR),
        help="Destination directory for final copied folders. Defaults to config.ADJDIR.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove an existing final adjoint folder before writing.",
    )
    parser.add_argument(
        "--no-zero-horizontal",
        action="store_true",
        help="Only write merged Z adjoints; do not create zero N/E files.",
    )
    parser.add_argument(
        "--strict-stations",
        action="store_true",
        help="Fail a workdir if any adjoint station is missing from STATIONS.",
    )
    parser.add_argument(
        "--no-copy-adjdir",
        action="store_true",
        help="Only write the folder inside each workdir; do not mirror it to ADJDIR.",
    )
    parser.add_argument(
        "--min-amp",
        type=float,
        default=0.0,
        help="Drop a receiver whose merged adjoint peak |amplitude| is <= this "
             "value: no SEM file is written and it is left out of "
             "STATIONS_ADJOINT. Default 0.0 removes only identically-null "
             "sources, which cc_traveltime produces whenever every window of a "
             "pair measured a tshift of exactly 0 (the CC lag is quantised to "
             "the sample interval, so ~1%% of pairs land there). Such receivers "
             "cost a SPECFEM adjoint receiver but contribute nothing to the "
             "gradient. Raise it to also cut near-null sources, but note that "
             "legitimate peak amplitudes span ~3 orders of magnitude, so this "
             "is an absolute threshold and should be set from the observed "
             "amplitude distribution, not guessed.",
    )
    return parser.parse_args()


def clean_dir(path: Path, parent: Path) -> None:
    """Remove and recreate a child directory, guarding against bad paths."""
    path = path.resolve()
    parent = parent.resolve()
    if path.parent != parent:
        raise ValueError(f"Refusing to clean non-child path: {path}")
    if path.is_symlink():
        path.unlink()
    elif path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def mirror_to_adjdir(src: Path, adjdir: Path, *, clean: bool) -> Path:
    """Copy a prepared workdir folder to ADJDIR/<workdir.name>."""
    adjdir = adjdir.resolve()
    adjdir.mkdir(parents=True, exist_ok=True)
    dest = adjdir / src.name
    resolved_dest = dest.resolve()
    if resolved_dest.parent != adjdir:
        raise ValueError(f"Refusing to copy outside ADJDIR: {dest}")

    if clean:
        if dest.is_symlink():
            dest.unlink()
        elif dest.is_file():
            dest.unlink()
        elif dest.exists():
            shutil.rmtree(dest)
    elif dest.exists():
        raise FileExistsError(f"{dest} already exists; rerun without --no-clean to replace it")

    shutil.copytree(src, dest)
    return dest


def load_adj_file(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"{path} must have at least two columns")
    return data[:, :2].astype(float, copy=False)


def merge_adj_files(paths: list[Path]) -> np.ndarray:
    """Sum adjoint amplitudes, interpolating to the first file's time axis."""
    ref = load_adj_file(paths[0])
    merged = ref.copy()
    merged[:, 1] = 0.0

    for path in paths:
        data = load_adj_file(path)
        if data.shape[0] == ref.shape[0] and np.allclose(data[:, 0], ref[:, 0]):
            amp = data[:, 1]
        else:
            logging.warning("Interpolating %s onto %s time grid", path, paths[0])
            amp = np.interp(ref[:, 0], data[:, 0], data[:, 1], left=0.0, right=0.0)
        merged[:, 1] += amp
    return merged


def component_filename(name: str, component: str) -> str:
    parts = name.split(".")
    if len(parts) < 4:
        raise ValueError(f"Unexpected adjoint filename: {name}")
    parts[2] = parts[2][:-1] + component
    return ".".join(parts)


def is_z_adjoint(path: Path) -> bool:
    parts = path.name.split(".")
    return len(parts) >= 4 and parts[2].endswith("Z") and parts[-1] == "adj"


def receiver_from_adj_name(name: str) -> tuple[str, str]:
    """Return (network, station) from {net}.{sta}.<BAND>Z.adj (band code follows the forward)."""
    parts = name.split(".")
    if len(parts) < 4:
        raise ValueError(f"Unexpected adjoint filename: {name}")
    return parts[0], parts[1]


def station_file_for(workdir: Path) -> Path:
    local_station_file = workdir / "STATIONS"
    if local_station_file.exists():
        return local_station_file
    return Path(config.STATIONFILE)


def load_station_lines(station_file: Path) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    by_pair: dict[tuple[str, str], str] = {}
    by_station: dict[str, str] = {}
    for line in station_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        sta, net = parts[0], parts[1]
        text = line.rstrip() + "\n"
        by_pair[(net, sta)] = text
        by_station.setdefault(sta, text)
    return by_pair, by_station


def collect_adj_groups(workdir: Path, bands: list[str]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for label in bands:
        optdir = workdir / f"OPT_{label}"
        if not optdir.exists():
            logging.info("Missing %s", optdir)
            continue
        for adj in sorted(optdir.glob("*.adj")):
            if is_z_adjoint(adj):
                groups[adj.name].append(adj)
    return dict(groups)


def copy_force_solution(workdir: Path, outdir: Path) -> None:
    for name in ("FORCESOLUTION", "FORCESOLUTION_Z"):
        src = workdir / name
        if src.exists():
            shutil.copy2(src, outdir / "FORCESOLUTION")
            return
    logging.warning("No FORCESOLUTION found in %s", workdir)


def write_station_adjoint(
    outdir: Path,
    adj_names: list[str],
    station_file: Path,
    *,
    strict: bool,
) -> int:
    by_pair, by_station = load_station_lines(station_file)
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    missing = 0

    for name in adj_names:
        net, sta = receiver_from_adj_name(name)
        key = (net, sta)
        if key in seen:
            continue
        seen.add(key)

        line = by_pair.get(key) or by_station.get(sta)
        if line is None:
            missing += 1
            msg = f"No station line for {net}.{sta} in {station_file}"
            if strict:
                raise KeyError(msg)
            logging.warning(msg)
            continue
        lines.append(line)

    (outdir / "STATIONS_ADJOINT").write_text("".join(lines))
    return missing


def process_workdir(
    workdir: Path,
    bands: list[str],
    *,
    adjdir: Path,
    clean: bool,
    copy_adjdir: bool,
    zero_horizontal: bool,
    strict_stations: bool,
    min_amp: float = 0.0,
) -> tuple[int, int, int]:
    groups = collect_adj_groups(workdir, bands)
    if not groups:
        logging.warning("No Z adjoint files found in %s for bands %s", workdir, bands)
        return 0, 0, 0

    # Merge before touching the output folder so a workdir whose sources all
    # turn out to be null does not lose its previous contents to clean_dir().
    merged_by_name: dict[str, np.ndarray] = {}
    dropped_null: list[str] = []
    n_band_inputs = 0
    for name, paths in sorted(groups.items()):
        merged = merge_adj_files(paths)
        if np.max(np.abs(merged[:, 1])) <= min_amp:
            dropped_null.append(name)
            continue
        merged_by_name[name] = merged
        n_band_inputs += len(paths)

    if dropped_null:
        logging.info(
            "Dropped %d/%d null adjoint source(s) (peak |amp| <= %g) in %s: %s",
            len(dropped_null), len(groups), min_amp, workdir.name,
            ", ".join(dropped_null),
        )
    if not merged_by_name:
        logging.warning(
            "All %d adjoint source(s) in %s are null (peak |amp| <= %g); "
            "no adjoint folder written",
            len(groups), workdir, min_amp,
        )
        return 0, 0, len(dropped_null)

    outdir = workdir / workdir.name
    semdir = outdir / "SEM"
    if clean:
        clean_dir(outdir, workdir)
    else:
        outdir.mkdir(parents=True, exist_ok=True)
    semdir.mkdir(parents=True, exist_ok=True)

    for name, merged in merged_by_name.items():
        np.savetxt(semdir / name, merged, fmt="%.10f %.10e")

        if zero_horizontal:
            zeros = merged.copy()
            zeros[:, 1] = 0.0
            for component in ("N", "E"):
                np.savetxt(
                    semdir / component_filename(name, component),
                    zeros,
                    fmt="%.10f %.10e",
                )

    station_file = station_file_for(workdir)
    missing = write_station_adjoint(
        outdir, sorted(merged_by_name), station_file, strict=strict_stations,
    )
    copy_force_solution(workdir, outdir)
    (outdir / "ADJOINT_BANDS.txt").write_text(
        "bands: " + " ".join(bands) + "\n"
        f"merged_z_files: {len(merged_by_name)}\n"
        f"band_input_files: {n_band_inputs}\n"
        f"dropped_null_sources: {len(dropped_null)}\n"
        f"min_amp: {min_amp}\n"
        f"zero_horizontal: {zero_horizontal}\n"
        f"station_file: {station_file}\n"
        f"missing_station_lines: {missing}\n"
    )

    copied_to = None
    if copy_adjdir:
        copied_to = mirror_to_adjdir(outdir, adjdir, clean=clean)
        with (outdir / "ADJOINT_BANDS.txt").open("a") as f:
            f.write(f"copied_to: {copied_to}\n")
        shutil.copy2(outdir / "ADJOINT_BANDS.txt", copied_to / "ADJOINT_BANDS.txt")

    logging.info(
        "Prepared %s%s: %d merged Z files from %d band files%s",
        outdir,
        f" -> {copied_to}" if copied_to is not None else "",
        len(merged_by_name),
        n_band_inputs,
        f" ({len(dropped_null)} null dropped)" if dropped_null else "",
    )
    return len(merged_by_name), n_band_inputs, len(dropped_null)


def main() -> None:
    args = parse_args()
    valid_bands = {label for label, *_ in config.BANDS}
    unknown = [label for label in args.bands if label not in valid_bands]
    if unknown:
        raise SystemExit(f"Unknown band label(s): {unknown}; valid labels: {sorted(valid_bands)}")

    if rank == 0:
        workdirs = [wd.resolve() for wd in args.workdirs]
        if not workdirs:
            workdirs = sorted(Path(config.DATADIR).glob("*_Z"))
        logging.info(
            "Found %d workdirs; distributing across %d ranks. bands=%s adjdir=%s",
            len(workdirs), size, args.bands, args.adjdir,
        )
    else:
        workdirs = None
    workdirs = comm.bcast(workdirs, root=0)
    comm.Barrier()

    my_workdirs = workdirs[rank::size]
    # workdirs, merged Z, band inputs, dropped null sources
    totals = np.array([0, 0, 0, 0], dtype=np.int64)

    with mpi_tqdm(comm, total=len(workdirs), unit="src", desc="prep_adj") as tick:
        for workdir in my_workdirs:
            try:
                n_merged, n_inputs, n_null = process_workdir(
                    workdir,
                    args.bands,
                    adjdir=args.adjdir,
                    clean=not args.no_clean,
                    copy_adjdir=not args.no_copy_adjdir,
                    zero_horizontal=not args.no_zero_horizontal,
                    strict_stations=args.strict_stations,
                    min_amp=args.min_amp,
                )
                totals[3] += n_null
                if n_merged:
                    totals += np.array([1, n_merged, n_inputs, 0], dtype=np.int64)
            except Exception:
                logging.exception("Failed preparing %s", workdir)
            finally:
                tick()

    grand = np.zeros_like(totals)
    comm.Reduce(totals, grand, op=MPI.SUM, root=0)

    for h in logging.getLogger().handlers:
        h.flush()
    comm.Barrier()

    if rank == 0:
        logging.info(
            "Prepared %d workdirs, %d merged Z files, %d band input files, "
            "%d null sources dropped (min_amp=%g).",
            grand[0], grand[1], grand[2], grand[3], args.min_amp,
        )
        rank_logs = sorted(LOG_DIR.glob("4_prep_adj.rank*.log"))
        lines: list[str] = []
        for p in rank_logs:
            lines.extend(p.read_text().splitlines(keepends=True))
        lines.sort()
        (LOG_DIR / "4_prep_adj.log").write_text("".join(lines))


if __name__ == "__main__":
    main()
