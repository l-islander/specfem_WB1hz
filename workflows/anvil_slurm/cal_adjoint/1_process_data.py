from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import obspy as obp
from mpi4py import MPI
from obspy import Trace

import config
from mpi_progress import mpi_tqdm
from utility import detect_channel, load_stations, parse_receiver, read_semd

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
        logging.FileHandler(LOG_DIR / f"1_process_data.rank{rank:03d}.log"),
        #logging.StreamHandler(),
    ],
    format=f"%(asctime)s - rank{rank:03d} - %(levelname)s - %(message)s",
)


def process_obs_file(
    obsfile: Path,
    syndir: Path,
    optsyndir: Path,
    station_info: dict[str, tuple[float, float]],
    *,
    source: str,
    channel: str,
    minf: float,
    maxf: float,
    t1_b: float,
    t2_a: float,
) -> None:
    # `source` is "NET.STA" from the workdir name; receiver = the other side
    # of the cc-pair filename. The previous parts[2:4] assumption silently
    # dropped pairs whose source ended up at parts[0:2] after sac_reorder.
    net_sta, sta = parse_receiver(obsfile.name, source)
    receiver = f"{net_sta}.{sta}"

    # `channel` = the forward's band+instrument code (BX, HX, ... from detect_channel),
    # so the whole pipeline follows the forward run. All stage-1..4 products carry it.
    output_path = optsyndir / f"{net_sta}.{sta}.{channel}Z.sac"
    # reprocess even if output exists, since we may have updated processing steps (e.g., bandpass filter parameters)
    #if output_path.exists():
    #    return

    st_obs = obp.read(str(obsfile))[0]

    # Input glob stays band-agnostic ([BH]XZ) so a stray HX/BX mix is still found; the
    # OUTPUT name above uses the detected `channel` so downstream stages stay consistent.
    syn_candidates = sorted(syndir.glob(f"{net_sta}.{sta}.[BH]XZ.semd"))
    if not syn_candidates:
        logging.warning("Synthetic file for %s %s not found, skipping %s", net_sta, sta, obsfile)
        return
    synfile = syn_candidates[0]

    # ascii or binary (SEISMO_FORMAT=bin) -- read_semd detects per file
    syn_times, syn_displacements = read_semd(synfile)

    oriheader = st_obs.stats.copy()
    oriheader.npts = len(syn_displacements)
    oriheader.starttime = st_obs.stats.starttime - t1_b + syn_times[0]
    oriheader.delta = syn_times[1] - syn_times[0] if len(syn_times) > 1 else 0.0

    st_syn = Trace(data=syn_displacements, header=oriheader)
    st_syn.filter("bandpass", freqmin=minf, freqmax=maxf, corners=4, zerophase=True)
    st_syn.resample(st_obs.stats.sampling_rate)
    st_syn.trim(st_obs.stats.starttime, st_obs.stats.endtime, pad=True, fill_value=0)
    st_syn.normalize()

    st_syn.stats.sac.evla = station_info[source][0]
    st_syn.stats.sac.evlo = station_info[source][1]
    st_syn.stats.sac.stla = station_info[receiver][0]
    st_syn.stats.sac.stlo = station_info[receiver][1]
    st_syn.stats.sac.kstnm = sta
    st_syn.stats.sac.knetwk = net_sta
    st_syn.stats.sac.kcmpnm = "MXZ"
    st_syn.stats.sac.evdp = 0.0
    st_syn.stats.sac.b = t1_b
    st_syn.stats.sac.e = t2_a

    st_syn.write(str(output_path), format="SAC")


def process_workdir(
    workdir: Path,
    ccdatadir: Path,
    station_info: dict[str, tuple[float, float]],
    bands: list[tuple[str, float, float]],
    *,
    t1_b: float,
    t2_a: float,
    tick=None,
) -> None:
    source = workdir.name.split("_")[0]
    logging.info("Processing %s", source)

    syndir = workdir / "SYN"
    if not syndir.exists():
        syndir.mkdir(parents=True, exist_ok=True)
        for semd in workdir.glob("*.semd"):
            shutil.move(str(semd), str(syndir))
        logging.info("Moved synthetic files to %s", syndir)

    # Band+instrument code from the forward synthetics (now under SYN/). Every product
    # this workdir writes carries it, so stages 2-4 stay consistent with the forward run.
    channel = detect_channel(workdir)

    for label, minf, maxf, *_ in bands:
        band_datadir = ccdatadir / label / source
        if not band_datadir.exists():
            logging.warning("cc data dir %s does not exist, skipping band %s", band_datadir, label)
            continue

        optsyndir = workdir / f"SYNP_{label}"
        datapdir = workdir / f"DATAP_{label}"
        optsyndir.mkdir(parents=True, exist_ok=True)
        #relink/copy data even if datapdir exists, since we may have updated processing steps (e.g., bandpass filter parameters)
        if datapdir.exists():
            shutil.rmtree(datapdir)
        # Use symlink if possible to save disk space, but fall back to copying if not (e.g., on Windows without admin rights).
        try:
            datapdir.symlink_to(band_datadir, target_is_directory=True)
        except OSError:
            shutil.copytree(band_datadir, datapdir)

        obsfiles = sorted(band_datadir.glob("*.sac"))
        if not obsfiles:
            logging.warning("No cc data in %s, skipping", band_datadir)
            continue

        for obsfile in obsfiles:
            try:
                process_obs_file(
                    obsfile, syndir, optsyndir, station_info,
                    source=source, channel=channel,
                    minf=minf, maxf=maxf, t1_b=t1_b, t2_a=t2_a,
                )
            except Exception:
                logging.exception("Failed processing %s [%s]", obsfile, label)
            finally:
                if tick is not None:
                    tick()


def _remove_dir_or_link(p: Path) -> None:
    """Remove a directory, handling the case where it is a symlink (DATAP_ is
    often a symlink to the cc data dir — shutil.rmtree on a symlink can delete
    the target, which we never want).
    """
    if p.is_symlink():
        p.unlink()
    elif p.exists():
        shutil.rmtree(p)


def clean_output_dirs(workdirs: list[Path]) -> None:
    """Remove ALL existing DATAP_*, SYNP_*, OPT_* dirs (any band label) so a
    new processing run starts clean — including stale dirs from old band
    configs that are no longer in config.BANDS.
    """
    prefixes = ("DATAP_", "SYNP_", "OPT_")
    for wd in workdirs:
        for child in wd.iterdir():
            if child.name.startswith(prefixes):
                _remove_dir_or_link(child)


def main() -> None:
    ccdatadir = config.CCDIR
    syndirlist = config.DATADIR
    stationfile = config.STATIONFILE

    cfg = dict(
        t1_b=config.T1_B,
        t2_a=config.T2_A,
    )

    station_info = load_stations(stationfile)

    if rank == 0:
        workdirs = sorted(syndirlist.glob("*_Z"))
        logging.info("Found %d workdirs; distributing across %d ranks.", len(workdirs), size)
    else:
        workdirs = None
    workdirs = comm.bcast(workdirs, root=0)

    my_workdirs = workdirs[rank::size]

    logging.info("Cleaning DATAP_*/SYNP_*/OPT_* dirs for %d local workdirs...",
                 len(my_workdirs))
    clean_output_dirs(my_workdirs)
    comm.Barrier()  # ensure all ranks finish cleanup before any starts writing
    local_pairs = sum(
        len(list((ccdatadir / label / wd.name.split("_")[0]).glob("*.sac")))
        for wd in my_workdirs
        for label, *_ in config.BANDS
        if (ccdatadir / label / wd.name.split("_")[0]).exists()
    )
    total_pairs = comm.allreduce(local_pairs, op=MPI.SUM)

    with mpi_tqdm(comm, total=total_pairs, unit="file", desc="process_data") as tick:
        for workdir in my_workdirs:
            try:
                process_workdir(workdir, ccdatadir, station_info, config.BANDS,
                                tick=tick, **cfg)
            except Exception:
                logging.exception("Failed processing workdir %s", workdir)

    for h in logging.getLogger().handlers:
        h.flush()
    comm.Barrier()

    if rank == 0:
        logging.info("All ranks finished. Combining per-rank logs.")
        for h in logging.getLogger().handlers:
            h.flush()

        rank_logs = sorted(LOG_DIR.glob("1_process_data.rank*.log"))
        lines: list[str] = []
        for p in rank_logs:
            lines.extend(p.read_text().splitlines(keepends=True))
        lines.sort()
        (LOG_DIR / "1_process_data.log").write_text("".join(lines))


if __name__ == "__main__":
    main()
