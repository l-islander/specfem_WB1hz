"""Measure pyflex windows for obs/syn pairs and plot them for inspection.

Mirrors the pyflex configuration from 3_calculateadj.py, but instead of
computing adjoint sources it writes a .win file and a .png plot per pair.

Usage (MPI; run with mpiexec / mpirun):
    mpiexec -n <N> python 2_measure_windows.py [--plot] [--no-copy] [--use-whitened] [--reuse]

Single-process is fine too: `python 2_measure_windows.py` (mpi4py treats it as N=1).

Options:
    --plot           Produce per-pair PNGs (off by default — faster bulk runs).
    --no-copy        Do not mirror .win files into config.WINDIR/<label>/<workdir>/.
                     Incompatible with --reuse.
    --use-whitened   Measure on the whitened traces (DATAP_<label>/*.sac.nor and
                     SYNP_<label>/*.HXZ.sac.nor) instead of the raw bandpassed
                     traces. Default is raw, because per-pair spectral whitening
                     destroys the amplitude envelope pyflex uses and conflates
                     source and receiver content. Outputs go to
                     WINNOR_<label>/WINNORPLOT_<label> when this is set, else
                     WIN_<label>/WINPLOT_<label>.
    --reuse          Skip pyflex; stage .win files from the config.WINDIR cache
                     into the workdir-local WINNOR_<band>/ (or WIN_<band>/) dirs.
                     The cache itself is never modified. Use this for later
                     SPECFEM iterations when you want to freeze the windows
                     picked in iteration 1.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import signal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pyflex
from mpi4py import MPI

import config
from mpi_progress import mpi_tqdm
from utility import detect_channel, parse_receiver

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
        logging.FileHandler(LOG_DIR / f"2_measure_windows.rank{rank:03d}.log"),
        #logging.StreamHandler(),
    ],
    format=f"%(asctime)s - rank{rank:03d} - %(levelname)s - %(message)s",
)


def _mode_names(whitened: bool, channel: str = "HX") -> tuple[str, str, str, str]:
    """Return (obs_glob, syn_suffix, win_prefix, plot_prefix) for the mode.

    `*.sac` does not match `*.sac.nor`, so the pre-whitening glob cleanly
    excludes whitened files sitting in the same DATAP_<label> directory.
    Whitened mode pairs whitened OBS with whitened SYN; pre-whitening mode
    pairs the raw OBS with the raw SYN.
    """
    # `channel` (band+instrument code, e.g. "BX") makes the SYN suffix follow the
    # forward run. win_prefix/plot_prefix are channel-independent, so callers that
    # only need those may leave channel at its default.
    if whitened:
        return "*.sac.nor", f".{channel}Z.sac.nor", "WINNOR", "WINNORPLOT"
    return "*.sac", f".{channel}Z.sac", "WIN", "WINPLOT"


def make_pyflex_config(minp: float, maxp: float) -> pyflex.Config:
    return pyflex.Config(
        min_period=minp, max_period=maxp,
        # stalta_waterlevel=0.09,                   # old
        stalta_waterlevel=0.10,
        # tshift_acceptance_level=0.3 * maxp,       # old
        tshift_acceptance_level=0.15 * maxp,
        # dlna_acceptance_level=1.0,                # old
        dlna_acceptance_level=3.0,
        # cc_acceptance_level=0.7,                  # old
        cc_acceptance_level=0.70,
        # s2n_limit=3,                              # old
        s2n_limit=3.0,
        # c_0=0.7,                                  # old
        c_0=1.0,
        c_1=2.0, c_2=0.0,
        c_3a=4.0, c_3b=2.0, c_4a=3.0, c_4b=10.0,
        check_global_data_quality=False,
        # snr_integrate_base=3, snr_max_base=3,     # old
        snr_integrate_base=3.0, snr_max_base=3.0,
        window_signal_to_noise_type="energy",
    )


# Post-filter: drop any pyflex window whose peak |obs| amplitude is below
# this fraction of the full trace's peak |obs|. Kills weak precursor / coda
# windows that match each other in shape but carry no real signal.
MIN_WINDOW_AMP_FRAC = 0.20

# Collapse contiguous windows into one per arrival? Disabled: a widened window
# has to be re-measured rather than inherit its members' criteria (see
# merge_contiguous_windows), and re-measuring rejects ~3 of every 4 merges for
# failing pyflex's own acceptance bar. What survived was not worth the exposure:
# while the merge went unvalidated it produced cycle-skipped shifts at ~100x the
# rate of unmerged pairs and, since cc_traveltime's misfit goes as tshift^2,
# those dominated the gradient -- plus ~20% of merged windows carried a
# cross-correlation below cc_acceptance_level. Splitting one surface-wave train
# across several windows is only cosmetic: pyadjoint sums their contributions.
# The implementation is correct and validated; flip this to re-enable it.
MERGE_CONTIGUOUS_WINDOWS = False


def filter_low_amplitude_windows(windows, obs_data, synth_data) -> list:
    """Keep only windows where BOTH obs and syn carry real signal — i.e. the
    in-window peak of each is at least MIN_WINDOW_AMP_FRAC of that trace's
    overall peak. This kills precursor / coda windows that match in shape but
    sit in a quiet region of either trace.
    """
    if not windows:
        return list(windows)
    obs_peak = float(np.max(np.abs(obs_data.data))) if obs_data.data.size else 0.0
    syn_peak = float(np.max(np.abs(synth_data.data))) if synth_data.data.size else 0.0
    if obs_peak <= 0 or syn_peak <= 0:
        return list(windows)
    obs_thr = MIN_WINDOW_AMP_FRAC * obs_peak
    syn_thr = MIN_WINDOW_AMP_FRAC * syn_peak
    kept = []
    for w in windows:
        obs_seg = obs_data.data[w.left:w.right]
        syn_seg = synth_data.data[w.left:w.right]
        if obs_seg.size == 0 or syn_seg.size == 0:
            continue
        if (float(np.max(np.abs(obs_seg))) >= obs_thr
                and float(np.max(np.abs(syn_seg))) >= syn_thr):
            kept.append(w)
    return kept


def apply_velocity_corridor(windows, *, dist: float, delta: float, t1_b: float,
                            vsmax: float, min_len_samples: float):
    """Reject / trim windows that start before the earliest possible arrival.

    An ambient-noise EGF contains essentially one Rayleigh arrival, which cannot
    precede ``dist / vsmax``. pyflex has no way to know this: it picks windows
    from the *synthetic* STA/LTA, and the synthetic (a SPECFEM point-force run)
    does contain body waves the EGF lacks. Its own traveltime guard is anchored
    on taup's first body arrival, is disabled in this fork, and could not
    express a surface-wave velocity ceiling anyway.

    Each window is anchored on an STA/LTA peak (``w.center``):
      * ``center`` before the cutoff -- the window exists *because of* the
        body-wave energy, so trimming would leave a window that no longer
        contains its own peak and whose cc/dlnA were measured over the deleted
        part. Drop it.
      * ``center`` valid but ``left`` overhanging -- a real surface-wave pick
        with a sloppy left bound. Trim ``left`` to the cutoff, then re-apply the
        minimum-length test the trim may have broken.

    Returns ``(kept, n_trimmed, n_dropped)``.
    """
    cutoff = (dist / vsmax - t1_b) / delta  # earliest legal sample index
    kept, n_trimmed, n_dropped = [], 0, 0
    for w in windows:
        if w.right <= cutoff or getattr(w, "center", w.left) < cutoff:
            n_dropped += 1
            continue
        if w.left < cutoff:
            w.left = int(np.ceil(cutoff))
            if (w.right - w.left) < min_len_samples:
                n_dropped += 1
                continue
            n_trimmed += 1
        kept.append(w)
    return kept, n_trimmed, n_dropped


def merge_contiguous_windows(windows, obs, syn, pyflex_cfg, *,
                             max_gap_samples: int = 1):
    """Merge windows that abut into one, so a pair yields one window per arrival.

    FLEXWIN generates one window per STA/LTA peak, bounded by the troughs on
    either side. A dispersed Rayleigh train has several envelope peaks, so a
    single arrival gets split into windows that share trough boundaries -- e.g.
    1173-1335, 1335-1499, 1499-1684, all with the same cc_shift. For distinct
    body-wave phases that separation is meaningful; for one surface-wave train
    it is an artifact.

    A merged window must be re-measured, not merely re-bounded. pyflex accepted
    each narrow member on its own bounds, but that verdict does not carry over to
    their union: pyadjoint re-measures across whatever bounds it is handed, and
    correlating the whole widened train can lock onto a neighbouring cycle.
    Inheriting the highest-cc member's criteria hid that -- the .win row looked
    self-consistent while describing measurements nobody had made on those
    bounds. Left unchecked this produced cycle-skipped measurements at ~100x the
    rate of unmerged pairs, and because cc_traveltime's misfit goes as tshift^2
    they dominated the gradient.

    So the merged window's criteria are recomputed here and held to *every*
    pyflex acceptance test, not just the traveltime one: widening the bounds
    admits signal the members never covered, which degrades ``max_cc_value`` and
    ``dlnA`` as readily as it moves ``cc_shift``. Checking only the shift let
    ~20%% of merged windows through with a cross-correlation below
    ``cc_acceptance_level`` -- windows pyflex would have rejected outright had it
    been shown them.

    A merge that fails any test restores its members untouched and returns them
    unmerged: each was valid alone, so the cost is the cosmetic
    one-window-per-arrival property, not a lost measurement.

    ``obs``/``syn`` must be the arrays pyflex itself measured against
    (``WindowSelector.observed.data`` / ``.synthetic.data``), not the raw traces,
    so a recomputed shift is comparable with the one pyflex recorded.

    Returns ``(kept, n_merges, n_reverted)``.
    """
    if len(windows) < 2:
        return list(windows), 0, 0

    def snapshot(w):
        return (w.left, w.right, w.max_cc_value, w.cc_shift, w.dlnA)

    def restore(w, snap):
        w.left, w.right, w.max_cc_value, w.cc_shift, w.dlnA = snap

    def accepted(w) -> bool:
        """Hold a re-measured window to every pyflex criterion it can fail."""
        shift = w.cc_shift_in_seconds
        if shift is not None and abs(shift) > pyflex_cfg.tshift_acceptance_level:
            return False
        if (w.max_cc_value is not None
                and w.max_cc_value < pyflex_cfg.cc_acceptance_level):
            return False
        if w.dlnA is not None and abs(w.dlnA) > pyflex_cfg.dlna_acceptance_level:
            return False
        return True

    # Group contiguous runs before mutating anything: merging in place while
    # scanning would overwrite the bounds a failed merge has to fall back to.
    ordered = sorted(windows, key=lambda w: w.left)
    runs = [[ordered[0]]]
    for w in ordered[1:]:
        if w.left - runs[-1][-1].right <= max_gap_samples:
            runs[-1].append(w)
        else:
            runs.append([w])

    kept, n_merges, n_reverted = [], 0, 0
    for run in runs:
        if len(run) == 1:
            kept.append(run[0])
            continue

        saved = [snapshot(w) for w in run]
        best = max(run, key=lambda x: getattr(x, "max_cc_value", 0.0) or 0.0)
        new_left = run[0].left
        new_right = max(w.right for w in run)
        best.left, best.right = new_left, new_right
        best._calc_criteria(obs, syn)

        if accepted(best):
            kept.append(best)
            n_merges += len(run) - 1
        else:
            for w, snap in zip(run, saved):
                restore(w, snap)
            kept.extend(run)
            n_reverted += 1

    kept.sort(key=lambda w: w.left)
    return kept, n_merges, n_reverted


_PYFLEX_TIMEOUT = 60


def select_windows_safe(obs_data, synth_data, pyflex_cfg, label: str):
    """Run pyflex via WindowSelector so we can re-use it for plotting."""
    def _handle_alarm(*_):
        raise TimeoutError(f"pyflex timed out after {_PYFLEX_TIMEOUT}s")

    has_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _handle_alarm)
        signal.alarm(_PYFLEX_TIMEOUT)
    try:
        ws = pyflex.WindowSelector(obs_data, synth_data, pyflex_cfg)
        windows = ws.select_windows()
    except TimeoutError as e:
        logging.warning("pyflex timeout [%s]: %s", label, e)
        return None, []
    except Exception as e:
        logging.warning("pyflex failed [%s]: %s", label, e)
        return None, []
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return ws, windows


def write_winfile(
    windows, obs_data, obsfile: Path, synfile: Path, outfile: Path, delta: float,
) -> None:
    min_left = int(round(1 / delta))
    with open(outfile, "w") as f:
        f.write(f"dist {obs_data.stats.sac.dist:.1f} win_num {len(windows)}\n")
        for i, w in enumerate(windows):
            if w.left * delta < 1:
                w.left = min_left
            # NB: synfile must stay last -- plotwindowandadj.py parses this with
            # split(maxsplit=7) so the final field absorbs the rest of the line
            # (paths may contain spaces). Appending a column here silently
            # corrupts synfile; add new fields before obsfile and bump maxsplit.
            f.write(
                f"{i} {w.left} {w.right} {delta} {w.cc_shift_in_seconds} "
                f"{w.max_cc_value} {obsfile} {synfile}\n"
            )


def plot_windows(ws: pyflex.WindowSelector, outfile: Path) -> None:
    """Use pyflex's built-in WindowSelector.plot() to render the diagnostic
    figure (obs/syn traces, STA/LTA, candidate windows + final selected windows).
    """
    ws.plot(filename=str(outfile))
    plt.close("all")


def plot_obs_syn(
    obs_data, synth_data, outfile: Path, *,
    title: str, windows=None,
) -> None:
    """Plot obs (black) vs syn (red), with pyflex windows shaded if provided.
    Used both as a fallback (no windows -> tagged NO WINDOWS) and as a
    companion to the pyflex diagnostic figure for pairs that have windows.
    """
    delta = obs_data.stats.delta
    npts = obs_data.stats.npts
    b = float(getattr(obs_data.stats.sac, "b", 0.0))
    t = np.arange(npts) * delta + b

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, obs_data.data, "k-", lw=0.8, label="obs")
    ax.plot(t, synth_data.data, "r-", lw=0.8, label="syn")

    if windows:
        _, ymax = ax.get_ylim()
        for i, w in enumerate(windows):
            wl = t[0] + w.left * delta
            wr = t[0] + w.right * delta
            ax.axvspan(wl, wr, color="tab:blue", alpha=0.18,
                       label="pyflex window" if i == 0 else None)
            ax.text(
                0.5 * (wl + wr), ymax * 0.92,
                f"cc={w.max_cc_value:.2f}\ndt={w.cc_shift_in_seconds:.2f}s",
                ha="center", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
            )

    dist = getattr(obs_data.stats.sac, "dist", float("nan"))
    suffix = f"nwin={len(windows)}" if windows else "NO WINDOWS"
    ax.set_title(f"{title}  dist={dist:.1f} km  {suffix}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    fig.savefig(str(outfile), dpi=120)
    plt.close(fig)


def process_pair(
    obsfile: Path, syndir: Path, windir: Path, plotdir: Path,
    pyflex_cfg: pyflex.Config, *, label: str, source: str, maxp: float,
    vsmax: float,
    channel: str = "HX",
    syn_suffix: str = ".HXZ.sac",
    sourcewindir: Path | None = None, make_plot: bool = True,
) -> None:
    # The workdir source can sit at either end of the cc-pair filename after
    # sac_reorder; pick the receiver as the side that isn't the source.
    net_sta, sta = parse_receiver(obsfile.name, source)

    synfile = syndir / f"{net_sta}.{sta}{syn_suffix}"
    if not synfile.exists():
        logging.warning("Synth not found [%s] %s.%s: %s", label, net_sta, sta, synfile)
        return

    # `channel` (forward band+instrument code) so the win/plot names follow the run.
    winfile = windir / f"{sta}.{net_sta}..{channel}Z.win"
    # Skip pairs that already have a win + (plot if requested) from a previous
    # (interrupted) run. The plot filename embeds the distance prefix, which we
    # don't know until we read the obs header, so we glob for it.
    plot_ok = (not make_plot) or bool(list(plotdir.glob(f"*_*_{net_sta}.{sta}.{channel}Z.png")))
    if plot_ok and winfile.exists():
        return

    obs_data = obspy.read(str(obsfile), format="SAC")[0]
    synth_data = obspy.read(str(synfile), format="SAC")[0]
    delta = obs_data.stats.sac.delta

    # Embed inter-station distance + source-to-receiver azimuth in the plot
    # filename so default sort order matches distance (then azimuth as a
    # tiebreaker). Both are zero-padded so lexicographic sort is numeric.
    dist = float(getattr(obs_data.stats.sac, "dist", 0.0) or 0.0)
    azi = float(getattr(obs_data.stats.sac, "az", 0.0) or 0.0)
    plotfile = plotdir / f"{dist:07.2f}_{azi:05.1f}_{net_sta}.{sta}.{channel}Z.png"

    tag = f"{label} {net_sta}.{sta}"
    ws, windows = select_windows_safe(obs_data, synth_data, pyflex_cfg, tag)

    if windows:
        n_before = len(windows)
        windows = filter_low_amplitude_windows(windows, obs_data, synth_data)
        if len(windows) < n_before:
            logging.info(
                "Amplitude filter dropped %d/%d windows [%s] %s.%s (thr=%.2f * peak)",
                n_before - len(windows), n_before, label, net_sta, sta,
                MIN_WINDOW_AMP_FRAC,
            )

    if windows and dist > 0:
        # pyflex's own traveltime guard is disabled in this fork (and is anchored
        # on body-wave arrivals regardless), so the surface-wave velocity ceiling
        # has to be enforced here. Reuse pyflex's own minimum-length definition
        # so a trimmed window is held to the same bar as an original one.
        min_len_samples = pyflex_cfg.c_1 * pyflex_cfg.min_period / delta
        windows, n_trim, n_drop = apply_velocity_corridor(
            windows, dist=dist, delta=delta, t1_b=config.T1_B,
            vsmax=vsmax, min_len_samples=min_len_samples,
        )
        if n_trim or n_drop:
            logging.info(
                "Corridor [%s] %s.%s (dist=%.1f km, earliest=%.1f s): "
                "trimmed %d, dropped %d, kept %d",
                label, net_sta, sta, dist, dist / vsmax, n_trim, n_drop, len(windows),
            )

    if windows and MERGE_CONTIGUOUS_WINDOWS:
        # ws is only None when pyflex returned no windows, so it is safe here.
        windows, n_merged, n_reverted = merge_contiguous_windows(
            windows, ws.observed.data, ws.synthetic.data, pyflex_cfg,
        )
        if n_merged:
            logging.info("Merged %d contiguous window(s) -> %d [%s] %s.%s",
                         n_merged, len(windows), label, net_sta, sta)
        if n_reverted:
            logging.info(
                "Reverted %d merge(s) [%s] %s.%s: re-measured window failed "
                "pyflex acceptance (tshift<=%.2fs, cc>=%.2f, |dlnA|<=%.2f); "
                "kept the unmerged windows",
                n_reverted, label, net_sta, sta,
                pyflex_cfg.tshift_acceptance_level,
                pyflex_cfg.cc_acceptance_level,
                pyflex_cfg.dlna_acceptance_level,
            )

    title = f"{net_sta}.{sta}.{channel}Z  band={1/maxp:.3f}-{1/pyflex_cfg.min_period:.3f} Hz"

    if not windows:
        logging.info("No window [%s] %s.%s", label, net_sta, sta)
        if make_plot:
            try:
                plot_obs_syn(obs_data, synth_data, plotfile, title=title)
            except Exception:
                logging.exception("obs/syn plot failed [%s] %s.%s", label, net_sta, sta)
        return

    write_winfile(windows, obs_data, obsfile, synfile, winfile, delta)
    if sourcewindir is not None:
        try:
            sourcewindir.mkdir(parents=True, exist_ok=True)
            shutil.copy(winfile, sourcewindir / winfile.name)
        except Exception:
            logging.exception("copy to sourcewindir failed [%s] %s.%s", label, net_sta, sta)
    if make_plot:
        try:
            plot_obs_syn(obs_data, synth_data, plotfile,
                         title=title, windows=windows)
        except Exception:
            logging.exception("obs/syn plot failed [%s] %s.%s", label, net_sta, sta)
        try:
            pyflex_plotfile = plotfile.with_name(plotfile.stem + ".pyflex.png")
            plot_windows(ws, pyflex_plotfile)
        except Exception:
            logging.exception("pyflex plot failed [%s] %s.%s", label, net_sta, sta)
    logging.info("Wrote %d windows%s [%s] %s.%s",
                 len(windows), " + plots" if make_plot else "",
                 label, net_sta, sta)


def process_band(
    workdir: Path, label: str, minp: float, maxp: float, *,
    vsmax: float,
    windowsdir: Path | None = None, make_plot: bool = True, tick=None,
    whitened: bool = False,
) -> None:
    obsdir = workdir / f"DATAP_{label}"
    syndir = workdir / f"SYNP_{label}"
    if not obsdir.exists() or not syndir.exists():
        logging.info("Skip [%s] %s: missing obs/syn dir", label, workdir.name)
        return

    channel = detect_channel(workdir)
    obs_glob, syn_suffix, win_prefix, plot_prefix = _mode_names(whitened, channel)
    windir = workdir / f"{win_prefix}_{label}"
    plotdir = workdir / f"{plot_prefix}_{label}"
    windir.mkdir(parents=True, exist_ok=True)
    if make_plot:
        plotdir.mkdir(parents=True, exist_ok=True)

    source = workdir.name.split("_")[0]
    sourcewindir = (
        windowsdir / label / workdir.name if windowsdir is not None else None
    )

    pyflex_cfg = make_pyflex_config(minp, maxp)
    for obsfile in sorted(obsdir.glob(obs_glob)):
        try:
            process_pair(obsfile, syndir, windir, plotdir, pyflex_cfg,
                         label=label, source=source,
                         maxp=maxp, vsmax=vsmax,
                         channel=channel, syn_suffix=syn_suffix,
                         sourcewindir=sourcewindir, make_plot=make_plot)
        except Exception:
            logging.exception("Failed [%s] %s", label, obsfile)
        finally:
            if tick is not None:
                tick()


def process_workdir(
    workdir: Path,
    bands: list[tuple[str, float, float]],
    *,
    windowsdir: Path | None = None,
    make_plot: bool = True,
    tick=None,
    whitened: bool = False,
) -> None:
    logging.info("Processing %s", workdir.name)
    for label, minf, maxf, _vg, _vsmin, vsmax in bands:
        minp, maxp = 1.0 / maxf, 1.0 / minf
        try:
            process_band(workdir, label, minp, maxp, vsmax=vsmax,
                         windowsdir=windowsdir, make_plot=make_plot, tick=tick,
                         whitened=whitened)
        except Exception:
            logging.exception("Failed band %s in %s", label, workdir)


def stage_from_cache(
    workdir: Path,
    bands: list[tuple[str, float, float]],
    *,
    windowsdir: Path,
    whitened: bool,
    tick=None,
) -> None:
    """Copy cached .win files from `windowsdir/<label>/<workdir>/` into the
    workdir-local WINNOR_<band>/ (or WIN_<band>/) dir without running pyflex.

    Used by `--reuse` so subsequent SPECFEM iterations can re-use the windows
    picked in iteration 1. The cache itself is never modified here.
    """
    _, _, win_prefix, _ = _mode_names(whitened)
    for label, *_ in bands:
        src = windowsdir / label / workdir.name
        dst = workdir / f"{win_prefix}_{label}"
        dst.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            logging.info("No cached windows for [%s] %s: %s", label, workdir.name, src)
            continue
        for winfile in sorted(src.glob("*.win")):
            try:
                shutil.copy2(winfile, dst / winfile.name)
            except Exception:
                logging.exception("copy from cache failed [%s] %s -> %s",
                                  label, winfile, dst)
            finally:
                if tick is not None:
                    tick()


def clean_output_dirs(
    workdirs: list[Path],
    bands: list[tuple[str, float, float]],
    *,
    windowsdir: Path | None = None,
    whitened: bool = False,
) -> None:
    """Remove the current mode's WIN*_<band> and WIN*PLOT_<band> dirs (and, if
    a global `windowsdir` is given, the matching <windowsdir>/<label>/<workdir>/
    subdirs) so a new measurement run does not mix with stale outputs.

    Only the *current* mode's dirs are removed: a whitened run leaves
    WIN_/WINPLOT_ intact, and a --pre-whitening run leaves WINNOR_/WINNORPLOT_
    intact, so the two stay available for side-by-side comparison.
    """
    _, _, win_prefix, plot_prefix = _mode_names(whitened)
    for wd in workdirs:
        for label, *_ in bands:
            for sub in (wd / f"{win_prefix}_{label}", wd / f"{plot_prefix}_{label}"):
                if sub.exists():
                    shutil.rmtree(sub)
            if windowsdir is not None:
                src_sub = windowsdir / label / wd.name
                if src_sub.exists():
                    shutil.rmtree(src_sub)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot", action="store_true",
        help="Produce per-pair obs/syn and pyflex diagnostic plots. Off by "
             "default; window files (.win) are written either way.",
    )
    parser.add_argument(
        "--no-copy", action="store_true",
        help="Do not mirror each .win file into config.WINDIR/<label>/<workdir>/.",
    )
    parser.add_argument(
        "--use-whitened", action="store_true",
        help="Measure on whitened traces (*.sac.nor) instead of the raw "
             "bandpassed ones. Default is raw — per-pair whitening flattens "
             "the envelope pyflex needs and conflates source/receiver content.",
    )
    parser.add_argument(
        "--reuse", action="store_true",
        help="Skip pyflex; stage windows from the config.WINDIR cache into "
             "<workdir>/WINNOR_<band>/ (or WIN_<band>/ in raw mode). The cache "
             "itself is never touched. Useful for SPECFEM iterations where you "
             "want to freeze the window picks across model updates.",
    )
    args = parser.parse_args()

    make_plot = args.plot
    windowsdir = None if args.no_copy else config.WINDIR
    whitened = args.use_whitened
    reuse = args.reuse

    if reuse and windowsdir is None:
        if rank == 0:
            print("--reuse requires the config.WINDIR cache (cannot combine with --no-copy).")
        return

    syndirlist = config.DATADIR
    bands = config.BANDS
    obs_glob, _syn_suffix, win_prefix, plot_prefix = _mode_names(whitened)

    if rank == 0:
        workdirs = sorted(syndirlist.glob("*_Z"))
        logging.info(
            "Found %d workdirs; distributing across %d ranks. bands=%s mode=%s reuse=%s",
            len(workdirs), size, [b[0] for b in bands],
            "whitened" if whitened else "raw", reuse,
        )
        # In --reuse mode the cache is the source of truth — never wipe it.
        clean_windowsdir = None if reuse else windowsdir
        logging.info(
            "Cleaning %s_/%s_ (and %s) dirs for %d workdirs...",
            win_prefix, plot_prefix,
            clean_windowsdir if clean_windowsdir is not None else "no source windir",
            len(workdirs),
        )
        clean_output_dirs(workdirs, bands,
                          windowsdir=clean_windowsdir, whitened=whitened)
    else:
        workdirs = None
    workdirs = comm.bcast(workdirs, root=0)
    comm.Barrier()  # ensure cleanup finishes before any rank starts writing

    my_workdirs = workdirs[rank::size]

    if reuse:
        local_pairs = sum(
            len(list((windowsdir / label / wd.name).glob("*.win")))
            for wd in my_workdirs
            for label, *_ in bands
            if (windowsdir / label / wd.name).is_dir()
        )
    else:
        local_pairs = sum(
            len(list((wd / f"DATAP_{label}").glob(obs_glob)))
            for wd in my_workdirs
            for label, *_ in bands
            if (wd / f"DATAP_{label}").exists() and (wd / f"SYNP_{label}").exists()
        )
    total_pairs = comm.allreduce(local_pairs, op=MPI.SUM)

    desc = "stage" if reuse else "win"
    with mpi_tqdm(comm, total=total_pairs, unit="pair", desc=desc,
                  mininterval=2.0) as tick:
        for workdir in my_workdirs:
            try:
                if reuse:
                    stage_from_cache(workdir, bands,
                                     windowsdir=windowsdir, whitened=whitened,
                                     tick=tick)
                else:
                    process_workdir(workdir, bands,
                                    windowsdir=windowsdir, make_plot=make_plot,
                                    tick=tick, whitened=whitened)
            except Exception:
                logging.exception("Failed processing workdir %s", workdir)

    for h in logging.getLogger().handlers:
        h.flush()
    comm.Barrier()

    if rank == 0:
        logging.info("All ranks finished. Combining per-rank logs.")
        for h in logging.getLogger().handlers:
            h.flush()

        rank_logs = sorted(LOG_DIR.glob("2_measure_windows.rank*.log"))
        lines: list[str] = []
        for p in rank_logs:
            lines.extend(p.read_text().splitlines(keepends=True))
        lines.sort()
        (LOG_DIR / "2_measure_windows.log").write_text("".join(lines))


if __name__ == "__main__":
    main()
