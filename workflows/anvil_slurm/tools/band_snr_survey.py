#!/usr/bin/env python3
"""band_snr_survey.py -- does the noise data carry usable 1-1.5 s energy? (the stage gate)

Runs cal_adjoint stage 0's exact per-pair test (make_base_trace -> filter_band
-> compute_time_snr, SNR_CUTOFF) on a random sample of the raw CC files in
obs_data/final_sac_new for the candidate short-period bands and the existing
bands as controls, and reports per band: the distance-gate pass rate, the SNR
pass rate, and the median SNR. The decision it informs: which of BAND_TABLE's
1_3 / 1.5_4 to enable (config.sh CALADJ_BANDS) and whether their group-velocity
gates (vg, vsmin, vsmax) are sane.

Usage (from anvil_cpu/, uses the parent tree's venv and raw data by default):
    ../cal_adjoint/.venv/bin/python tools/band_snr_survey.py --n 2000
    ... --bands 1_3,1.5_4,2_6 --sac ../obs_data/final_sac_new --workers 16 --csv out.csv
"""
import argparse, os, random, sys
from multiprocessing import Pool
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "cal_adjoint"))
os.environ.setdefault("CALADJ_BANDS", "1_3,1.5_4,2_6,3_10,6_18")
import numpy as np
import config                                                    # noqa: E402
from utility import compute_time_snr                             # noqa: E402

_BANDS = None


def _init(bands):
    global _BANDS
    _BANDS = bands


def _one(path):
    import obspy as obp
    from importlib import import_module
    st0 = import_module("0_newprocess")
    try:
        base = st0.make_base_trace(Path(path), config.T1_B, config.T2_A, config.SAMPLING_RATE)
    except Exception as e:  # unreadable file
        return None
    dist = float(base.stats.sac.dist)
    out = {"dist": dist}
    for label, minf, maxf, vg, vsmin, vsmax in _BANDS:
        gate = dist >= vg * config.MINPERIODNUM / maxf
        tr = st0.filter_band(base, minf, maxf, config.SAMPLING_RATE, config.T1_B, config.T2_A)
        sp, se = compute_time_snr(tr.data, 1.0 / tr.stats.sampling_rate, dist, config.T1_B, vsmin, vsmax, eps=config.EPS)
        out[label] = (gate, sp, se)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sac", default=os.path.join(ROOT, "..", "obs_data", "final_sac_new"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bands", default="1_3,1.5_4,2_6,3_10")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    bands = [config.BAND_TABLE[b] for b in a.bands.split(",")]
    files = sorted(Path(a.sac).glob("*/*.sac"))
    if not files:
        sys.exit(f"no SAC files under {a.sac}")
    random.seed(a.seed); sample = random.sample(files, min(a.n, len(files)))
    print(f"{len(files)} raw CC files; sampling {len(sample)}; bands {[b[0] for b in bands]}; SNR_CUTOFF={config.SNR_CUTOFF}")
    with Pool(a.workers, initializer=_init, initargs=(bands,)) as pool:
        res = [r for r in pool.imap_unordered(_one, [str(f) for f in sample], chunksize=8) if r]
    print(f"read {len(res)} pairs; distance median {np.median([r['dist'] for r in res]):.1f} km\n")
    print(f"{'band':>6} {'gate%':>6} {'snr>=cut%':>10} {'both%':>6} {'med_peak':>8} {'med_energy':>10} {'N_both':>7}")
    rows = []
    for label, *_ in bands:
        g = np.array([r[label][0] for r in res]); sp = np.array([r[label][1] for r in res]); se = np.array([r[label][2] for r in res])
        ok = (sp >= config.SNR_CUTOFF) | (se >= config.SNR_CUTOFF)
        both = g & ok
        print(f"{label:>6} {100*g.mean():6.1f} {100*ok.mean():10.1f} {100*both.mean():6.1f} {np.median(sp[g]) if g.any() else 0:8.2f} "
              f"{np.median(se[g]) if g.any() else 0:10.2f} {both.sum():7d}")
        rows.append((label, g.mean(), ok.mean(), both.mean(), both.sum()))
    # distance-binned survival for the short bands: where does 1-3 s live?
    print("\nSNR pass rate (gate-passing pairs) by distance bin:")
    edges = [0, 10, 20, 40, 80, 160, 400]
    d = np.array([r["dist"] for r in res])
    print(f"{'band':>6} " + " ".join(f"{e0:>3}-{e1:<3}km" for e0, e1 in zip(edges[:-1], edges[1:])))
    for label, *_ in bands:
        line = f"{label:>6} "
        for e0, e1 in zip(edges[:-1], edges[1:]):
            m = (d >= e0) & (d < e1) & np.array([r[label][0] for r in res])
            if m.sum() == 0:
                line += f"{'-':>9} "; continue
            ok = np.array([(r[label][1] >= config.SNR_CUTOFF) or (r[label][2] >= config.SNR_CUTOFF) for r in res])[m]
            line += f"{100*ok.mean():5.0f}%/{m.sum():<3d} "
        print(line)
    if a.csv:
        with open(a.csv, "w") as fh:
            fh.write("pair,dist," + ",".join(f"{b[0]}_gate,{b[0]}_peak,{b[0]}_energy" for b in bands) + "\n")
            for r in res:
                fh.write(",".join([str(r.get("pair", "")), f"{r['dist']:.2f}"] + [f"{int(r[b[0]][0])},{r[b[0]][1]:.3f},{r[b[0]][2]:.3f}" for b in bands]) + "\n")
        print("wrote", a.csv)
    print("\nReading: the 2_6 row is the control (it carried the stage-2 inversion). A short band whose 'both%' is a\n"
          "sizeable fraction of 2_6's and whose pass rate holds at 10-40 km is usable; if it only passes at <10 km\n"
          "the band is dominated by near-field pairs and the mesh stage would inverting for very little.")


if __name__ == "__main__":
    main()
