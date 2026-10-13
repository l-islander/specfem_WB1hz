#!/usr/bin/env python3
"""validate_forward.py -- compare one event's SPECFEM seismograms against a reference.

Validation of a port: the forward must match the
x86/RTX-5090 reference (work/forward/result_M6/<event>/) within single-precision
tolerance. It reports:

  * max relative amplitude difference  max|test-ref| / max|ref|   (per trace, then
    worst / median / p99 over traces)
  * relative L2 misfit                 ||test-ref|| / ||ref||
  * zero-lag correlation coefficient   and the lag (in samples and seconds) that
    maximises the cross-correlation ("cc/dt of the diff")
  * time-axis identity                 (same NSTEP, same DT, same t0)
  * file-set identity                  (same station list)

Traces may be ascii or binary (config.sh SEISMO_FORMAT); formats can be mixed
between ref and test, e.g. an ascii x86 reference against a binary GH200 run.

Verdict bands (override with --pass/--warn):
  PASS   worst rel-amp <= 1e-4 and min cc >= 0.99999 and every lag == 0
         (float32 round-off over 6900 steps lands ~1e-6..1e-5; 1e-4 leaves
          headroom for a different GPU's reduction order)
  WARN   worst rel-amp <= 1e-2               (look at the per-station table)
  FAIL   anything else, or any structural mismatch

Usage:
  validate_forward.py --ref DIR --test DIR [--pattern '*.BXZ.semd']
                      [--out report.md] [--json report.json] [--top 10]
                      [--pass 1e-4] [--warn 1e-2] [--maxlag 50]
  exit code: 0 PASS, 1 WARN, 2 FAIL, 3 usage/structural error

Only numpy is required (no obspy), so it runs from the system python on the
GH200 login node before the cal_adjoint venv exists.
"""
import argparse, glob, json, os, sys, time
import numpy as np


_TEXT = frozenset(b" \t\r\n+-.0123456789EeDd")
_META = {}


def seismo_meta(dirpath):
    """dt / start_time for binary traces: the runner's seismo_meta.txt, else the
    solver log, looked up in the trace dir and its parent (cached per dir)."""
    d = os.path.realpath(dirpath)
    if d in _META:
        return _META[d]
    meta = None
    for dd in (d, os.path.dirname(d)):
        f = os.path.join(dd, "seismo_meta.txt")
        if os.path.isfile(f):
            kv = dict(l.strip().split("=", 1) for l in open(f) if "=" in l)
            if kv.get("start_time", "NA") != "NA":
                meta = (float(kv["dt"]), float(kv["start_time"])); break
        f = os.path.join(dd, "output_solver.txt")
        if os.path.isfile(f):
            dt = st = None
            for line in open(f, errors="replace"):
                s = line.strip()
                if s.startswith("time step:"): dt = float(s.split(":")[1].split()[0])
                elif s.startswith("start time:"): st = float(s.split(":")[1].split()[0])
                if dt is not None and st is not None:
                    meta = (dt, st); break
            if meta: break
    _META[d] = meta
    return meta


def read_semd(path):
    """SPECFEM trace, ascii (2-column t u) or binary (bare float32, SEISMO_FORMAT=bin)."""
    with open(path, "rb") as f:
        raw = f.read()
    head = raw[:64]
    if head and all(c in _TEXT for c in head) and b"\n" in raw[:4096]:
        a = np.fromstring(raw, sep=" ", dtype=np.float64)
        if a.size % 2:
            raise ValueError(f"{path}: odd token count {a.size}")
        a = a.reshape(-1, 2)
        return a[:, 0], a[:, 1]
    u = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    meta = seismo_meta(os.path.dirname(path))
    if meta is None:
        raise RuntimeError(f"{path}: binary trace but no seismo_meta.txt/output_solver.txt to rebuild its time axis")
    dt, t0 = meta
    return t0 + np.arange(u.size, dtype=np.float64) * dt, u


def best_lag(ref, tst, maxlag):
    """Lag (samples) maximising normalised cross-correlation within +-maxlag,
    plus the cc at that lag and at zero lag. Positive lag = test is LATE."""
    n = ref.size
    nr, nt = np.linalg.norm(ref), np.linalg.norm(tst)
    if nr == 0 or nt == 0:
        return 0, float("nan"), float("nan")
    L = min(maxlag, n - 1)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(tst, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    cc = np.concatenate([cc[-L:], cc[:L + 1]]) / (nr * nt)  # lags -L..+L
    k = int(np.argmax(cc))
    return k - L, float(cc[k]), float(cc[L])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference event dir (x86 result_M6/<event>)")
    ap.add_argument("--test", required=True, help="event dir under test (GH200 output)")
    ap.add_argument("--pattern", default="*.semd")
    ap.add_argument("--out", help="markdown report path")
    ap.add_argument("--json", dest="jsonout", help="json report path")
    ap.add_argument("--top", type=int, default=10, help="worst-N stations to list")
    ap.add_argument("--pass", dest="tol_pass", type=float, default=1e-4)
    ap.add_argument("--warn", dest="tol_warn", type=float, default=1e-2)
    ap.add_argument("--cc-pass", dest="cc_pass", type=float, default=0.99999)
    ap.add_argument("--maxlag", type=int, default=50, help="lag search half-window (samples)")
    ap.add_argument("--label", default="", help="free-text label for the report header")
    a = ap.parse_args()

    ref_files = {os.path.basename(p): p for p in glob.glob(os.path.join(a.ref, a.pattern))}
    tst_files = {os.path.basename(p): p for p in glob.glob(os.path.join(a.test, a.pattern))}
    if not ref_files:
        print(f"FATAL: no {a.pattern} under {a.ref}", file=sys.stderr); return 3
    if not tst_files:
        print(f"FATAL: no {a.pattern} under {a.test}", file=sys.stderr); return 3
    common = sorted(set(ref_files) & set(tst_files))
    only_ref = sorted(set(ref_files) - set(tst_files))
    only_tst = sorted(set(tst_files) - set(ref_files))

    rows, structural = [], []
    t_start = time.time()
    for name in common:
        tr, ur = read_semd(ref_files[name])
        tt, ut = read_semd(tst_files[name])
        if tr.size != tt.size:
            structural.append(f"{name}: NSTEP {tr.size} vs {tt.size}"); continue
        dt_r = float(np.median(np.diff(tr))) if tr.size > 1 else 0.0
        dt_t = float(np.median(np.diff(tt))) if tt.size > 1 else 0.0
        # ascii time columns are float32 (7.6e-6 s resolution at t=100 s), so compare
        # the axis by its end points rather than a per-step dt: 1e-6 s at t0, 1e-3 s at the end
        if abs(tr[0] - tt[0]) > 1e-6 or abs(tr[-1] - tt[-1]) > 1e-3:
            structural.append(f"{name}: time axis differs (t0 {tr[0]} vs {tt[0]}, t_end {tr[-1]} vs {tt[-1]})"); continue
        amax = float(np.max(np.abs(ur)))
        d = ut - ur
        if amax == 0.0:
            rel_amp = 0.0 if np.max(np.abs(d)) == 0.0 else float("inf")
            rel_l2 = rel_amp
        else:
            rel_amp = float(np.max(np.abs(d)) / amax)
            rel_l2 = float(np.linalg.norm(d) / np.linalg.norm(ur))
        lag, cc_best, cc0 = best_lag(ur, ut, a.maxlag)
        rows.append(dict(station=name, amax=amax, rel_amp=rel_amp, rel_l2=rel_l2,
                         cc0=cc0, cc_best=cc_best, lag=lag, lag_s=lag * dt_r, dt=dt_r, nstep=int(tr.size)))
    elapsed = time.time() - t_start

    if not rows:
        print("FATAL: no comparable traces", file=sys.stderr)
        for s in structural: print("  " + s, file=sys.stderr)
        return 3

    ra = np.array([r["rel_amp"] for r in rows]); rl = np.array([r["rel_l2"] for r in rows])
    cc = np.array([r["cc0"] for r in rows if np.isfinite(r["cc0"])])
    lags = np.array([r["lag"] for r in rows])
    summ = dict(
        n_ref=len(ref_files), n_test=len(tst_files), n_common=len(common), n_compared=len(rows),
        only_ref=only_ref, only_test=only_tst, structural=structural,
        nstep=rows[0]["nstep"], dt=rows[0]["dt"],
        rel_amp_max=float(ra.max()), rel_amp_p99=float(np.percentile(ra, 99)), rel_amp_median=float(np.median(ra)),
        rel_l2_max=float(rl.max()), rel_l2_median=float(np.median(rl)),
        cc0_min=float(cc.min()) if cc.size else float("nan"),
        cc0_median=float(np.median(cc)) if cc.size else float("nan"),
        lag_abs_max=int(np.abs(lags).max()), n_nonzero_lag=int((lags != 0).sum()),
        bitwise=bool(ra.max() == 0.0), seconds=elapsed,
    )
    if structural or only_ref or only_tst:
        verdict = "FAIL"
    elif summ["rel_amp_max"] <= a.tol_pass and summ["cc0_min"] >= a.cc_pass and summ["lag_abs_max"] == 0:
        verdict = "PASS"
    elif summ["rel_amp_max"] <= a.tol_warn:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    summ["verdict"] = verdict

    worst = sorted(rows, key=lambda r: -r["rel_amp"])[: a.top]
    lines = []
    lines.append(f"### Forward validation: {a.label or os.path.basename(a.test.rstrip('/'))}")
    lines.append("")
    lines.append(f"- ref : `{a.ref}`")
    lines.append(f"- test: `{a.test}`")
    lines.append(f"- traces: ref {summ['n_ref']}, test {summ['n_test']}, compared {summ['n_compared']}"
                 f" (NSTEP {summ['nstep']}, DT {summ['dt']:.4g} s)")
    if only_ref: lines.append(f"- **only in ref ({len(only_ref)})**: {' '.join(only_ref[:8])}{' ...' if len(only_ref) > 8 else ''}")
    if only_tst: lines.append(f"- **only in test ({len(only_tst)})**: {' '.join(only_tst[:8])}{' ...' if len(only_tst) > 8 else ''}")
    for s in structural: lines.append(f"- **structural**: {s}")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| max rel. amplitude diff (worst trace) | {summ['rel_amp_max']:.3e} |")
    lines.append(f"| rel. amplitude diff p99 / median | {summ['rel_amp_p99']:.3e} / {summ['rel_amp_median']:.3e} |")
    lines.append(f"| rel. L2 diff max / median | {summ['rel_l2_max']:.3e} / {summ['rel_l2_median']:.3e} |")
    lines.append(f"| zero-lag cc min / median | {summ['cc0_min']:.7f} / {summ['cc0_median']:.7f} |")
    lines.append(f"| max abs best-lag (samples) / traces with lag != 0 | {summ['lag_abs_max']} / {summ['n_nonzero_lag']} |")
    lines.append(f"| bitwise identical | {'yes' if summ['bitwise'] else 'no'} |")
    lines.append(f"| **verdict** (pass<= {a.tol_pass:g}, warn<= {a.tol_warn:g}, cc>= {a.cc_pass}) | **{verdict}** |")
    lines.append("")
    lines.append(f"Worst {len(worst)} traces by relative amplitude difference:")
    lines.append("")
    lines.append("| trace | max abs ref | rel amp | rel L2 | cc0 | lag |")
    lines.append("|---|---|---|---|---|---|")
    for r in worst:
        lines.append(f"| {r['station']} | {r['amax']:.3e} | {r['rel_amp']:.3e} | {r['rel_l2']:.3e} | {r['cc0']:.7f} | {r['lag']} |")
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        with open(a.out, "w") as f: f.write(report)
    if a.jsonout:
        with open(a.jsonout, "w") as f: json.dump(dict(summary=summ, traces=rows), f, indent=1)
    return {"PASS": 0, "WARN": 1, "FAIL": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
