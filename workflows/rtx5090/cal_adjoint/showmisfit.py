#!/usr/bin/env python3
"""Summarise adjoint misfits from per-source OPT_<band>/*.adjinfo files.

Each .adjinfo file is written by 3_calculateadj.py and holds two whitespace-
separated numbers:  <dt/tshift>  <misfit>  (see 3_calculateadj.py:146).

This script walks  config.DATADIR/<source>_Z/OPT_<band>/*.adjinfo  and reports,
per band and overall:

  * adjinfo count  - number of .adjinfo files found
  * window average - mean over every window (every .adjinfo file)
  * source average - mean of the per-source means (each source weighted equally)
  * total          - summed misfit / dt over all windows

Both dt (|tshift|) and misfit are reported.

Usage:
    python read_misfit.py [--dir DIR] [--band LABEL ...] [--per-source]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import config


def read_adjinfo(path: Path) -> tuple[float, float] | None:
    """Return (dt, misfit) from one .adjinfo file, or None if unparsable."""
    parts = path.read_text().split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def collect(datadir: Path, bands: list[str] | None):
    """records[band] -> list of (source, dt, misfit)."""
    records: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    skipped: list[Path] = []

    for workdir in sorted(datadir.glob("*_Z")):
        source = workdir.name.split("_")[0]
        for optdir in sorted(workdir.glob("OPT_*")):
            band = optdir.name[len("OPT_"):]
            if bands and band not in bands:
                continue
            for f in sorted(optdir.glob("*.adjinfo")):
                vals = read_adjinfo(f)
                if vals is None:
                    skipped.append(f)
                    continue
                dt, misfit = vals
                records[band].append((source, dt, misfit))
    return records, skipped


def summarise(label: str, rows: list[tuple[str, float, float]]) -> None:
    n = len(rows)
    if n == 0:
        print(f"\n=== {label} ===  (no adjinfo files)")
        return

    # dt reported as absolute time shift; misfit as written.
    dts = [abs(dt) for _, dt, _ in rows]
    misfits = [m for _, _, m in rows]

    # per-source means
    by_source_dt: dict[str, list[float]] = defaultdict(list)
    by_source_mf: dict[str, list[float]] = defaultdict(list)
    for src, dt, m in rows:
        by_source_dt[src].append(abs(dt))
        by_source_mf[src].append(m)
    n_src = len(by_source_mf)
    src_avg_dt = mean(mean(v) for v in by_source_dt.values())
    src_avg_mf = mean(mean(v) for v in by_source_mf.values())

    print(f"\n=== {label} ===")
    print(f"  adjinfo count : {n}")
    print(f"  sources       : {n_src}")
    print(f"  windows/source: {n / n_src:.2f}")
    print(f"  window avg    :  dt = {mean(dts):.6g}   misfit = {mean(misfits):.6g}")
    print(f"  source avg    :  dt = {src_avg_dt:.6g}   misfit = {src_avg_mf:.6g}")
    print(f"  total         :  dt = {sum(dts):.6g}   misfit = {sum(misfits):.6g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=config.DATADIR,
                    help=f"Workdir root holding <source>_Z/ (default: {config.DATADIR})")
    ap.add_argument("--band", nargs="+", default=None,
                    help="Limit to these band labels (default: all OPT_* found).")
    ap.add_argument("--per-source", action="store_true",
                    help="Also list the per-source mean misfit/dt within each band.")
    args = ap.parse_args()

    records, skipped = collect(args.dir, args.band)

    if not records:
        print(f"No .adjinfo files found under {args.dir}")
        return

    # All bands combined first, then per band.
    all_rows = [r for rows in records.values() for r in rows]
    summarise("ALL BANDS", all_rows)

    for band in sorted(records):
        summarise(f"band {band}", records[band])
        if args.per_source:
            by_src: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for src, dt, m in records[band]:
                by_src[src].append((abs(dt), m))
            for src in sorted(by_src):
                vals = by_src[src]
                print(f"      {src:<12} n={len(vals):<5} "
                      f"dt={mean(d for d, _ in vals):.6g}  "
                      f"misfit={mean(m for _, m in vals):.6g}")

    if skipped:
        print(f"\n[warn] skipped {len(skipped)} unparsable file(s):")
        for f in skipped[:10]:
            print(f"    {f}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")


if __name__ == "__main__":
    main()
