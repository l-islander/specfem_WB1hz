"""Select a line-search source group balancing PATH COVERAGE and PAIR COUNT.

    python tools/select_ls_group.py --result work/forward/result_M1 \\
        --n 40 --out events/c1_ls40h [--validate-old DIR --validate-new DIR]

WHY THIS EXISTS (2026-08-02). A line-search subset is an estimator of the
full-basis misfit change along the search direction. Two naive constructions
failed the same way on a real committed step (alpha=0.05, full-set truth
dChi=+4.72 -> gate REJECTED):

    top-40 by pair count:      predicted -30.4   (blind to sparse regions the
                                                  gradient itself under-weights)
    rank-stratified 40:        predicted -30.0   (rank diversity != path
                                                  diversity)
    THIS selector's hybrid-40: predicted  -5.7   (near-flat: would have steered
                                                  to a small, gate-safe step)

ALGORITHM
---------
1. For every source in the measured result dir, collect its post-cluster
   pairs (the OPT_*/... adjinfo) and their pair MIDPOINTS (source-receiver
   midpoint = where the measurement actually constrains structure).
2. Bin midpoints into cell_km x cell_km cells. The full basis occupies some
   set of cells; a good subset must touch ALL of them.
3. Greedy selection, deterministic: repeatedly take the source maximizing
   (newly covered cells, pair count, name) lexicographically -- coverage
   dominates until every cell is touched, after which the criterion
   degenerates to max-pairs fill. A concentration cap (max sources per
   SOURCE-location cell) prevents re-clustering during the fill phase.
4. Optional retro-validation: given two measured result dirs for a committed
   step (old model, rejected/accepted new model), print each candidate
   subset's predicted dChi next to the full-set truth.

The written members file + events dir are the reproducible record; the
greedy is deterministic (sorted pools, total ordering on the score).
"""
import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "figures"))
from geo2utm import geo2utm  # SPECFEM's Clarke-1866 UTM -- matches the mesh


def load_stations():
    sta = {}
    for p in (l.split() for l in open(os.path.join(ROOT, "DATA/STATIONS"))):
        if len(p) >= 4:
            e, n = geo2utm(float(p[3]), float(p[2]), 10)
            sta[f"{p[1]}.{p[0]}"] = (e / 1e3 - 449.321, n / 1e3 - 4837.704)
    return sta


def load_pairs(resdir):
    """{src: [(rec, chi), ...]} from a measured result dir."""
    out = {}
    for f in glob.glob(f"{resdir}/*_Z/OPT_*/*.adjinfo"):
        src = f.split(os.sep)[-3][:-2]
        rec = ".".join(os.path.basename(f).split(".")[:2])
        out.setdefault(src, []).append((rec, float(open(f).read().split()[1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=os.path.join(ROOT, "work/forward/result_M1"),
                    help="measured result dir defining the basis (OPT_* adjinfo)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--cell-km", type=float, default=18.0,
                    help="midpoint-coverage cell size")
    ap.add_argument("--cap-frac", type=float, default=0.12,
                    help="max fraction of the subset allowed from one "
                         "source-location cell (anti-clustering)")
    ap.add_argument("--out", default=None, help="events dir to write (else dry run)")
    ap.add_argument("--validate-old", default=None)
    ap.add_argument("--validate-new", default=None)
    args = ap.parse_args()

    sta = load_stations()
    P = load_pairs(args.result)
    srcs = sorted(P)
    CELL = args.cell_km

    def cell(pt):
        return (int(pt[0] // CELL), int(pt[1] // CELL))

    npairs = {s: len(P[s]) for s in srcs}
    cover = {s: {cell(((sta[s][0] + sta[r][0]) / 2, (sta[s][1] + sta[r][1]) / 2))
                 for r, _ in P[s] if s in sta and r in sta}
             for s in srcs}
    allcells = set().union(*cover.values())
    total = sum(npairs.values())
    print(f"basis: {total} pairs, {len(srcs)} sources, {len(allcells)} midpoint cells")

    # ---- deterministic greedy -------------------------------------------
    cap = max(2, int(args.n * args.cap_frac))
    chosen, covered, cellcount = [], set(), {}
    pool = list(srcs)
    while len(chosen) < args.n and pool:
        best = max(
            (s for s in pool
             if cellcount.get(cell(sta[s]), 0) < cap or not sta.get(s)),
            key=lambda s: (len(cover[s] - covered), npairs[s], s),
            default=None)
        if best is None:
            break
        chosen.append(best)
        covered |= cover[best]
        cellcount[cell(sta[best])] = cellcount.get(cell(sta[best]), 0) + 1
        pool.remove(best)

    pp = sum(npairs[s] for s in chosen)
    print(f"selected {len(chosen)}: {pp} pairs ({pp/total*100:.1f}%), "
          f"cell coverage {len(covered)/len(allcells)*100:.1f}%")

    # ---- optional retro-validation --------------------------------------
    if args.validate_old and args.validate_new:
        old, new = load_pairs(args.validate_old), load_pairs(args.validate_new)
        def chi(res, mem): return sum(c for s in mem if s in res for _, c in res[s])
        d_full = chi(new, srcs) - chi(old, srcs)
        d_sub = chi(new, chosen) - chi(old, chosen)
        print(f"validation: full dChi={d_full:+.2f}   subset dChi={d_sub:+.2f} "
              f"({'consistent' if d_full * d_sub >= 0 else 'SIGN MISMATCH'})")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for s in chosen:
            d = os.path.join(args.out, f"{s}_Z")
            if not os.path.isdir(d):
                shutil.copytree(os.path.join(ROOT, f"events/allforward_all/{s}_Z"), d)
        open(args.out.rstrip("/") + "_members.txt", "w").write("\n".join(chosen) + "\n")
        print(f"wrote {args.out} + members file")


if __name__ == "__main__":
    main()
