#!/usr/bin/env python3
"""Parabolic line-search fit: read '<step> <misfit>' lines, print optimal step alpha*.

alpha* = vertex of the least-squares quadratic, clamped to [0, max_step].
Fallbacks:
  < 3 valid points     -> the step with the lowest misfit
  non-convex (c2<=0)   -> max_step if misfit is decreasing, else 0
Non-finite / NA misfits are dropped. The caller (run_loop) decides convergence
(alpha* < LS_CONV) and bracket meaning; this just returns the minimizer.

Usage:  polyfit.py points.txt    (or pipe on stdin)   # lines: "step misfit"
"""
import sys
import numpy as np


def main():
    src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    pts = []
    for line in src:
        f = line.split('#', 1)[0].split()
        if len(f) < 2:
            continue
        try:
            s, m = float(f[0]), float(f[1])
        except ValueError:
            continue
        if np.isfinite(s) and np.isfinite(m):
            pts.append((s, m))
    if not pts:
        sys.exit("polyfit: no valid (step,misfit) points")

    pts.sort()
    steps = np.array([p[0] for p in pts], float)
    mis = np.array([p[1] for p in pts], float)
    smax = float(steps.max())

    if len(pts) < 3:
        astar = float(steps[int(np.argmin(mis))])
    else:
        c2, c1, _c0 = np.polyfit(steps, mis, 2)     # m ~ c2 s^2 + c1 s + c0
        if c2 > 0:                                  # convex -> interior vertex
            astar = min(max(-c1 / (2 * c2), 0.0), smax)
        else:
            # Non-convex fit: near a misfit floor the valley is narrower than
            # the bracket, and a dip-rise-dip point set turns the LSQ parabola
            # concave -- the old `smax if c1<0 else 0` then discarded a trial
            # that genuinely beat the anchor (phase1_diag 2026-08-25: anchor
            # 873.04, 0.01 -> 865.20, alpha* came out 0). Trust the
            # measurements instead: take the lowest measured point (which is
            # still 0 when no trial beat the anchor).
            astar = float(steps[int(np.argmin(mis))])
    print(f"{astar:.6g}")


if __name__ == "__main__":
    main()
