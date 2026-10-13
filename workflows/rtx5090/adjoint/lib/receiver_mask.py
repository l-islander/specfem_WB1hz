#!/usr/bin/env python3
"""Per-event receiver taper (anisotropic Gaussian notch).

Borrows the SPECFEM source-mask method (locate_sources.f90 calc_mask_source,
applied in sum_preconditioned_kernels.f90): one multiplicative Gaussian notch
per receiver, product over all of an event's receivers, applied to that event's
kernel before summing. 0 at the receiver, ->1 away from it; amplitude-independent
so it suppresses the near-receiver singularity regardless of its height.

Anisotropic adaptation for this mesh (h~1700 m, v~500 m elements) with all
receivers on the free surface:

    D2(x) = (dh / sigma_h)^2 + (dz / sigma_v)^2
    mask(x) = prod_receivers ( 1 - exp(-D2) )

dh is horizontal distance, dz vertical. With sigma_h=sigma_v this reduces to the
isotropic global mask. Receivers farther than `n_sigma` horizontal sigmas from a
proc's footprint contribute a factor ~1 and are skipped (the mask is built only
where it differs from 1).

read_station_coords parses an OUTPUT_FILES/output_list_stations.txt
(name network X Y Z) into an (R,3) array of receiver mesh coordinates.
"""
import numpy as np

_R = np.dtype("<f4")


def read_station_coords(path):
    """(R,3) float64 receiver coords from an output_list_stations.txt."""
    xs = []
    with open(path) as fh:
        for ln in fh:
            p = ln.split()
            if len(p) < 5:
                continue
            try:
                xs.append((float(p[2]), float(p[3]), float(p[4])))
            except ValueError:
                continue
    return np.asarray(xs, dtype=np.float64)


def build_mask(xg, yg, zg, rec_xyz, sigma_h, sigma_v, n_sigma=4.0):
    """Anisotropic Gaussian-notch mask for one proc / one event.

    xg,yg,zg : (N,) per-GLL coords for the proc (from mesh_coords).
    rec_xyz  : (R,3) receiver coords for the event.
    Returns float32 (N,) mask in [0,1]; exactly 1.0 (untouched) if no receiver
    is within n_sigma horizontal sigmas of the proc footprint.
    """
    mask = np.ones(xg.size, dtype=_R)
    if rec_xyz.size == 0:
        return mask

    # Cull receivers outside the proc's horizontal footprint + halo. (procs span
    # the full depth column, so no z culling.)
    halo = n_sigma * sigma_h
    xmin, xmax = xg.min() - halo, xg.max() + halo
    ymin, ymax = yg.min() - halo, yg.max() + halo
    sel = ((rec_xyz[:, 0] >= xmin) & (rec_xyz[:, 0] <= xmax) &
           (rec_xyz[:, 1] >= ymin) & (rec_xyz[:, 1] <= ymax))
    rec = rec_xyz[sel]
    if rec.shape[0] == 0:
        return mask

    inv_h2 = 1.0 / (sigma_h * sigma_h)
    inv_v2 = 1.0 / (sigma_v * sigma_v)
    xg64 = xg.astype(np.float64)
    yg64 = yg.astype(np.float64)
    zg64 = zg.astype(np.float64)
    acc = np.ones(xg.size, dtype=np.float64)
    for rx, ry, rz in rec:
        d2 = ((xg64 - rx) ** 2 + (yg64 - ry) ** 2) * inv_h2 \
            + (zg64 - rz) ** 2 * inv_v2
        acc *= (1.0 - np.exp(-d2))
    return acc.astype(_R)
