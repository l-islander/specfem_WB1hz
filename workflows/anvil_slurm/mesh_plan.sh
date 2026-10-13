# written by tools/plan_mesh.py 1.5 --apply  -- sourced by config.sh; do not edit by hand
MESH_TARGET_S=1.5
NEX_XI=144; NEX_ETA=256; NEX_Z=33
NPROC_XI=16; NPROC_ETA=32; NPROC=512
DT=0.011; NSTEP=9091
WAVEFIELD_GB_EST=173
CALADJ_BANDS_PLAN=1.5_4,2_6,3_10,6_18
SMOOTH_SIGMA_H_PLAN=1500
# expected: Tmin~1.48 s, band code HX, 4 nodes/event, iteration ~19047 SU at the planner's core rate
