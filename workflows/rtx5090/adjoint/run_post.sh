#!/bin/bash
# ite3_gpu/adjoint/run_post.sh -- kernels -> gradient.  clip+mask -> sum -> smooth.
#
# Replaces ite3's run_post_python_anvil.slurm + run_smooth_group_anvil.slurm.
# The three stages, and what each is for:
#
#   1. CLIP + RECEIVER MASK  (clip_iso.py, CPU, multiprocessing)
#      Clamps each kernel to +-CLIP_VALUE and multiplies in a per-event Gaussian
#      notch around that event's receivers. The notch matters: the near-receiver
#      singularity is a numerical artefact of the point source/receiver, not
#      structure, and without masking it accumulates coherently across events
#      into the global gradient maximum.
#
#   2. PRECONDITIONED SUM  (sum_precond.py, CPU)
#      Sums the clipped kernels over events and divides by the summed approximate
#      Hessian (normalised, thresholded at 1e-3) -- the standard preconditioner.
#
#   3. SMOOTH  (xsmooth_sem, GPU)
#      Gaussian smoothing on the SEM mesh, sigma_h=2500 m / sigma_v=200 m.
#      THIS IS THE ONE POST-PROCESSING TOOL WITH A CUDA PATH in SPECFEM 4.1.1 --
#      it takes GPU_MODE as a 6th positional argument. On Anvil this was a
#      40-minute 10-node MPI job; here it runs on the 5090. Note the GPU path
#      uses a brute-force neighbour search rather than the CPU kd-tree, so it is
#      compute-heavy but embarrassingly parallel -- a good fit. Set SMOOTH_GPU=0
#      in config.sh to fall back to the CPU path if it runs out of VRAM.
#
# Only alpha (Vp) and beta (Vs) are smoothed; rho is derived downstream via
# RHO_SCALING and its kernel is never used.
#
# Usage:  ./run_post.sh  [--force] [--skip-smooth]
# Resumable via per-stage sentinels.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=post
source "$ROOT/lib/common.sh"
source "$ROOT/lib/solver.sh"

FORCE=0; SKIP_SMOOTH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --skip-smooth) SKIP_SMOOTH=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done

ADJ_WORK="$WORKBASE/adjoint"
# One definition, from config.sh -- run_update.sh's --grad default reads the same
# variables, so they cannot drift apart.
CONDI="$KERNEL_SUM"
SMOOTH="$KERNEL_SMOOTH"
NJOBS=$(nproc)
resolve_python

banner "POST-PROCESSING  clip -> sum -> smooth"

[[ -d "$ORI_KERNELS" ]] || die "no $ORI_KERNELS -- run ./run_adjoint.sh first"
NEV=$(find "$ORI_KERNELS" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | wc -l)
[[ "$NEV" -gt 0 ]] || die "$ORI_KERNELS is empty"
log "events with kernels: $NEV"
log "clip=$CLIP_VALUE  mask=$MASK (sigma_h=$SIGMA_H m, sigma_v=$SIGMA_V m)"
log "cpu jobs: $NJOBS"

[[ "$FORCE" -eq 1 ]] && { clear_done post_clip; clear_done post_sum; clear_done post_smooth; }

# clip_iso.py resolves each event's receivers with a RELATIVE glob
# (group_*/RUN/<event>/OUTPUT_FILES/output_list_stations.txt), so CWD matters.
cd "$ADJ_WORK" || die "cannot cd $ADJ_WORK"

# ---------------------------------------------------------------------------
# 1. clip + mask
# ---------------------------------------------------------------------------
mask_args=()
if [[ "$MASK" == "1" ]]; then
  # --topo-dir must hold proc*_external_mesh.bin so the mask knows GLL coords.
  mask_args=( --mask --topo-dir "$ADJ_WORK/solver/DATABASES_MPI"
              --sigma-h "$SIGMA_H" --sigma-v "$SIGMA_V" )
  [[ -e "$ADJ_WORK/solver/DATABASES_MPI/proc000000_external_mesh.bin" ]] \
    || die "receiver mask needs proc*_external_mesh.bin in $ADJ_WORK/solver/DATABASES_MPI"
fi

mkdir -p "$KERNELS"
for ev in "$ORI_KERNELS"/*; do
  [[ -e "$ev" ]] || continue
  mkdir -p "$KERNELS/$(basename "$ev")"
done

step post_clip "clip kernels to +-$CLIP_VALUE${MASK:+ + receiver mask}" -- \
  "$PYBIN" "$HERE/clip_iso.py" \
    --clip "$CLIP_VALUE" \
    --ori  "$ORI_KERNELS" \
    --out  "$KERNELS" \
    --nproc "$NPROC" \
    --jobs "$NJOBS" \
    "${mask_args[@]}" \
  || die "clip stage failed -- see $LOGFILE"

# ---------------------------------------------------------------------------
# 2. preconditioned sum
# ---------------------------------------------------------------------------
step post_sum "sum kernels, precondition by the approximate Hessian" -- \
  "$PYBIN" "$HERE/sum_precond.py" \
    --input      "$KERNELS" \
    --hess-input "$ORI_KERNELS" \
    --out        "$CONDI" \
    --nproc      "$NPROC" \
    --jobs       "$NJOBS" \
  || die "sum stage failed -- see $LOGFILE"

for k in alpha beta; do
  [[ -e "$CONDI/proc000000_${k}_kernel_clip.bin" ]] \
    || die "sum finished but $CONDI/proc000000_${k}_kernel_clip.bin is missing"
done
log "summed gradient -> $CONDI"

# ---------------------------------------------------------------------------
# 3. smooth (GPU)
# ---------------------------------------------------------------------------
if [[ "$SKIP_SMOOTH" -eq 1 ]]; then
  log "--skip-smooth: stopping before smoothing"
  exit 0
fi

[[ -x "$ROOT/bin/xsmooth_sem" ]] || die "xsmooth_sem not built -- run ./1_build_specfem_gpu.sh"

# Smoothing runs on the GPU, so it takes the same lock as every other GPU stage.
# (Re-entrant: a no-op when run_loop already holds it.)
take_lock

# xsmooth_sem reads the mesh from LOCAL_PATH (./DATABASES_MPI) relative to CWD,
# so it runs in the solver workdir where the mesh already lives.
SWD="$ADJ_WORK/solver"
[[ -e "$SWD/DATABASES_MPI/proc000000_external_mesh.bin" ]] \
  || die "no mesh in $SWD/DATABASES_MPI -- run ./run_adjoint.sh first"
mkdir -p "$SMOOTH"

GPUFLAG=$( [[ "$SMOOTH_GPU" -eq 1 ]] && echo ".true." || echo ".false." )
log "smoothing with sigma_h=$SMOOTH_SIGMA_H m  sigma_v=$SMOOTH_SIGMA_V m  GPU=$GPUFLAG"

if ! is_done post_smooth || [[ "$FORCE" -eq 1 ]]; then
  t0=$(date +%s)
  if [[ "${SMOOTH_COMBINED:-1}" -eq 1 ]]; then
    # xsmooth_sem accepts a COMMA-DELIMITED kernel list (smooth_sem.F90:284,
    # parse_kernel_names -> nker) and carries nker through the whole pipeline,
    # GPU path included (prepare_smooth_gpu(...,nker) at :793). The expensive
    # part is the neighbour search over the mesh, which is done ONCE regardless
    # of nker -- so smoothing alpha and beta together is close to half the cost
    # of two separate invocations. Memory grows only as NGLL3*NSPEC*nker
    # (~100 MB per kernel here), which is nothing on a 32 GB card.
    klist=$(tr ' ' ',' <<<"$SMOOTH_KERNELS")
    log "  xsmooth_sem $klist  (single pass, $(wc -w <<<"$SMOOTH_KERNELS") kernels)"
    ( cd "$SWD" && gpu_run "$NPROC" "$ROOT/bin/xsmooth_sem" \
        "$SMOOTH_SIGMA_H" "$SMOOTH_SIGMA_V" "$klist" "$CONDI/" "$SMOOTH" "$GPUFLAG" ) \
      >>"$LOGFILE" 2>&1 || die "xsmooth_sem failed on '$klist'.
     If this is an out-of-memory error, first try SMOOTH_COMBINED=0 in config.sh
     (one kernel at a time), then SMOOTH_GPU=0 (CPU kd-tree path -- much slower
     but far less memory)."
  else
    for kname in $SMOOTH_KERNELS; do
      log "  xsmooth_sem $kname"
      ( cd "$SWD" && gpu_run "$NPROC" "$ROOT/bin/xsmooth_sem" \
          "$SMOOTH_SIGMA_H" "$SMOOTH_SIGMA_V" "$kname" "$CONDI/" "$SMOOTH" "$GPUFLAG" ) \
        >>"$LOGFILE" 2>&1 || die "xsmooth_sem failed on $kname.
     If this is an out-of-memory error, set SMOOTH_GPU=0 in config.sh to use the
     CPU kd-tree path instead (slower, but far less memory)."
    done
  fi
  t1=$(date +%s)
  log "smoothing finished in $(human_time $((t1 - t0)))"
  mark_done post_smooth
else
  log "SKIP  post_smooth -- already done"
fi

for k in alpha beta; do
  [[ -e "$SMOOTH/proc000000_${k}_kernel_clip_smooth.bin" ]] \
    || die "smoothing finished but $SMOOTH/proc000000_${k}_kernel_clip_smooth.bin is missing"
done
log "smoothed gradient -> $SMOOTH"

# ---------------------------------------------------------------------------
# archive this iteration's gradient
# ---------------------------------------------------------------------------
if [[ "$KERNEL_ARCHIVE" == "1" ]]; then
  ARCH="$WORKBASE/adjoint/m${ITER}to$((ITER + 1))"
  if [[ ! -e "$ARCH/proc000000_alpha_kernel_clip_smooth.bin" ]]; then
    mkdir -p "$ARCH"
    cp "$SMOOTH"/proc*_kernel_clip_smooth.bin "$ARCH/"
    log "archived gradient -> $ARCH  ($(du -sh "$ARCH" | cut -f1))"
  else
    log "gradient already archived at $ARCH"
  fi
fi

hr
log "POST-PROCESSING COMPLETE -> $SMOOTH"
log "next:  ./run_update.sh   (or let run_loop.sh line-search the step length)"
hr
