#!/bin/bash
# anvil_cpu/adjoint/run_post.sh -- kernels -> gradient: clip+mask -> sum -> smooth.
#
# Two SLURM jobs, submitted in order and waited on:
#   post_clipsum  one node, 128 python workers: clip_iso.py over (proc, event)
#                 tasks, then sum_precond.py per proc  (adjoint/post_job.sh)
#   post_smooth   NPROC ranks: xsmooth_sem, CPU kd-tree path, run in the model's
#                 DB dir where the mesh lives  (adjoint/smooth_job.sh)
# Sentinels are per stage in $STATEDIR (=.state/ite$ITER), as in ite3_gpu.
#
# Usage:  ./run_post.sh [--force] [--skip-smooth]

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=post
source "$ROOT/lib/common.sh"; source "$ROOT/lib/solver.sh"; source "$ROOT/lib/slurm.sh"

FORCE=0; SKIP_SMOOTH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;; --skip-smooth) SKIP_SMOOTH=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done
ADJ_WORK="$WORKBASE/adjoint"; CONDI="$KERNEL_SUM"; SMOOTH="$KERNEL_SMOOTH"
banner "POST-PROCESSING  clip -> sum -> smooth  (NPROC=$NPROC)"
[[ -d "$ORI_KERNELS" ]] || die "no $ORI_KERNELS -- run ./run_adjoint.sh first"
NEV=$(find "$ORI_KERNELS" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | wc -l)
[[ "$NEV" -gt 0 ]] || die "$ORI_KERNELS is empty"
MODEL_DIR=$(cat "$ADJ_WORK/.kernel_model" 2>/dev/null || echo "")
[[ -n "$MODEL_DIR" ]] || die "no $ADJ_WORK/.kernel_model stamp -- which model were the kernels computed on?"
DB=$(db_dir "$MODEL_DIR")
[[ -f "$DB/.done" ]] || die "database $DB missing (the receiver mask and xsmooth_sem need its mesh)"
log "events with kernels: $NEV   model: $(basename "$MODEL_DIR")   db: $DB"
[[ "$FORCE" -eq 1 ]] && { clear_done post_clip; clear_done post_sum; clear_done post_smooth; }

# --- 1+2 clip + sum (one node) --------------------------------------------
if is_done post_clip && is_done post_sum && [[ "$FORCE" -eq 0 ]]; then
  log "SKIP clip+sum -- already done"
else
  sbatch_wait "post_clipsum" 1 1 "$SERVICE_WALL" "${PART_COMPUTE:?}" -- \
    bash "$HERE/post_job.sh" "$DB" $( [[ "$FORCE" -eq 1 ]] && echo --force ) \
    || die "clip/sum job failed -- see $LOGDIR/slurm/ and $LOGDIR/post_job.log"
fi
for k in alpha beta; do
  [[ -e "$CONDI/$(printf 'proc%06d' $((NPROC-1)))_${k}_kernel_clip.bin" ]] \
    || die "sum finished but $CONDI is incomplete for ${k}_kernel_clip"
done
log "summed gradient -> $CONDI"
[[ "$SKIP_SMOOTH" -eq 1 ]] && { log "--skip-smooth: stopping before smoothing"; exit 0; }

# --- 3 smooth (NPROC ranks, CPU kd-tree) ----------------------------------
[[ -x "$ROOT/bin/xsmooth_sem" ]] || die "xsmooth_sem not built -- run ./1_build_specfem_cpu.sh"
mkdir -p "$SMOOTH"
if is_done post_smooth && [[ "$FORCE" -eq 0 ]]; then
  log "SKIP post_smooth -- already done"
else
  log "smoothing sigma_h=$SMOOTH_SIGMA_H m sigma_v=$SMOOTH_SIGMA_V m  (CPU, $NPROC ranks)"
  sbatch_wait "post_smooth" "$NODES_PER_EVENT" "$NPROC" "$SERVICE_WALL" "${PART_COMPUTE:?}" -- \
    bash "$HERE/smooth_job.sh" "$DB" "$CONDI" "$SMOOTH" \
    || die "xsmooth_sem job failed -- see $LOGDIR/slurm/"
  mark_done post_smooth
fi
for k in alpha beta; do
  [[ -e "$SMOOTH/$(printf 'proc%06d' $((NPROC-1)))_${k}_kernel_clip_smooth.bin" ]] \
    || die "smoothing finished but $SMOOTH is incomplete for ${k}"
done
log "smoothed gradient -> $SMOOTH"

if [[ "$KERNEL_ARCHIVE" == "1" ]]; then
  ARCH="$WORKBASE/adjoint/m${ITER}to$((ITER + 1))"
  if [[ ! -e "$ARCH/proc000000_alpha_kernel_clip_smooth.bin" ]]; then
    mkdir -p "$ARCH"
    find "$SMOOTH" -maxdepth 1 -name 'proc*_kernel_clip_smooth.bin' -print0 | xargs -0 cp -t "$ARCH/"
    log "archived gradient -> $ARCH  ($(du -sh "$ARCH" | cut -f1))"
  else log "gradient already archived at $ARCH"; fi
fi
hr; log "POST-PROCESSING COMPLETE -> $SMOOTH"; hr
