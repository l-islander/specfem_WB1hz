#!/bin/bash
# anvil_cpu/adjoint/run_update.sh -- L-BFGS model update, M_x + step -> M_{x+1}.
#
# lbfgs_update.py is verbatim from ite3/ite3_gpu (it always handled NPROC>1;
# the local port merely ran it at 1). It holds (2*MBFGS+6) float64 copies of
# the 2-parameter model vector: ~40 GB at the 1.5 s mesh, ~160 GB at 1 s.
# Runs as one single-node job; UPDATE_PARTITION=highmem for the 1 s mesh.
# Outside SLURM it submits itself and waits; inside, it runs.
#
# Usage: ./run_update.sh [--model DIR] [--out DIR] [--grad DIR] [--state DIR] [--step A] [--no-clamp] [--tag T]

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=update
source "$ROOT/lib/common.sh"; source "$ROOT/lib/slurm.sh"

IN_MODEL="$ORIMODEL"; OUT_MODEL="$OUTMODEL"; GRAD="$KERNEL_SMOOTH"; STATE="$LBFGS_STATE"; STEP="$MAXPERT"; NOCLAMP=0; TAG=""
ARGS_IN=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) IN_MODEL="$2"; shift ;; --out) OUT_MODEL="$2"; shift ;; --grad) GRAD="$2"; shift ;;
    --state) STATE="$2"; shift ;; --step) STEP="$2"; shift ;; --tag) TAG="$2"; shift ;;
    --no-clamp) NOCLAMP=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done
: "${TAG:=$(basename "$IN_MODEL")}"

if [[ -e "$OUT_MODEL/.done" ]]; then log "$OUT_MODEL/.done already present -- nothing to do"; exit 0; fi

if ! in_slurm; then
  sbatch_wait "update_$TAG" 1 1 "$SERVICE_WALL" "${UPDATE_PARTITION:-${PART_COMPUTE:?}}" -- \
    bash "$0" "${ARGS_IN[@]}" || die "update job failed -- see $LOGDIR/slurm/"
  [[ -e "$OUT_MODEL/.done" ]] || die "update job ended but $OUT_MODEL/.done is absent"
  exit 0
fi

# ---- inside the job -------------------------------------------------------
# REWINDOW => objective changes every iteration => no cross-basis curvature pairs
if [[ "${REWINDOW:-0}" -eq 1 && -d "$STATE" ]]; then rm -rf "$STATE"/*; fi
banner "L-BFGS UPDATE  $(basename "$IN_MODEL") -> $(basename "$OUT_MODEL")  step=$STEP$( [[ "$NOCLAMP" -eq 1 ]] && echo ' (no-clamp)')"
[[ -f "$IN_MODEL/.done" ]] || die "input model $IN_MODEL has no .done sentinel"
last=$(printf 'proc%06d' $((NPROC - 1)))
for f in "$GRAD/${last}_alpha_kernel_clip_smooth.bin" "$GRAD/${last}_beta_kernel_clip_smooth.bin" \
         "$IN_MODEL/${last}_vp.bin" "$IN_MODEL/${last}_vs.bin" "$IN_MODEL/${last}_rho.bin"; do
  [[ -e "$f" ]] || die "missing required input: $f"
done
mkdir -p "$OUT_MODEL" "$STATE"; rm -f "$OUT_MODEL/.done"
VECMB=$(( NEX_XI * NEX_ETA * NEX_Z * 125 * 2 * 8 / 1048576 ))
log "model vector ~${VECMB} MB (float64); L-BFGS m=$MBFGS -> peak ~$(( VECMB * (2 * MBFGS + 6) / 1024 )) GB on $(hostname)"
resolve_python
NT="${SLURM_CPUS_ON_NODE:-$(nproc)}"; export OMP_NUM_THREADS="$NT" OPENBLAS_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT"
ARGS=( --nproc "$NPROC" --grad "$GRAD" --model "$IN_MODEL" --out "$OUT_MODEL" --state "$STATE"
       --m "$MBFGS" --max-pert "$STEP" --params alpha,beta --tag "$TAG" )
[[ "$NOCLAMP" -eq 1 ]] && ARGS+=( --no-clamp )
t0=$(date +%s)
"$PYBIN" "$HERE/lbfgs_update.py" "${ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
rc=${PIPESTATUS[0]}; [[ "$rc" -eq 0 ]] || die "lbfgs_update.py exited $rc"
for p in vp vs rho; do
  n=$(count_procfiles "$OUT_MODEL" "_$p.bin"); [[ "$n" -eq "$NPROC" ]] || die "update wrote $n proc*_$p.bin, expected $NPROC"
done
: > "$OUT_MODEL/.done"
log "update complete in $(human_time $(( $(date +%s) - t0 ))) -> $OUT_MODEL/.done"
