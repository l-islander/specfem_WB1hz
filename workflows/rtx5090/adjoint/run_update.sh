#!/bin/bash
# ite3_gpu/adjoint/run_update.sh -- L-BFGS model update.  M_x + step -> M_{x+1}.
#
# Replaces ite3's run_update_python_anvil.slurm. The science is untouched:
# lbfgs_update.py is copied verbatim from ite3. Only the job wrapper changes
# (no SLURM, no modules, local thread counts).
#
#   g  = smoothed gradient  d(misfit)/d(ln m)
#   r  = H_k g                                  (two-loop recursion)
#   d  = -r                                     (descent direction)
#   a  = max_pert / max|d|                      (--no-clamp: step IS max_pert)
#     or min(1, max_pert/max|d|)                (default trust-region cap)
#   vp  *= exp(a * d_alpha)
#   vs  *= exp(a * d_beta)
#   rho *= exp(a * 0.33 * d_beta)               (RHO_SCALING)
#
# TWO MODES, and the difference matters:
#
#   default        a = min(1, MAXPERT/max|d|)  -- a trust region. Used when you
#                  run this stage by hand without a line search.
#   --no-clamp     a = MAXPERT/max|d|          -- MAXPERT literally becomes the
#                  step length. This is what the line search needs: without it,
#                  once the L-BFGS direction is well scaled every trial step
#                  collapses to a=1 and all the candidate models come out
#                  identical, so the parabola is fitted to noise.
#
# THE L-BFGS STATE IS FRESH FOR THIS TREE. Curvature pairs are stored as
# concatenated per-rank vectors; a state built at NPROC=1250 has a different
# memory layout from one at NPROC=1 and CANNOT be reused. The first update here
# is therefore pure steepest descent, exactly as ite3's first update was.
#
# Usage:
#   ./run_update.sh                             # M$ITER -> M$((ITER+1)), capped
#   ./run_update.sh --step 0.023 --no-clamp     # commit a line-searched step
#   ./run_update.sh --out /path/to/M9 --state /tmp/throwaway   # trial model

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=update
source "$ROOT/lib/common.sh"

IN_MODEL="$ORIMODEL"
OUT_MODEL="$OUTMODEL"
GRAD="$KERNEL_SMOOTH"
STATE="$LBFGS_STATE"
STEP="$MAXPERT"
NOCLAMP=0
TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) IN_MODEL="$2"; shift ;;
    --out)   OUT_MODEL="$2"; shift ;;
    --grad)  GRAD="$2"; shift ;;
    --state) STATE="$2"; shift ;;
    --step)  STEP="$2"; shift ;;
    --tag)   TAG="$2"; shift ;;
    --no-clamp) NOCLAMP=1 ;;
    -h|--help) sed -n '2,34p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done
: "${TAG:=$(basename "$IN_MODEL")}"

# REWINDOW => the misfit functional changes EVERY iteration (fresh windows), so
# L-BFGS curvature pairs y = g_{k+1}-g_k mix objective drift into curvature --
# theoretically invalid (stage-2 M1->M2 near-null
# step was the first cross-basis direction). Under REWINDOW every update is
# preconditioned steepest descent: empty the state so no cross-basis pair can
# form (the post-commit snapshot append below is then harmless -- it is wiped
# again here before it can ever pair).
if [[ "${REWINDOW:-0}" -eq 1 && -d "$STATE" ]]; then
  rm -rf "$STATE"/*
fi

banner "L-BFGS UPDATE  $(basename "$IN_MODEL") -> $(basename "$OUT_MODEL")  step=$STEP$( [[ "$NOCLAMP" -eq 1 ]] && echo ' (no-clamp)')"

# ---------------------------------------------------------------------------
# preflight -- every one of these has bitten someone
# ---------------------------------------------------------------------------
[[ -f "$IN_MODEL/.done" ]] || die "input model $IN_MODEL has no .done sentinel"
last=$(printf 'proc%06d' $((NPROC - 1)))
for f in "$GRAD/${last}_alpha_kernel_clip_smooth.bin" \
         "$GRAD/${last}_beta_kernel_clip_smooth.bin" \
         "$IN_MODEL/${last}_vp.bin" "$IN_MODEL/${last}_vs.bin" "$IN_MODEL/${last}_rho.bin"; do
  [[ -e "$f" ]] || die "missing required input: $f"
done
[[ -f "$HERE/lbfgs_update.py" ]] || die "missing $HERE/lbfgs_update.py"

if [[ -e "$OUT_MODEL/.done" ]]; then
  log "$OUT_MODEL/.done already present -- update already committed, nothing to do"
  log "force a redo with: rm $OUT_MODEL/.done"
  exit 0
fi

mkdir -p "$OUT_MODEL" "$STATE"
# Clear the sentinel first so an interrupted run cannot leave a partial model
# looking complete.
rm -f "$OUT_MODEL/.done"

# The updater holds the full model vector (both parameters) as float64, plus the
# L-BFGS history. At NPROC=1 that vector is the same total length as at
# NPROC=1250 -- partitioning changes the file count, not the physics -- so the
# memory need is unchanged: roughly (2m + 6) copies at m=$MBFGS.
VECMB=$(( NEX_XI * NEX_ETA * NEX_Z * 125 * 2 * 8 / 1048576 ))
log "model vector ~${VECMB} MB (float64); L-BFGS m=$MBFGS -> peak ~$(( VECMB * (2 * MBFGS + 6) / 1024 )) GB"

resolve_python
NT=$(nproc)
export OMP_NUM_THREADS="$NT" OPENBLAS_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT"

ARGS=( --nproc "$NPROC"
       --grad  "$GRAD"
       --model "$IN_MODEL"
       --out   "$OUT_MODEL"
       --state "$STATE"
       --m     "$MBFGS"
       --max-pert "$STEP"
       --params alpha,beta
       --tag   "$TAG" )
[[ "$NOCLAMP" -eq 1 ]] && ARGS+=( --no-clamp )

log "$PYBIN lbfgs_update.py ${ARGS[*]}"
t0=$(date +%s)
if ! "$PYBIN" "$HERE/lbfgs_update.py" "${ARGS[@]}" 2>&1 | tee -a "$LOGFILE"; then
  die "lbfgs_update.py FAILED -- $OUT_MODEL left without a .done sentinel"
fi
rc=${PIPESTATUS[0]}
[[ "$rc" -eq 0 ]] || die "lbfgs_update.py exited $rc"
t1=$(date +%s)

# verify before blessing
for p in vp vs rho; do
  n=$(count_procfiles "$OUT_MODEL" "_$p.bin")
  [[ "$n" -eq "$NPROC" ]] || die "update wrote $n proc*_$p.bin, expected $NPROC"
done

: > "$OUT_MODEL/.done"
log "update complete in $(human_time $((t1 - t0))) -> $OUT_MODEL/.done"
hr
