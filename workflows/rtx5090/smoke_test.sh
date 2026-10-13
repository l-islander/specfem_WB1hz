#!/bin/bash
# ite3_gpu/smoke_test.sh -- exercise the WHOLE chain on 2 events, in minutes.
#
# The point: run_loop.sh on 99 events can spend hours before reaching a stage
# that was broken from the start. This drives every stage in order on a tiny
# subset, so a wiring problem surfaces immediately.
#
#   forward(2 events) -> cal_adjoint -> kernels(2) -> clip/sum/smooth -> update
#
# It writes into a SEPARATE scratch area ($WORKBASE/smoke) and a throwaway model
# dir, so it cannot disturb a real run's state, sentinels, or L-BFGS history.
#
# A PASS means the plumbing works end to end. It does NOT mean the inversion is
# scientifically right -- 2 events give a meaningless gradient. Judge the science
# from the full loop's misfit series.
#
# Usage:  ./smoke_test.sh [--events N]   (default 2)
#         ./smoke_test.sh --keep         keep the scratch dir for inspection

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"

NEV=2; KEEP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --events) NEV="$2"; shift ;;
    --keep) KEEP=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac; shift
done

# Redirect ALL workflow state into a sandbox. Every script derives its paths from
# these two, so this is enough to fully isolate the run.
export WORKBASE="$WORKBASE/smoke"
# Re-source config.sh so every WORKBASE-derived path (KERNELS, KERNEL_SMOOTH,
# ORI_KERNELS, ADJSRC ...) points INTO the sandbox. Sub-scripts re-source it
# themselves and would sandbox correctly regardless, but this script's own
# references -- notably --grad "$KERNEL_SMOOTH" in phase 5 -- would otherwise
# still name the real run's directories.
source "$ROOT/config.sh"
SMOKE_OUT="$WORKBASE/M_smoke"          # models are READ from the real MODELBASE, written here
mkdir -p "$WORKBASE"

STAGE=smoke
source "$ROOT/lib/common.sh"

banner "SMOKE TEST -- $NEV events through the full chain"
log "sandbox: $WORKBASE"
log "NOTE: a $NEV-event gradient is scientifically meaningless. This tests PLUMBING."

PASS=0; FAIL=0
phase() {
  local name="$1"; shift
  hr; log ">>> $name"
  local t0; t0=$(date +%s)
  if "$@"; then
    log "<<< $name PASSED ($(human_time $(( $(date +%s) - t0 ))))"; PASS=$((PASS+1)); return 0
  else
    warn "<<< $name FAILED"; FAIL=$((FAIL+1)); return 1
  fi
}

RES="$WORKBASE/forward/result_smoke"

phase "1/5 forward" \
  bash "$ROOT/forward/run_forward.sh" --model "$ORIMODEL" --events "$SRC_SET" \
       --out "$RES" --limit "$NEV" \
  || die "forward failed -- nothing downstream can work. Check log/forward.log.
     Most likely: the CUDA build, the mesh, or the model. Try ./4_bench.sh first."

phase "2/5 cal_adjoint" \
  bash "$ROOT/cal_adjoint/run_caladj_local.sh" --full --datadir "$RES" \
  || warn "cal_adjoint failed. Common causes: venv not built, observed data not
     staged, or no observed data for these particular 2 sources (try --events 5)."

if [[ "$FAIL" -eq 0 ]]; then
  phase "3/5 kernels" \
    bash "$ROOT/adjoint/run_adjoint.sh" --model "$ORIMODEL" --limit "$NEV" \
    || warn "kernel stage failed -- see log/adjoint.log"
fi

if [[ "$FAIL" -eq 0 ]]; then
  phase "4/5 clip/sum/smooth" \
    bash "$ROOT/adjoint/run_post.sh" \
    || warn "post-processing failed -- see log/post.log"
fi

if [[ "$FAIL" -eq 0 ]]; then
  phase "5/5 L-BFGS update" \
    bash "$ROOT/adjoint/run_update.sh" --model "$ORIMODEL" --grad "$KERNEL_SMOOTH" \
         --out "$SMOKE_OUT" --state "$WORKBASE/lbfgs_state_smoke" --step 0.01 \
    || warn "update failed -- see log/update.log"
fi

hr
if [[ "$FAIL" -eq 0 ]]; then
  log "SMOKE TEST PASSED -- all $PASS stages completed."
  log ""
  log "The plumbing works. Now:"
  log "  ./4_bench.sh      time one event properly and extrapolate the loop"
  log "  ./run_loop.sh     the real inversion"
  log ""
  log "Sanity numbers from this run:"
  [[ -s "$ROOT/cal_adjoint/misfit_total.txt" ]] && \
    log "  misfit (2 events): $(cat "$ROOT/cal_adjoint/misfit_total.txt")"
  [[ -d "$SMOKE_OUT" ]] && \
    log "  update wrote: $(count_procfiles "$SMOKE_OUT" "_vp.bin") vp / $(count_procfiles "$SMOKE_OUT" "_vs.bin") vs / $(count_procfiles "$SMOKE_OUT" "_rho.bin") rho files"
  rc=0
else
  warn "SMOKE TEST FAILED -- $PASS passed, $FAIL failed. Fix before run_loop.sh."
  rc=1
fi

if [[ "$KEEP" -eq 1 ]]; then
  log "sandbox kept at $WORKBASE (--keep)"
else
  log "removing sandbox $WORKBASE (use --keep to inspect it)"
  rm -rf "$WORKBASE"
fi
hr
exit $rc
