#!/bin/bash
# anvil_cpu/smoke_test.sh -- 2 events through the WHOLE chain, sandboxed.
#
#   forward(2) -> cal_adjoint -> kernels(2) -> clip/sum/smooth -> update
# in $WORKBASE/smoke with its own sentinels, so nothing of a real run is touched.
# On Anvil that is ~6 SLURM jobs; with tools/mock_slurm on PATH (README "dry
# run on the 5090 box") it runs locally at a small NPROC. PASS = plumbing only.
#
# Usage: ./smoke_test.sh [--events N] [--keep] [--windir DIR]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
NEV=2; KEEP=0; WINDIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in --events) NEV="$2"; shift ;; --keep) KEEP=1 ;; --windir) WINDIR="$2"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;; *) echo "unknown option: $1" >&2; exit 2 ;; esac; shift
done
export WORKBASE="$WORKBASE/smoke"
source "$ROOT/config.sh"                      # re-derive every WORKBASE path into the sandbox
export STATEDIR="$WORKBASE/.state"            # sentinels sandboxed too
export ARRAY_THROTTLE="${SMOKE_THROTTLE:-$NEV}"
[[ -n "$WINDIR" ]] && export CALADJ_WINDIR="$WINDIR"
SMOKE_OUT="$WORKBASE/M_smoke"; mkdir -p "$WORKBASE"
STAGE=smoke; source "$ROOT/lib/common.sh"
banner "SMOKE TEST -- $NEV events, NPROC=$NPROC, sandbox $WORKBASE"
PASS=0; FAIL=0
phase() { local name="$1"; shift; log ">>> $name"; local t0; t0=$(date +%s)
  if "$@"; then log "<<< $name PASSED ($(human_time $(( $(date +%s) - t0 ))))"; PASS=$((PASS+1)); return 0
  else warn "<<< $name FAILED"; FAIL=$((FAIL+1)); return 1; fi; }
RES="$WORKBASE/forward/result_smoke"
phase "1/5 forward"      bash "$ROOT/forward/run_forward.sh" --model "$ORIMODEL" --events "$SRC_SET" --out "$RES" --limit "$NEV" || die "forward failed -- log/forward.log, log/slurm/"
phase "2/5 cal_adjoint"  bash "$ROOT/cal_adjoint/run_caladj.sh" --full --datadir "$RES" || warn "cal_adjoint failed -- venv? observed bands for this stage generated? windows?"
[[ "$FAIL" -eq 0 ]] && phase "3/5 kernels" bash "$ROOT/adjoint/run_adjoint.sh" --model "$ORIMODEL" --limit "$NEV"
[[ "$FAIL" -eq 0 ]] && phase "4/5 clip/sum/smooth" bash "$ROOT/adjoint/run_post.sh"
[[ "$FAIL" -eq 0 ]] && phase "5/5 L-BFGS update" bash "$ROOT/adjoint/run_update.sh" --model "$ORIMODEL" --grad "$KERNEL_SMOOTH" --out "$SMOKE_OUT" --state "$WORKBASE/lbfgs_state_smoke" --step 0.01
hr
if [[ "$FAIL" -eq 0 ]]; then
  log "SMOKE TEST PASSED -- $PASS stages. misfit ($NEV events): $(cat "$ROOT/cal_adjoint/misfit_total.txt" 2>/dev/null)"
  log "update wrote $(count_procfiles "$SMOKE_OUT" "_vs.bin") vs files (expect $NPROC)"
  if [[ "$KEEP" -eq 0 ]]; then rm -rf "$WORKBASE"; log "sandbox removed (--keep to inspect)"; else log "sandbox kept: $WORKBASE"; fi
  exit 0
else
  log "SMOKE TEST FAILED at stage $((PASS + 1)) -- sandbox kept at $WORKBASE"; exit 1
fi
