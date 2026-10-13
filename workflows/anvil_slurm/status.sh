#!/bin/bash
# anvil_cpu/status.sh -- where is the inversion? Read-only. `-w [sec]` to watch.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCH=0; INTERVAL=60; [[ "${1:-}" == "-w" ]] && { WATCH=1; [[ -n "${2:-}" ]] && INTERVAL="$2"; }
show() {
  source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh" >/dev/null 2>&1
  local B=$'\033[1m' D=$'\033[2m' G=$'\033[32m' N=$'\033[0m'; [[ -t 1 ]] || { B=""; D=""; G=""; N=""; }
  printf '%s=== anvil_cpu  %s  %s  plan %ss NPROC=%s x%s ===%s\n' "$B" "$(hostname -s)" "$(date '+%F %T')" "$MESH_TARGET_S" "$NPROC" "$ARRAY_THROTTLE" "$N"
  local lp; lp=$(pgrep -f 'bash .*run_loop\.sh' | head -1)
  [[ -n "$lp" ]] && printf ' loop      : %sRUNNING%s pid %s up %s\n' "$G" "$N" "$lp" "$(ps -o etime= -p "$lp" | tr -d ' ')" || printf ' loop      : %snot running%s\n' "$D" "$N"
  if command -v squeue >/dev/null 2>&1; then
    printf ' queue     :\n'; squeue -u "$USER" -o '   %.10i %.14j %.9P %.2t %.10M %.6D %R' 2>/dev/null | head -20
  fi
  local ITER; ITER=$(grep -E '^ITER=' "$ROOT/config.sh" | head -1 | sed -E 's/^ITER=([0-9]+).*/\1/')
  printf '\n%s iteration : M%s -> M%s%s\n' "$B" "$ITER" "$((ITER+1))" "$N"
  local r="$WORKBASE/forward/result_M$ITER"
  [[ -d "$r" ]] && printf ' forward   : %s/%s events done%s\n' "$(find "$r" -maxdepth 2 -name .event_done | wc -l)" "$(list_events "$SRC_SET" 2>/dev/null | wc -l)" "$( [[ -f "$r/.stage_done" ]] && echo '  [stage done]' )"
  [[ -f "$ROOT/cal_adjoint/misfit_total.txt" ]] && printf ' last misfit: %s\n' "$(cat "$ROOT/cal_adjoint/misfit_total.txt")"
  local a="$WORKBASE/adjoint/group_0/RUN"
  [[ -d "$a" ]] && printf ' kernels   : %s done%s\n' "$(find "$a" -maxdepth 2 -name .kernel_done | wc -l)" "$( [[ -f "$WORKBASE/adjoint/.stage_done" ]] && echo '  [stage done]' )"
  [[ -f "$WORKBASE/linesearch/ls_points.txt" ]] && { printf ' linesearch:'; tr '\n' ';' < "$WORKBASE/linesearch/ls_points.txt"; echo; }
  [[ -f "$WORKBASE/linesearch/ls_alpha.txt" ]] && printf ' alpha*    : %s\n' "$(cat "$WORKBASE/linesearch/ls_alpha.txt")"
  printf ' scratch   : %s GB free on %s\n' "$(df -BG --output=avail "$WORKBASE" 2>/dev/null | tail -1 | tr -dc '0-9')" "$WORKBASE"
  [[ -f "$ROOT/misfit_history.txt" ]] && { printf ' history   :\n'; sed 's/^/   /' "$ROOT/misfit_history.txt" | tail -6; }
  [[ -f "$LOGDIR/run_loop.log" ]] && { printf ' log tail  :\n'; tail -4 "$LOGDIR/run_loop.log" | cut -c1-150 | sed 's/^/   /'; }
}
source "$ROOT/config.sh"; STAGE=status; source "$ROOT/lib/common.sh" >/dev/null 2>&1
if [[ "$WATCH" -eq 1 ]]; then while :; do clear; show; sleep "$INTERVAL"; done; else show; fi
