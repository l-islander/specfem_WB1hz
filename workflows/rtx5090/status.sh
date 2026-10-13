#!/bin/bash
# ite3_gpu/status.sh -- where is the inversion right now?
#
# Read-only. Safe to run at any time, including while run_loop.sh is working.
# Written for the "I just walked back to the machine" case: what iteration, what
# stage, how far through, is anything actually running, and what does the misfit
# series look like.
#
# Usage:  ./status.sh          one snapshot
#         ./status.sh -w       refresh every 20 s
#         ./status.sh -w 5     refresh every 5 s

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WATCH=0; INTERVAL=20
if [[ "${1:-}" == "-w" ]]; then WATCH=1; [[ -n "${2:-}" ]] && INTERVAL="$2"; fi

show() {
  source "$ROOT/config.sh"
  [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"

  local B=$'\033[1m' D=$'\033[2m' G=$'\033[32m' Y=$'\033[33m' R=$'\033[31m' N=$'\033[0m'
  [[ -t 1 ]] || { B=""; D=""; G=""; Y=""; R=""; N=""; }

  printf '%s========================================================%s\n' "$B" "$N"
  printf '%s ite3_gpu  --  %s  --  %s%s\n' "$B" "$(hostname)" "$(date '+%F %H:%M:%S')" "$N"
  printf '%s========================================================%s\n' "$B" "$N"

  # --- is anything running? ------------------------------------------------
  local loop_pid solver
  loop_pid=$(pgrep -f 'bash .*run_loop\.sh' | head -1)
  solver=$(pgrep -f 'xspecfem3D|xgenerate_databases|xmeshfem3D|xsmooth_sem' | wc -l)
  if [[ -n "$loop_pid" ]]; then
    printf ' loop      : %sRUNNING%s (pid %s, up %s)\n' "$G" "$N" "$loop_pid" \
      "$(ps -o etime= -p "$loop_pid" 2>/dev/null | tr -d ' ')"
  else
    printf ' loop      : %snot running%s\n' "$D" "$N"
  fi
  if [[ "$solver" -gt 0 ]]; then
    printf ' solver    : %s%d SPECFEM process(es) active%s\n' "$G" "$solver" "$N"
  else
    printf ' solver    : %sidle%s\n' "$D" "$N"
  fi

  # --- GPU ------------------------------------------------------------------
  if command -v nvidia-smi >/dev/null 2>&1; then
    local g
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits -i "${GPU_DEVICE:-0}" 2>/dev/null)
    [[ -n "$g" ]] && printf ' gpu       : %s%% util, %s / %s MiB, %s C\n' \
      "$(cut -d, -f1 <<<"$g" | tr -d ' ')" "$(cut -d, -f2 <<<"$g" | tr -d ' ')" \
      "$(cut -d, -f3 <<<"$g" | tr -d ' ')" "$(cut -d, -f4 <<<"$g" | tr -d ' ')"
  fi

  # --- where in the loop ----------------------------------------------------
  local ITER cur nxt
  ITER=$(grep -E '^ITER=' "$ROOT/config.sh" | head -1 | sed -E 's/^ITER=([0-9]+).*/\1/')
  cur="M$ITER"; nxt="M$((ITER + 1))"
  printf '\n%s iteration : %s -> %s%s' "$B" "$cur" "$nxt" "$N"
  [[ -f "$ROOT/.loop_state" ]] && printf '   (%s)' "$(cat "$ROOT/.loop_state")"
  printf '\n'
  [[ -f "$ROOT/STOP" ]] && printf ' %sSTOP sentinel present -- will halt at the next iteration boundary%s\n' "$Y" "$N"

  # --- per-stage progress ---------------------------------------------------
  local mark
  printf '\n%s stages%s\n' "$B" "$N"

  # forward
  local fres="$WORKBASE/forward/result_$cur" nev ndone
  nev=$(find "$SRC_SET" -mindepth 1 -maxdepth 3 -type d -name '*_Z' 2>/dev/null | wc -l)
  ndone=$(find "$fres" -maxdepth 2 -name '.event_done' 2>/dev/null | wc -l)
  if [[ -f "$fres/.stage_done" ]]; then mark="${G}done${N}"
  elif [[ "$ndone" -gt 0 ]]; then mark="${Y}running${N}"
  else mark="${D}pending${N}"; fi
  printf '  %-22s %-18s %s/%s events\n' "forward($cur)" "$mark" "$ndone" "$nev"

  # cal_adjoint
  if [[ -f "$ROOT/.loop_marks/caladj.$cur" ]]; then mark="${G}done${N}"; else mark="${D}pending${N}"; fi
  local nadj
  nadj=$(find "$ADJSRC" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  printf '  %-22s %-18s %s adjoint-source dir(s)\n' "cal_adjoint($cur)" "$mark" "$nadj"

  # adjoint kernels
  local nk
  nk=$(find "$WORKBASE/adjoint/group_0/RUN" -maxdepth 2 -name '.kernel_done' 2>/dev/null | wc -l)
  if [[ -f "$WORKBASE/adjoint/.stage_done" ]]; then mark="${G}done${N}"
  elif [[ "$nk" -gt 0 ]]; then mark="${Y}running${N}"
  else mark="${D}pending${N}"; fi
  printf '  %-22s %-18s %s/%s events\n' "kernels($cur)" "$mark" "$nk" "$nadj"

  # gradient
  local grad="$WORKBASE/adjoint/m${ITER}to$((ITER + 1))"
  [[ -e "$grad/proc000000_alpha_kernel_clip_smooth.bin" ]] && mark="${G}done${N}" || mark="${D}pending${N}"
  printf '  %-22s %-18s %s\n' "gradient" "$mark" "$(basename "$grad")"

  # line search
  local lsa="$WORKBASE/linesearch/ls_alpha.txt"
  if [[ -s "$lsa" ]]; then mark="${G}done${N}  alpha* = $(cat "$lsa")"
  else
    local ndone_ls=0 s
    for s in $LS_STEPS; do [[ -f "$WORKBASE/linesearch/Result_$s/.stage_done" ]] && ndone_ls=$((ndone_ls+1)); done
    if [[ "$ndone_ls" -gt 0 ]]; then mark="${Y}running${N}  $ndone_ls/$(wc -w <<<"$LS_STEPS") trial forwards done"
    else mark="${D}pending${N}"; fi
  fi
  printf '  %-22s %b\n' "line search" "$mark"

  # commit
  [[ -e "$MODELBASE/$nxt/.done" ]] && mark="${G}done${N}" || mark="${D}pending${N}"
  printf '  %-22s %-18s %s\n' "commit($nxt)" "$mark" "$MODELBASE/$nxt"

  # --- misfit ---------------------------------------------------------------
  if [[ -s "$ROOT/misfit_history.txt" ]]; then
    printf '\n%s misfit history%s  (model  misfit  n_meas  when)\n' "$B" "$N"
    tail -8 "$ROOT/misfit_history.txt" | sed 's/^/  /'
    # direction of travel
    local a b
    a=$(tail -2 "$ROOT/misfit_history.txt" | head -1 | awk '{print $2}')
    b=$(tail -1 "$ROOT/misfit_history.txt" | awk '{print $2}')
    if [[ -n "$a" && -n "$b" && "$a" != NA && "$b" != NA ]]; then
      awk -v a="$a" -v b="$b" -v G="$G" -v R="$R" -v N="$N" \
        'BEGIN{ d=(b-a)/a*100;
                printf "  %s%+.2f%%%s vs the previous model\n", (b<a? G : R), d, N }'
    fi
  fi

  # --- disk -----------------------------------------------------------------
  printf '\n%s disk%s\n' "$B" "$N"
  local avail
  avail=$(df -h --output=avail "$WORKBASE" 2>/dev/null | tail -1 | tr -d ' ')
  printf '  %-22s %s free\n' "$WORKBASE" "${avail:-?}"
  local d
  for d in forward adjoint linesearch; do
    [[ -d "$WORKBASE/$d" ]] && printf '  %-22s %s\n' "  $d/" "$(du -sh "$WORKBASE/$d" 2>/dev/null | cut -f1)"
  done
  [[ -d "$MODELBASE" ]] && printf '  %-22s %s\n' "$MODELBASE" "$(du -sh "$MODELBASE" 2>/dev/null | cut -f1)"

  # --- recent log ------------------------------------------------------------
  if [[ -s "$ROOT/log/run_loop.log" ]]; then
    printf '\n%s last log lines%s (%s)\n' "$B" "$N" "$ROOT/log/run_loop.log"
    tail -6 "$ROOT/log/run_loop.log" | sed 's/^/  /'
  fi
  printf '\n'
}

if [[ "$WATCH" -eq 1 ]]; then
  while :; do clear; show; printf '%s(refreshing every %ss -- Ctrl-C to exit)%s\n' $'\033[2m' "$INTERVAL" $'\033[0m'; sleep "$INTERVAL"; done
else
  show
fi
