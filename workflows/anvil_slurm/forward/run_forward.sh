#!/bin/bash
# anvil_cpu/forward/run_forward.sh -- forward simulations for one model as a SLURM job array.
#
# Driver (runs on a login node or inside a tmux): genmesh the model once, list
# the events that still need running, submit ONE job array with one event per
# task (each task = NPROC ranks over NODES_PER_EVENT nodes, ARRAY_THROTTLE in
# flight), wait, then verify every event's sentinel exactly as the local port
# did. The worker that runs inside each task is forward/fwd_worker.sh.
#
# Usage (same flags as ite3_gpu):
#   ./run_forward.sh --model DIR --events DIR --out DIR [--limit N] [--force] [--list] [--save-forward]
# Resumable: re-run it; finished events are skipped by their .event_done sentinel.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=forward
source "$ROOT/lib/common.sh"; source "$ROOT/lib/solver.sh"; source "$ROOT/lib/slurm.sh"

MODEL_DIR="$ORIMODEL"; EVENTS="$SRC_SET"; OUTDIR=""; LIMIT=0; FORCE=0; LISTONLY=0; SAVE_FWD="$FRAME_KEEP"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL_DIR="$2"; shift ;;
    --events) EVENTS="$2"; shift ;;
    --out)    OUTDIR="$2"; shift ;;
    --limit)  LIMIT="$2"; shift ;;
    --force)  FORCE=1 ;;
    --list)   LISTONLY=1 ;;
    --save-forward) SAVE_FWD=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done
TAG="$(basename "$MODEL_DIR")"
: "${OUTDIR:=$WORKBASE/forward/result_$TAG}"

banner "FORWARD  model=$TAG  events=$(basename "$EVENTS")  NPROC=$NPROC x $ARRAY_THROTTLE in flight"
[[ -f "$MODEL_DIR/.done" ]] || die "model $MODEL_DIR has no .done sentinel"
mapfile -t EVLIST < <(list_events "$EVENTS")
[[ "${#EVLIST[@]}" -gt 0 ]] || die "no events found under $EVENTS"
[[ "$LIMIT" -gt 0 ]] && EVLIST=("${EVLIST[@]:0:$LIMIT}")

TODO=(); DONE=0
for e in "${EVLIST[@]}"; do
  d="$OUTDIR/$(basename "$e")"
  if [[ "$FORCE" -eq 0 ]] && event_complete "$d"; then DONE=$((DONE + 1)); else TODO+=("$e"); fi
done
log "events: ${#EVLIST[@]} total, $DONE already complete, ${#TODO[@]} to run"
log "model : $MODEL_DIR"; log "output: $OUTDIR"
[[ "$LISTONLY" -eq 1 ]] && { printf '%s\n' "${TODO[@]}"; exit 0; }
if [[ "${#TODO[@]}" -eq 0 ]]; then log "nothing to do"; : > "$OUTDIR/.stage_done"; exit 0; fi

mkdir -p "$OUTDIR" "$WORKBASE/forward/ev"
# ~60 MB/event of binary seismograms (350 MB ascii); the transient wavefield if kept
PER_EV_MB=400; [[ "${SEISMO_FORMAT:-ascii}" == bin ]] && PER_EV_MB=80
NEED=$(( ${#TODO[@]} * PER_EV_MB / 1024 + 2 ))
[[ "$SAVE_FWD" -eq 1 ]] && NEED=$(( NEED + ${#TODO[@]} * ${WAVEFIELD_GB_EST:-200} ))
require_disk "$WORKBASE" "$NEED" "forward output"

ensure_genmesh_model "$MODEL_DIR"

LIST="$WORKBASE/forward/todo_${TAG}_$$.list"
printf '%s\n' "${TODO[@]}" > "$LIST"
export FWD_OUTDIR="$OUTDIR" FWD_DB="$(db_dir "$MODEL_DIR")" FWD_SAVE="$SAVE_FWD" FWD_MODEL="$MODEL_DIR"
T_START=$(date +%s)
sbatch_array "fwd_$TAG" "$LIST" "$NODES_PER_EVENT" "$NPROC" "$EVENT_WALL" "${PART_COMPUTE:?}" "$ARRAY_THROTTLE" -- \
  bash "$HERE/fwd_worker.sh"
rm -f "$LIST"
T_END=$(date +%s)

NOK=0; NFAIL=0; FAILED=()
for e in "${TODO[@]}"; do
  if event_complete "$OUTDIR/$(basename "$e")"; then NOK=$((NOK + 1)); else NFAIL=$((NFAIL + 1)); FAILED+=("$(basename "$e")"); fi
done
hr; log "FORWARD DONE  ok=$NOK  failed=$NFAIL  wall=$(human_time $((T_END - T_START)))  (${#TODO[@]} events, $ARRAY_THROTTLE in flight)"
[[ "$NFAIL" -gt 0 ]] && { warn "failed events: ${FAILED[*]}"; warn "re-run this script to retry them (completed events are skipped)"; }

ALLOK=1
for e in "${EVLIST[@]}"; do event_complete "$OUTDIR/$(basename "$e")" || { ALLOK=0; break; }; done
if [[ "$ALLOK" -eq 1 ]]; then
  : > "$OUTDIR/.stage_done"; log "all ${#EVLIST[@]} events complete -> $OUTDIR/.stage_done"
  if [[ "$(basename "$OUTDIR")" == result_* ]]; then ln -sfn "$OUTDIR" "$WORKBASE/forward/result"; log "result symlink -> $OUTDIR"; fi
  hr; exit 0
else
  warn "stage NOT complete -- $OUTDIR/.stage_done not written"; hr; exit 1
fi
