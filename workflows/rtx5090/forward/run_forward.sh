#!/bin/bash
# ite3_gpu/forward/run_forward.sh -- forward simulations for one model, one GPU.
#
# Replaces ite3's 1_auto_forward.bash + run_forward_anvil.slurm + the IB watchdog.
# All the machinery those need -- grouping into FWD_NGROUPS concurrent SLURM jobs,
# genmesh dependency chaining, squeue liveness probes, resubmit-on-InfiniBand-hang
# -- exists to coordinate 200 nodes. With one GPU it collapses to: mesh once,
# then loop over events.
#
# What is genuinely different (not just simpler):
#
#   SAVE_FORWARD IS OFF HERE. On Anvil this stage writes the reconstruction
#   wavefield to FRAME/ so the adjoint can skip its own forward -- 29 GB per
#   event, 2.8 TB for 99 events. That trade makes sense on Lustre and none at
#   all on a workstation. Here the forward writes seismograms only (~250 MB per
#   event) and the adjoint stage re-runs its own forward per event, keeping the
#   wavefield for the minutes it needs it. Set FRAME_KEEP=1 in config.sh to
#   restore the Anvil behaviour if you ever have the disk.
#
#   COMPLETION IS AN EXPLICIT SENTINEL, not `du -sm >= 300`. See config.sh.
#
# Usage:
#   ./run_forward.sh                          # forward $ORIMODEL over $SRC_SET
#   ./run_forward.sh --model /path/to/M1      # a specific model
#   ./run_forward.sh --events /path/to/set    # a specific event set
#   ./run_forward.sh --limit 3                # first 3 events only (smoke test)
#   ./run_forward.sh --force                  # redo events already marked done
#   ./run_forward.sh --list                   # show what would run, then exit
#
# Resumable: re-run it. Finished events are skipped by their .event_done sentinel.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=forward
source "$ROOT/lib/common.sh"
source "$ROOT/lib/solver.sh"

MODEL_DIR="$ORIMODEL"
EVENTS="$SRC_SET"
OUTDIR=""
LIMIT=0
FORCE=0
LISTONLY=0
SAVE_FWD="$FRAME_KEEP"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL_DIR="$2"; shift ;;
    --events) EVENTS="$2"; shift ;;
    --out)    OUTDIR="$2"; shift ;;
    --limit)  LIMIT="$2"; shift ;;
    --force)  FORCE=1 ;;
    --list)   LISTONLY=1 ;;
    --save-forward) SAVE_FWD=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done

TAG="$(basename "$MODEL_DIR")"
: "${OUTDIR:=$WORKBASE/forward/result_$TAG}"
WD="$WORKBASE/forward/solver"

banner "FORWARD  model=$TAG  events=$(basename "$EVENTS")"

[[ -f "$MODEL_DIR/.done" ]] || die "model $MODEL_DIR has no .done sentinel -- it is incomplete or was never built"

mapfile -t EVLIST < <(list_events "$EVENTS")
[[ "${#EVLIST[@]}" -gt 0 ]] || die "no events found under $EVENTS"
[[ "$LIMIT" -gt 0 ]] && EVLIST=("${EVLIST[@]:0:$LIMIT}")

# ---------------------------------------------------------------------------
# work out what actually needs running
# ---------------------------------------------------------------------------
TODO=(); DONE=0
for e in "${EVLIST[@]}"; do
  d="$OUTDIR/$(basename "$e")"
  if [[ "$FORCE" -eq 0 ]] && event_complete "$d"; then DONE=$((DONE + 1)); else TODO+=("$e"); fi
done

log "events: ${#EVLIST[@]} total, $DONE already complete, ${#TODO[@]} to run"
log "model : $MODEL_DIR"
log "output: $OUTDIR"
log "wavefield (SAVE_FORWARD): $( [[ "$SAVE_FWD" -eq 1 ]] && echo 'ON (~29 GB/event kept)' || echo 'off (seismograms only)')"

if [[ "$LISTONLY" -eq 1 ]]; then
  printf '%s\n' "${TODO[@]}"
  exit 0
fi
if [[ "${#TODO[@]}" -eq 0 ]]; then
  log "nothing to do -- all events complete"
  : > "$OUTDIR/.stage_done"
  exit 0
fi

take_lock
mkdir -p "$OUTDIR"

# Disk sanity: ~250 MB of ASCII seismograms per event, plus the wavefield if kept.
NEED=$(( ${#TODO[@]} * 250 / 1024 + 2 ))
[[ "$SAVE_FWD" -eq 1 ]] && NEED=$(( NEED + ${#TODO[@]} * 30 ))
require_disk "$WORKBASE" "$NEED" "forward output"

# ---------------------------------------------------------------------------
# mesh once for this model
# ---------------------------------------------------------------------------
prepare_workdir "$WD"
ensure_genmesh "$WD" "$MODEL_DIR"

# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
NFAIL=0; NOK=0; FAILED=()
T_START=$(date +%s)
i=0
for e in "${TODO[@]}"; do
  i=$((i + 1))
  name=$(basename "$e")
  dest="$OUTDIR/$name"

  # ETA from the running mean -- the whole point of a sequential run is that it
  # is predictable, so say so out loud.
  if [[ "$NOK" -gt 0 ]]; then
    elapsed=$(( $(date +%s) - T_START ))
    per=$(( elapsed / NOK ))
    remain=$(( per * (${#TODO[@]} - i + 1) ))
    eta=" | ~$(human_time $per)/event, ETA $(human_time $remain)"
  else
    eta=""
  fi
  log "[$i/${#TODO[@]}] $name$eta"

  ok=0
  for try in $(seq 0 "${MAX_EVENT_RETRIES:-1}"); do
    [[ "$try" -gt 0 ]] && warn "  retry $try for $name"
    if run_forward_event "$WD" "$e" "$dest" "$SAVE_FWD"; then ok=1; break; fi
  done

  if [[ "$ok" -eq 1 ]]; then
    NOK=$((NOK + 1))
    if [[ "$SAVE_FWD" -eq 1 ]]; then
      fd="$FRAME_DIR/$name"; mkdir -p "$fd"; rm -rf "${fd:?}"/*
      mv "$WD"/DATABASES_MPI/proc*_save_forward_arrays.bin "$fd/" 2>/dev/null || true
      mv "$WD"/DATABASES_MPI/proc*_absorb_field.bin        "$fd/" 2>/dev/null || true
      printf '%s\n' "$MODEL_DIR" > "$fd/.model"
    else
      drop_wavefield "$WD"
    fi
  else
    NFAIL=$((NFAIL + 1)); FAILED+=("$name")
    drop_wavefield "$WD"
    if [[ "${KEEP_GOING_ON_EVENT_FAIL:-1}" -eq 0 ]]; then
      die "event $name failed and KEEP_GOING_ON_EVENT_FAIL=0"
    fi
    warn "  $name FAILED -- continuing (KEEP_GOING_ON_EVENT_FAIL=1)"
  fi
done

T_END=$(date +%s)
hr
log "FORWARD DONE  ok=$NOK  failed=$NFAIL  wall=$(human_time $((T_END - T_START)))"
[[ "$NOK" -gt 0 ]] && log "mean per event: $(human_time $(( (T_END - T_START) / NOK )))"
if [[ "$NFAIL" -gt 0 ]]; then
  warn "failed events: ${FAILED[*]}"
  warn "re-run this script to retry them (completed events are skipped)"
fi

# The stage sentinel is what run_loop keys off. Only written when every event in
# the set is complete, so a partial run cannot be mistaken for a finished one.
ALLOK=1
for e in "${EVLIST[@]}"; do
  event_complete "$OUTDIR/$(basename "$e")" || { ALLOK=0; break; }
done
if [[ "$ALLOK" -eq 1 ]]; then
  : > "$OUTDIR/.stage_done"
  log "all ${#EVLIST[@]} events complete -> $OUTDIR/.stage_done"
  # Stable pointer, the way ite3's `result` symlink works (cal_adjoint's default
  # DATADIR reads it). Only repoint it for a REAL per-iteration result: the line
  # search also calls this script, with --out .../Result_<step>, and pointing
  # `result` at a throwaway trial model's synthetics would leave the symlink
  # lying about which model the current results belong to.
  if [[ "$(basename "$OUTDIR")" == result_* ]]; then
    ln -sfn "$OUTDIR" "$WORKBASE/forward/result"
    log "result symlink -> $OUTDIR"
  else
    log "(not a result_* dir -- leaving the 'result' symlink alone)"
  fi
  hr
  exit 0
else
  warn "stage NOT complete -- $OUTDIR/.stage_done not written"
  hr
  exit 1
fi
