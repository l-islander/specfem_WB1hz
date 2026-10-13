#!/bin/bash
# anvil_cpu/adjoint/run_adjoint.sh -- per-event kernels as a SLURM job array.
#
# Each array task (adjoint/adj_worker.sh) does, in the event's own workdir:
#     xspecfem3D  SIMULATION_TYPE=1  SAVE_FORWARD=.true.   -> wavefield in its DATABASES_MPI
#     xspecfem3D  SIMULATION_TYPE=3                        -> kernels
#     rm the wavefield
# ARRAY_THROTTLE such tasks run at once, so scratch transiently holds
# ARRAY_THROTTLE x (wavefield per event) -- see tools/plan_mesh.py for the
# number at your mesh, and keep it inside the scratch quota.
#
# Usage:  ./run_adjoint.sh [--model DIR] [--limit N] [--force] [--list]
# Resumable via .kernel_done; the STALE-KERNEL GUARD from ite3_gpu is kept.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=adjoint
source "$ROOT/lib/common.sh"; source "$ROOT/lib/solver.sh"; source "$ROOT/lib/slurm.sh"

MODEL_DIR="$ORIMODEL"; LIMIT=0; FORCE=0; LISTONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_DIR="$2"; shift ;;
    --limit) LIMIT="$2"; shift ;;
    --force) FORCE=1 ;;
    --list)  LISTONLY=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done

ADJ_WORK="$WORKBASE/adjoint"
RUNDIR="$ADJ_WORK/group_0/RUN"        # load-bearing: clip_iso.py globs group_*/RUN/<event>/OUTPUT_FILES/
banner "ADJOINT (kernels)  model=$(basename "$MODEL_DIR")  NPROC=$NPROC x $ARRAY_THROTTLE in flight"
[[ -f "$MODEL_DIR/.done" ]] || die "model $MODEL_DIR has no .done sentinel"
[[ -d "$ADJSRC" ]] || die "no adjoint sources at $ADJSRC -- run cal_adjoint first"
mapfile -t SRCS < <(find "$ADJSRC" -mindepth 1 -maxdepth 1 -type d | sort)
[[ "${#SRCS[@]}" -gt 0 ]] || die "no event dirs under $ADJSRC"
[[ "$LIMIT" -gt 0 ]] && SRCS=("${SRCS[@]:0:$LIMIT}")

# STALE-KERNEL GUARD (ite3_gpu): kernels on disk belong to one model only.
STAMPF="$ADJ_WORK/.kernel_model"
if [[ -f "$STAMPF" && "$(cat "$STAMPF")" != "$MODEL_DIR" ]]; then
  log "kernels on disk belong to $(basename "$(cat "$STAMPF")"), now computing for $(basename "$MODEL_DIR") -- wiping"
  rm -rf "$RUNDIR" "$ORI_KERNELS" "$KERNELS" "$ADJ_WORK/.stage_done" "$ADJ_WORK/ev"
fi
mkdir -p "$ADJ_WORK" "$RUNDIR" "$ADJ_WORK/ev"
printf '%s\n' "$MODEL_DIR" > "$STAMPF"

TODO=(); DONE=0
for s in "${SRCS[@]}"; do
  name=$(basename "$s"); ev="$RUNDIR/$name"
  if [[ "$FORCE" -eq 0 ]] && kernel_event_complete "$ev"; then DONE=$((DONE + 1)); continue; fi
  mkdir -p "$ev"
  cp -f "$s/FORCESOLUTION"    "$ev/" 2>/dev/null || { warn "$name: no FORCESOLUTION -- skipping"; continue; }
  cp -f "$s/STATIONS_ADJOINT" "$ev/" 2>/dev/null || { warn "$name: no STATIONS_ADJOINT -- skipping"; continue; }
  [[ -d "$s/SEM" ]] || { warn "$name: no SEM/ -- skipping"; continue; }
  ln -sfn "$s/SEM" "$ev/SEM"
  TODO+=("$ev")
done
log "events: ${#SRCS[@]} total, $DONE already have kernels, ${#TODO[@]} to run"
log "model : $MODEL_DIR"; log "work  : $ADJ_WORK"
[[ "$LISTONLY" -eq 1 ]] && { printf '%s\n' "${TODO[@]}"; exit 0; }
if [[ "${#TODO[@]}" -eq 0 ]]; then log "nothing to do"; : > "$ADJ_WORK/.stage_done"; exit 0; fi

# scratch: throttle x transient wavefield + all kernels (4 x NSPEC x 125 x 4 B per event)
KERN_GB=$(( NEX_XI * NEX_ETA * NEX_Z * 125 * 4 * 4 / 1073741824 + 1 ))
require_disk "$WORKBASE" $(( ARRAY_THROTTLE * ${WAVEFIELD_GB_EST:-200} + ${#TODO[@]} * KERN_GB )) "adjoint kernels + transient wavefields"

ensure_genmesh_model "$MODEL_DIR"

LIST="$ADJ_WORK/todo_$$.list"
printf '%s\n' "${TODO[@]}" > "$LIST"
export ADJ_DB="$(db_dir "$MODEL_DIR")" ADJ_MODEL="$MODEL_DIR"
T_START=$(date +%s)
sbatch_array "adj_$(basename "$MODEL_DIR")" "$LIST" "$NODES_PER_EVENT" "$NPROC" "$ADJ_WALL" "${PART_COMPUTE:?}" "$ARRAY_THROTTLE" -- \
  bash "$HERE/adj_worker.sh"
rm -f "$LIST"
T_END=$(date +%s)

NOK=0; NFAIL=0; FAILED=()
for ev in "${TODO[@]}"; do
  if kernel_event_complete "$ev"; then NOK=$((NOK + 1)); else NFAIL=$((NFAIL + 1)); FAILED+=("$(basename "$ev")"); fi
done
hr; log "ADJOINT DONE  ok=$NOK  failed=$NFAIL  wall=$(human_time $((T_END - T_START)))"
[[ "$NFAIL" -gt 0 ]] && warn "failed events: ${FAILED[*]}"

mkdir -p "$ORI_KERNELS"; nlink=0
for ev in "$RUNDIR"/*/; do
  ev="${ev%/}"; kernel_event_complete "$ev" || continue
  ln -sfn "$ev/KERNEL" "$ORI_KERNELS/$(basename "$ev")"; nlink=$((nlink + 1))
done
log "linked $nlink event kernel dir(s) -> $ORI_KERNELS"
[[ "$nlink" -gt 0 ]] || die "no complete kernels -- cannot continue to post-processing"
if [[ "$NFAIL" -eq 0 ]]; then : > "$ADJ_WORK/.stage_done"; log "all events complete -> $ADJ_WORK/.stage_done"; hr; exit 0
else warn "stage incomplete; re-run to retry the $NFAIL failed event(s)"; hr; exit 1; fi
