#!/bin/bash
# ite3_gpu/adjoint/run_adjoint.sh -- per-event sensitivity kernels, one GPU.
#
# Replaces ite3's run_iteration.sh Phase A/B + run_forward_n_adjoint_anvil.slurm.
#
# THE STRUCTURAL CHANGE
# ---------------------
# On Anvil this stage does NOT run a forward: the earlier forward stage parked
# every event's reconstruction wavefield in FRAME/ (29 GB each), and this job
# copies 29 GB back into DATABASES_MPI before the SIMULATION_TYPE=3 run. That is
# 2.8 TB of storage and 5.6 TB of copying per iteration for 99 events.
#
# Here each event does forward-then-kernel back to back in the SAME workdir:
#
#     xspecfem3D  SIMULATION_TYPE=1  SAVE_FORWARD=.true.   -> wavefield in DATABASES_MPI
#     xspecfem3D  SIMULATION_TYPE=3                        -> reads it in place, writes kernels
#     rm the wavefield                                     -> back to ~0 GB
#
# The wavefield is never copied and never leaves the workdir. Cost: one extra
# forward per event. Benefit: peak disk goes from 2.8 TB to ~30 GB, and the
# 5.6 TB of I/O disappears -- on a single NVMe that I/O would have dominated the
# runtime far more than the recomputation does.
#
# Set FRAME_KEEP=1 in config.sh to reuse a pre-computed FRAME instead (skips the
# forward leg, requires the frame's .model to match $ORIMODEL).
#
# Usage:
#   ./run_adjoint.sh                     # kernels for every event in $ADJSRC
#   ./run_adjoint.sh --limit 2           # first 2 only (smoke test)
#   ./run_adjoint.sh --force             # redo events already marked done
#   ./run_adjoint.sh --list              # show the plan, then exit
#
# Resumable: re-run it. Finished events skip via their .kernel_done sentinel.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=adjoint
source "$ROOT/lib/common.sh"
source "$ROOT/lib/solver.sh"

MODEL_DIR="$ORIMODEL"
LIMIT=0; FORCE=0; LISTONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_DIR="$2"; shift ;;
    --limit) LIMIT="$2"; shift ;;
    --force) FORCE=1 ;;
    --list)  LISTONLY=1 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done

ADJ_WORK="$WORKBASE/adjoint"
# group_0/RUN/<event> is NOT a leftover of the SLURM grouping -- clip_iso.py
# locates each event's receiver list via the literal glob
#   group_*/RUN/<event>/OUTPUT_FILES/output_list_stations.txt
# so the layout is load-bearing. One group is enough for one GPU.
RUNDIR="$ADJ_WORK/group_0/RUN"
WD="$ADJ_WORK/solver"

banner "ADJOINT (kernels)  model=$(basename "$MODEL_DIR")"

[[ -f "$MODEL_DIR/.done" ]] || die "model $MODEL_DIR has no .done sentinel"
[[ -d "$ADJSRC" ]] || die "no adjoint sources at $ADJSRC -- run cal_adjoint first"

mapfile -t SRCS < <(find "$ADJSRC" -mindepth 1 -maxdepth 1 -type d | sort)
[[ "${#SRCS[@]}" -gt 0 ]] || die "no event dirs under $ADJSRC"
[[ "$LIMIT" -gt 0 ]] && SRCS=("${SRCS[@]:0:$LIMIT}")

# ---------------------------------------------------------------------------
# STALE-KERNEL GUARD -- do not remove.
# ---------------------------------------------------------------------------
# RUNDIR is model-independent, and kernel_event_complete() only tests
# .kernel_done + the four .bin files. Both survive across iterations. Without
# this stamp, iteration 1 would find every event already "done", skip the entire
# adjoint, and leave ORI_KERNELS pointing at iteration 0's kernels -- so
# run_post.sh (whose sentinels live in the NEW .state/ite1) would happily
# re-smooth M0's gradient and archive it as m1to2. The loop would see the
# gradient appear and carry on. Wrong science, no error, no way to notice.
#
# Same pattern linesearch.sh uses for its candidate models (.ls_model).
STAMPF="$ADJ_WORK/.kernel_model"
if [[ -f "$STAMPF" && "$(cat "$STAMPF")" != "$MODEL_DIR" ]]; then
  log "kernels on disk belong to $(basename "$(cat "$STAMPF")"), now computing for $(basename "$MODEL_DIR") -- wiping"
  rm -rf "$RUNDIR" "$ORI_KERNELS" "$KERNELS" "$ADJ_WORK/.stage_done"
fi
mkdir -p "$ADJ_WORK" "$RUNDIR"
printf '%s\n' "$MODEL_DIR" > "$STAMPF"

# ---------------------------------------------------------------------------
# stage events into the group layout
# ---------------------------------------------------------------------------
TODO=(); DONE=0
for s in "${SRCS[@]}"; do
  name=$(basename "$s")
  ev="$RUNDIR/$name"
  if [[ "$FORCE" -eq 0 ]] && kernel_event_complete "$ev"; then DONE=$((DONE + 1)); continue; fi
  mkdir -p "$ev"
  cp -f "$s/FORCESOLUTION"    "$ev/" 2>/dev/null || { warn "$name: no FORCESOLUTION -- skipping"; continue; }
  cp -f "$s/STATIONS_ADJOINT" "$ev/" 2>/dev/null || { warn "$name: no STATIONS_ADJOINT -- skipping"; continue; }
  # SEM is ~340 MB per event (3 components x 756 stations x 5000 samples of
  # ASCII). Symlink rather than copy -- 99 events would be 34 GB of duplication.
  [[ -d "$s/SEM" ]] || { warn "$name: no SEM/ -- skipping"; continue; }
  ln -sfn "$s/SEM" "$ev/SEM"
  TODO+=("$ev")
done

log "events: ${#SRCS[@]} total, $DONE already have kernels, ${#TODO[@]} to run"
log "model : $MODEL_DIR"
log "work  : $ADJ_WORK"
log "mode  : $( [[ "$FRAME_KEEP" -eq 1 ]] && echo 'FRAME reuse (skip forward leg)' || echo 'forward+kernel back-to-back, transient wavefield')"

if [[ "$LISTONLY" -eq 1 ]]; then printf '%s\n' "${TODO[@]}"; exit 0; fi
if [[ "${#TODO[@]}" -eq 0 ]]; then log "nothing to do"; : > "$ADJ_WORK/.stage_done"; exit 0; fi

take_lock

# Disk: the transient wavefield high-water mark, plus accumulated kernels.
KERN_MB=$(( NEX_XI * NEX_ETA * NEX_Z * 125 * 4 * 5 / 1048576 ))
require_disk "$WORKBASE" $(( 40 + ${#TODO[@]} * KERN_MB / 1024 )) "adjoint kernels"

prepare_workdir "$WD"
ensure_genmesh "$WD" "$MODEL_DIR"

# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
NOK=0; NFAIL=0; FAILED=()
T_START=$(date +%s)
i=0
for ev in "${TODO[@]}"; do
  i=$((i + 1))
  name=$(basename "$ev")

  if [[ "$NOK" -gt 0 ]]; then
    elapsed=$(( $(date +%s) - T_START )); per=$(( elapsed / NOK ))
    eta=" | ~$(human_time $per)/event, ETA $(human_time $(( per * (${#TODO[@]} - i + 1) )))"
  else eta=""; fi
  log "[$i/${#TODO[@]}] $name$eta"

  drop_wavefield "$WD"
  ok=0

  # --- leg 1: the forward wavefield -----------------------------------------
  if [[ "$FRAME_KEEP" -eq 1 && -d "$FRAME_DIR/$name" ]]; then
    fm=$(cat "$FRAME_DIR/$name/.model" 2>/dev/null || echo "")
    if [[ "$fm" == "$MODEL_DIR" ]]; then
      log "  restoring FRAME for $name"
      cp "$FRAME_DIR/$name"/proc*_save_forward_arrays.bin "$WD/DATABASES_MPI/" 2>/dev/null
      cp "$FRAME_DIR/$name"/proc*_absorb_field.bin        "$WD/DATABASES_MPI/" 2>/dev/null
      ok=1
    else
      warn "  FRAME for $name was built on '$fm', not '$MODEL_DIR' -- recomputing"
    fi
  fi
  if [[ "$ok" -eq 0 ]]; then
    tmp="$ADJ_WORK/.fwd_scratch/$name"
    mkdir -p "$tmp"
    if run_forward_event "$WD" "$ev" "$tmp" 1; then
      ok=1
      # The seismograms from this leg are a by-product; the misfit comes from
      # the dedicated forward stage. Drop them to save ~250 MB per event.
      rm -rf "$tmp"
    else
      warn "  $name: forward leg failed"
    fi
  fi

  # --- leg 2: the kernel run -------------------------------------------------
  if [[ "$ok" -eq 1 ]]; then
    if run_kernel_event "$WD" "$ev"; then NOK=$((NOK + 1)); else
      NFAIL=$((NFAIL + 1)); FAILED+=("$name"); ok=0
    fi
  else
    NFAIL=$((NFAIL + 1)); FAILED+=("$name")
  fi

  # Always drop the ~29 GB wavefield before the next event, success or not.
  drop_wavefield "$WD"

  if [[ "$ok" -eq 0 && "${KEEP_GOING_ON_EVENT_FAIL:-1}" -eq 0 ]]; then
    die "event $name failed and KEEP_GOING_ON_EVENT_FAIL=0"
  fi
done
rm -rf "$ADJ_WORK/.fwd_scratch"

T_END=$(date +%s)
hr
log "ADJOINT DONE  ok=$NOK  failed=$NFAIL  wall=$(human_time $((T_END - T_START)))"
[[ "$NOK" -gt 0 ]] && log "mean per event: $(human_time $(( (T_END - T_START) / NOK )))"
[[ "$NFAIL" -gt 0 ]] && warn "failed events: ${FAILED[*]}"

# ---------------------------------------------------------------------------
# link the completed kernels for the post-processing chain
# ---------------------------------------------------------------------------
# ORI_KERNELS/<event> -> the per-event KERNEL dir. clip_iso.py reads this.
mkdir -p "$ORI_KERNELS"
nlink=0
for ev in "$RUNDIR"/*/; do
  ev="${ev%/}"; name=$(basename "$ev")
  kernel_event_complete "$ev" || continue
  ln -sfn "$ev/KERNEL" "$ORI_KERNELS/$name"
  nlink=$((nlink + 1))
done
log "linked $nlink event kernel dir(s) -> $ORI_KERNELS"
[[ "$nlink" -gt 0 ]] || die "no complete kernels -- cannot continue to post-processing"

if [[ "$NFAIL" -eq 0 ]]; then
  : > "$ADJ_WORK/.stage_done"
  log "all events complete -> $ADJ_WORK/.stage_done"
  hr; exit 0
else
  warn "stage incomplete; re-run to retry the $NFAIL failed event(s)"
  hr; exit 1
fi
