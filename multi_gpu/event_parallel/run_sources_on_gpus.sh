#!/bin/bash
# multi_gpu/event_parallel/run_sources_on_gpus.sh -- run a set of virtual sources across all
# GPUs of one node (e.g. the 4 GPUs of one Vera Rubin compute tray, or several GH200
# superchips), one simulation per GPU at a time.
#
# Why one simulation per GPU: measured on a GH200, packing 2/4/8 forward simulations onto one
# GPU raised throughput only 1.09/1.16/1.20x (the time loop already saturates the GPU), so a
# campaign scales with the NUMBER of GPUs. A forward at 1.5 s (and every forward at 1 Hz) fits
# one Rubin GPU; 150 sources x 1 GPU each is the unit the NVL72 backfills.
#
# Design: the mesh database for the model is built once and shared read-only; each GPU runs a
# worker loop that claims the next unfinished source with an atomic mkdir, runs it in its own
# work directory, and moves on. Completion markers (.event_done) make the whole thing
# restartable: re-run the same command and only unfinished sources are dispatched.
#
# Usage:
#   run_sources_on_gpus.sh --gpus 4 --events <dir of <src>_Z/> --model <GLL model dir> --out <dir>
#   run_sources_on_gpus.sh --gpus 4 --events <dir> --model <dir> --out <dir> --dry-run
#       --dry-run replaces the solver with a 1-3 s stub that writes fake seismograms, to test the
#       dispatch logic on a machine without GPUs (no SPECFEM needed).
#
# Status: the per-source solver call is the production one (15 validated updates on the RTX 5090;
# forward/kernel timing and validation on the GH200). This pool has been exercised in --dry-run
# mode; it has not yet run on a multi-GPU node (the hardware available so far has one GPU per
# node).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MG="$(dirname "$HERE")"; WF="$(cd "$MG/../workflows/rtx5090" && pwd)"
NGPU=""; EVENTS=""; MODEL=""; OUT=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) NGPU="$2"; shift ;; --events) EVENTS="$2"; shift ;; --model) MODEL="$2"; shift ;;
    --out) OUT="$2"; shift ;; --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;; *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n "$EVENTS" && -n "$OUT" ]] || { echo "need --events and --out" >&2; exit 2; }
IFS=, read -r -a DEVS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$NGPU" ]]; then NGPU=${#DEVS[@]}; [[ "$NGPU" -gt 0 ]] || NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [[ "$NGPU" -gt 0 ]] || NGPU=1; fi
mkdir -p "$OUT"; exec 8>"$OUT/.pool.lock"; flock -n 8 || { echo "another pool is running on $OUT" >&2; exit 1; }
rm -rf "$OUT/.claims"; mkdir -p "$OUT/.claims"     # claims left by an interrupted run are stale
LIST="$OUT/.sources.list"
find -L "$EVENTS" -mindepth 1 -maxdepth 1 -type d -name '*_Z' | sort > "$LIST"
NSRC=$(wc -l < "$LIST")
echo "[pool] $NSRC sources, $NGPU GPUs, out=$OUT$( [[ $DRY -eq 1 ]] && echo '  (DRY RUN)')"

if [[ $DRY -eq 1 ]]; then
  # stub: same contract as the real worker (writes >= EVENT_MIN_SEMD traces + .event_done)
  worker() { local src="$1" dest="$OUT/$(basename "$1")"; mkdir -p "$dest"
             sleep "$(( (RANDOM % 3) + 1 ))"
             for i in $(seq 1 700); do : > "$dest/XX.S$i.BXZ.semd"; done; : > "$dest/.event_done"; }
  done_already() { [[ -f "$OUT/$(basename "$1")/.event_done" ]]; }
else
  source "$WF/config.sh"; [[ -r "$WF/env.sh" ]] && source "$WF/env.sh"
  STAGE=mgpu_pool; source "$WF/lib/common.sh"; source "$WF/lib/solver.sh"; source "$MG/lib_multi_gpu.sh"
  [[ -n "$MODEL" ]] || die "need --model"
  DB="$WORKBASE/mgpu_db/$(basename "$MODEL")"
  build_model_database "$DB" "$MODEL"          # once per model, CPU, shared by all GPUs
  worker() { MGPU_DEVICE="${DEVS[$GPU]:-$GPU}" "$MG/worker_one_source.sh" "$1" "$DB" "$MODEL" "$OUT"; }
  done_already() { event_complete "$OUT/$(basename "$1")"; }
fi

gpu_loop() {   # one loop per GPU: claim -> run -> next
  local GPU="$1" src name n=0
  while read -r src; do
    name=$(basename "$src")
    done_already "$src" && continue
    mkdir "$OUT/.claims/$name" 2>/dev/null || continue          # atomic claim
    echo "[gpu $GPU] $name start $(date +%T)"
    if worker "$src"; then n=$((n+1)); echo "[gpu $GPU] $name done  $(date +%T)"
    else echo "[gpu $GPU] $name FAILED (claim released; re-run to retry)"; rmdir "$OUT/.claims/$name"; fi
  done < "$LIST"
  echo "[gpu $GPU] finished, $n sources"
}

t0=$(date +%s)
for g in $(seq 0 $((NGPU-1))); do gpu_loop "$g" & done
wait
rm -rf "$OUT/.claims"
ndone=0; while read -r s; do [[ -f "$OUT/$(basename "$s")/.event_done" ]] && ndone=$((ndone+1)); done < "$LIST"
echo "[pool] $ndone / $NSRC sources complete in $(( $(date +%s) - t0 )) s"
[[ $ndone -eq $NSRC ]]
