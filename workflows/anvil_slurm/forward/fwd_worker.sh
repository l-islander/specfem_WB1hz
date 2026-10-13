#!/bin/bash
# anvil_cpu/forward/fwd_worker.sh <listfile> -- ONE array task = ONE event at NPROC ranks.
# Reads its event from line $SLURM_ARRAY_TASK_ID; FWD_OUTDIR / FWD_DB / FWD_SAVE
# arrive in the environment from run_forward.sh (sbatch --export=ALL).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=forward; LOGFILE="$LOGDIR/forward_tasks.log"
source "$ROOT/lib/common.sh"; source "$ROOT/lib/solver.sh"; source "$ROOT/lib/slurm.sh"

src=$(array_line "${1:?list file}")
[[ -n "$src" ]] || die "task $SLURM_ARRAY_TASK_ID: empty line in $1"
name=$(basename "$src")
dest="${FWD_OUTDIR:?}/$name"
wd="$WORKBASE/forward/ev/$name"
log "[task $SLURM_ARRAY_TASK_ID job $SLURM_JOB_ID] $name  nodes=$SLURM_JOB_NUM_NODES ntasks=${SLURM_NTASKS:-?}"

if event_complete "$dest"; then log "  $name already complete -- skipping"; exit 0; fi
prepare_event_workdir "$wd" "${FWD_DB:?}"
ok=0
for try in $(seq 0 "${MAX_EVENT_RETRIES:-1}"); do
  [[ "$try" -gt 0 ]] && warn "  retry $try for $name"
  if run_forward_event "$wd" "$src" "$dest" "${FWD_SAVE:-0}"; then ok=1; break; fi
done
if [[ "$ok" -eq 1 && "${FWD_SAVE:-0}" -eq 1 ]]; then
  fd="$FRAME_DIR/$name"; mkdir -p "$fd"; rm -rf "${fd:?}"/*
  find "$wd/DATABASES_MPI" -maxdepth 1 -type f \( -name 'proc*_save_forward_arrays.bin' -o -name 'proc*_absorb_field.bin' \) -print0 \
    | xargs -0 mv -t "$fd/"
  printf '%s\n' "${FWD_MODEL:-}" > "$fd/.model"
fi
drop_wavefield "$wd"
if [[ "$ok" -eq 1 ]]; then rm -rf "$wd"; exit 0; fi
warn "  $name FAILED -- workdir kept for inspection: $wd"
exit 1
