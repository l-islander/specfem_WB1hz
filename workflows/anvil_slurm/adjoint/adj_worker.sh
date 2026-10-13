#!/bin/bash
# anvil_cpu/adjoint/adj_worker.sh <listfile> -- ONE array task = forward(save) + kernel for ONE event.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=adjoint; LOGFILE="$LOGDIR/adjoint_tasks.log"
source "$ROOT/lib/common.sh"; source "$ROOT/lib/solver.sh"; source "$ROOT/lib/slurm.sh"

ev=$(array_line "${1:?list file}")
[[ -n "$ev" ]] || die "task $SLURM_ARRAY_TASK_ID: empty line"
name=$(basename "$ev")
wd="$WORKBASE/adjoint/ev/$name"
log "[task $SLURM_ARRAY_TASK_ID job $SLURM_JOB_ID] $name  nodes=$SLURM_JOB_NUM_NODES"
if kernel_event_complete "$ev"; then log "  $name already has kernels -- skipping"; exit 0; fi

prepare_event_workdir "$wd" "${ADJ_DB:?}"
ok=0
if [[ "$FRAME_KEEP" -eq 1 && -d "$FRAME_DIR/$name" && "$(cat "$FRAME_DIR/$name/.model" 2>/dev/null)" == "${ADJ_MODEL:-}" ]]; then
  log "  restoring FRAME for $name"
  find "$FRAME_DIR/$name" -maxdepth 1 -type f -name 'proc*.bin' -print0 | xargs -0 cp -t "$wd/DATABASES_MPI/" && ok=1
fi
if [[ "$ok" -eq 0 ]]; then
  tmp="$WORKBASE/adjoint/.fwd_scratch/$name"; mkdir -p "$tmp"
  if run_forward_event "$wd" "$ev" "$tmp" 1; then ok=1; rm -rf "$tmp"; else warn "  $name: forward leg failed"; fi
fi
rc=1
if [[ "$ok" -eq 1 ]]; then
  if run_kernel_event "$wd" "$ev"; then rc=0; else warn "  $name: kernel leg failed"; fi
fi
drop_wavefield "$wd"
if [[ $rc -eq 0 ]]; then rm -rf "$wd"; else warn "  workdir kept: $wd"; fi
exit $rc
