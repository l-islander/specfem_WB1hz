#!/bin/bash
# multi_gpu/worker_one_source.sh -- the unit of work every scheduler dispatches: ONE virtual
# source on ONE GPU. Called by event_parallel/run_sources_on_gpus.sh (one shell worker per
# GPU), slurm/forward_array_one_gpu_per_source.sbatch (one array task per source) and
# kubernetes/indexed_job_one_gpu_per_source.yaml (one pod per source).
#
# Usage: worker_one_source.sh <source_dir | list_file:index> <db_wd> <model_dir> <out_root>
#   list_file:index selects line <index> (1-based) of a list of source directories, which is how
#   array/indexed jobs map their task id to a source.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF="$(cd "$HERE/../workflows/rtx5090" && pwd)"
source "$WF/config.sh"; [[ -r "$WF/env.sh" ]] && source "$WF/env.sh"
STAGE="mgpu_worker_${MGPU_DEVICE:-${SLURM_ARRAY_TASK_ID:-${JOB_COMPLETION_INDEX:-0}}}"
source "$WF/lib/common.sh"; source "$WF/lib/solver.sh"; source "$HERE/lib_multi_gpu.sh"

# Which GPU: the shell pool passes MGPU_DEVICE explicitly; under Slurm (--gres=gpu:1) or
# Kubernetes (nvidia.com/gpu: 1) keep the device the scheduler already exposed. config.sh sets
# GPU_DEVICE=0 for the single-card workstation, and lib/common.sh:gpu_run exports
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE, so it must be set here, after config.sh is sourced.
if   [[ -n "${MGPU_DEVICE:-}" ]];          then GPU_DEVICE="$MGPU_DEVICE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then GPU_DEVICE="$CUDA_VISIBLE_DEVICES"
fi

sel="${1:?source dir or list:index}"; db="${2:?db workdir}"; model="${3:?model dir}"; out="${4:?out root}"
if [[ "$sel" == *:* && -f "${sel%:*}" ]]; then
  src=$(sed -n "${sel##*:}p" "${sel%:*}")
  [[ -n "$src" ]] || die "no line ${sel##*:} in ${sel%:*}"
else
  src="$sel"
fi
log "worker: $(basename "$src") on GPU ${GPU_DEVICE:-0} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}) host $(hostname -s)"
run_one_source "$src" "$db" "$model" "$out"
