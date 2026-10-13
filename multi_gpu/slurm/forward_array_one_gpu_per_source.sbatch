#!/bin/bash
# multi_gpu/slurm/forward_array_one_gpu_per_source.sbatch -- the forward stage as a Slurm job
# array: one array task = one virtual source = one GPU. With %72 throttling, one NVL72 rack runs
# 72 sources at a time; the scheduler backfills the rest as GPUs free up. Re-submitting the same
# array re-runs only sources without a completion marker (the worker skips finished ones).
#
#   cd multi_gpu/slurm                                    # submit from here (the scripts call ../)
#   T=/abs/path/to/workflows/rtx5090                      # tree holding events/ and models/
#   ls -d "$T"/events/s2_keep150/*_Z > sources.list      # 150 ABSOLUTE paths
#   jid=$(sbatch --parsable --export=ALL,MODEL=$T/models/M6 build_database.sbatch)
#   sbatch --dependency=afterok:$jid --array=1-150%72 \
#          --export=ALL,MODEL=$T/models/M6,LIST=$PWD/sources.list,OUT=$T/work/forward/result_M6 \
#          forward_array_one_gpu_per_source.sbatch
#
# Same pattern as workflows/anvil_slurm (one array task per source, NPROC CPU ranks per task),
# with one GPU per task instead of 4-16 CPU nodes per task.
#SBATCH --job-name=wb_fwd
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=wb_fwd_%A_%a.out
##SBATCH --partition=<gpu-partition>   ##SBATCH --account=<allocation>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"                       # submit from multi_gpu/slurm
WF="$(cd ../../workflows/rtx5090 && pwd)"
WB="$(source "$WF/config.sh"; [[ -r "$WF/env.sh" ]] && source "$WF/env.sh"; echo "$WORKBASE")"   # same WORKBASE as build_database.sbatch
../worker_one_source.sh "${LIST:?}:${SLURM_ARRAY_TASK_ID:?}" "$WB/mgpu_db/$(basename "${MODEL:?}")" "$MODEL" "${OUT:?}"
