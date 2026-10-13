#!/bin/bash
# multi_gpu/slurm/build_database.sbatch -- build the shared mesh database for one model, once.
# xgenerate_databases at NPROC=1 is serial CPU work; every GPU task of the forward array then
# symlinks the baked mesh read-only (multi_gpu/lib_multi_gpu.sh).
#   sbatch --export=ALL,MODEL=/path/to/models/M6 build_database.sbatch
#SBATCH --job-name=wb_db
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=wb_db_%j.out
##SBATCH --partition=<cpu-or-gpu-partition>   ##SBATCH --account=<allocation>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"                       # submit from multi_gpu/slurm
MG="$(cd .. && pwd)"; WF="$(cd ../../workflows/rtx5090 && pwd)"
source "$WF/config.sh"; [[ -r "$WF/env.sh" ]] && source "$WF/env.sh"
STAGE=mgpu_db; source "$WF/lib/common.sh"; source "$WF/lib/solver.sh"; source "$MG/lib_multi_gpu.sh"
build_model_database "$WORKBASE/mgpu_db/$(basename "${MODEL:?}")" "$MODEL"
