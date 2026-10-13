#!/bin/bash
# workflows/grace_hopper/gh200_forward.sbatch -- Slurm wrapper for the timing run.
# Edit the [EDIT] lines for the site (partition, account, module names), then, from
# workflows/rtx5090 (mkdir -p log first: Slurm opens the --output file before the script runs):
#   sbatch ../grace_hopper/gh200_forward.sbatch                # 1 event
#   sbatch --export=ALL,N=40,EVSET=events/lsauto_iter9,TAG=ls40 ../grace_hopper/gh200_forward.sbatch
# If the system runs Run:ai / Kubernetes instead of Slurm, run the same body in
# a 1-GPU interactive job.
#SBATCH --job-name=ite3_gh200_fwd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72          # one Grace superchip; genmesh is serial, caladj wants the cores
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00             # 1 event: ~30 min incl. genmesh; 40 events: ~1.5 h
#SBATCH --output=log/slurm_%x_%j.out
##SBATCH --partition=gh200          # [EDIT]
##SBATCH --account=xxxx             # [EDIT]
set -uo pipefail
# submit from workflows/rtx5090:  cd workflows/rtx5090 && sbatch ../grace_hopper/gh200_forward.sbatch
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" && [[ -f config.sh && -x ../grace_hopper/time_forward.sh ]] || { echo "submit from workflows/rtx5090 (config.sh not found in $ROOT)" >&2; exit 1; }
# module load gcc openmpi cuda       # [EDIT] -- or leave to env.sh, which is sourced by every script
mkdir -p log
echo "node=$(hostname) arch=$(uname -m) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
./0_preflight.sh || true
../grace_hopper/time_forward.sh --n "${N:-1}" --events "${EVSET:-events/s2_keep150}" --tag "${TAG:-slurm$SLURM_JOB_ID}"
