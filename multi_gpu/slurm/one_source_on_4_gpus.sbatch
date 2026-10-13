#!/bin/bash
# multi_gpu/slurm/one_source_on_4_gpus.sbatch -- ONE simulation split across 4 GPUs.
#
# This is SPECFEM3D's standard domain decomposition: the mesh is cut into NPROC = NPROC_XI x
# NPROC_ETA slices, one MPI rank per slice, one GPU per rank, and neighbouring slices exchange
# boundary values every time step. It is how the project ran on CPUs on Purdue Anvil (1,250 ranks
# per simulation) and how the 1 Hz adjoint (~100 GB state + ~0.6 TB wavefield), the margin-scale
# scenario runs (~1.1 TB, 6-8 GPUs) and a 2 Hz stage (>= 4 GPUs) will run on the NVL72.
#
# GPU binding: SPECFEM selects device (rank % visible_devices) (src/gpu/initialize_gpu.cu), so
# either one GPU per task (--gpus-per-task=1, each rank sees device 0) or all node GPUs visible
# (rank r -> device r) works without code changes.
#
# Halo exchange today: SPECFEM 4.1.1 packs the boundary buffers on the GPU and stages them through
# host memory for MPI (src/gpu/assemble_MPI_vector_cuda.cu, gpuMemcpy_tohost_realw). On a Vera Rubin
# superchip that hop is the 1.8 TB/s NVLink-C2C link; the planned device-resident path
# (CUDA-aware MPI or NCCL) keeps it on NVLink between GPUs.
#
# Status: standard SPECFEM capability, not yet executed in this project on multiple GPUs (our
# GPU nodes so far have one GPU each). Steps below are SPECFEM's standard mesher, database and solver sequence, with GPU_MODE on.
#SBATCH --job-name=wb_np4
#SBATCH --nodes=1 --ntasks=4 --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=wb_np4_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
RUN="${RUN:?a SPECFEM run directory with bin/, DATA/, DATABASES_MPI/, OUTPUT_FILES/}"
cd "$RUN"
# 1. decomposition: 2 x 2 slices of the 104 x 208 x 25 mesh (NEX divisible by NPROC per direction)
sed -i -E 's/^(NPROC_XI\s*=).*/\1 2/; s/^(NPROC_ETA\s*=).*/\1 2/' DATA/meshfem3D_files/Mesh_Par_file
sed -i -E 's/^(NPROC\s*=).*/\1 4/; s/^(GPU_MODE\s*=).*/\1 .true./; s/^(MODEL\s*=).*/\1 tomo/' DATA/Par_file
[[ -s DATA/tomo_files/tomography_model.xyz ]] || { echo "export the model first (workflows/rtx5090/tools/gll_to_tomo.py)" >&2; exit 1; }
# 2. mesher and database generation, one rank per slice. A GLL model is welded to its partition,
#    so the model is re-baked from a tomography-file export (workflows/rtx5090/tools/gll_to_tomo.py,
#    then MODEL=tomo here) -- a one-time data step per partition count.
# srun may need --mpi=pmix (site-specific)
srun --ntasks=4 ./bin/xmeshfem3D
srun --ntasks=4 ./bin/xgenerate_databases
# 3. the solver: 4 ranks, 4 GPUs
srun --ntasks=4 --gpus-per-task=1 ./bin/xspecfem3D
