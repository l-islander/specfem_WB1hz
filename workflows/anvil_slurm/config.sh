# anvil_cpu/config.sh -- single source of truth for the ANVIL CPU port of the
# ite3_gpu FWI workflow (the 1.5 s / 1 s mesh stage).
#
# This is the local single-GPU workflow moved BACK onto Purdue Anvil, but not
# back to the old ite3 layout. What is kept from ite3_gpu (all of it proven in
# three campaigns): the science files, the sentinel-resumable stages, the
# explicit completion tests, the transient wavefield, the REWINDOW loop and
# its gates, the kernel-pair manifest line search. What changes is only the
# execution model:
#
#            local ite3_gpu (RTX 5090)         |   anvil_cpu (SLURM, CPU nodes)
#   ---------------------------------------- + ----------------------------------------
#    NPROC = 1 (one GPU, one partition)      |   NPROC = NPROC_XI x NPROC_ETA ranks per event
#    events one at a time                    |   a SLURM JOB ARRAY, one event per task,
#                                            |   ARRAY_THROTTLE events in flight
#    plain bash, flock on the GPU            |   drivers submit with `sbatch --wait`;
#                                            |   SLURM is the lock
#    CUDA solver (sm_120)                    |   CPU solver (gcc/AOCC, -march=znver3)
#    xsmooth_sem on the GPU                  |   xsmooth_sem CPU kd-tree at NPROC ranks
#    2 s mesh 104x208x25, DT 0.0145          |   1.5 s or 1 s mesh from tools/plan_mesh.py
#
# Machine paths, modules and SLURM account/partitions live in env.sh (copy
# env.sh.anvil). The mesh geometry + DT/NSTEP live in mesh_plan.sh, WRITTEN by
# tools/plan_mesh.py --apply; this file sources it so one planner decision
# rules Mesh_Par_file, interfaces.dat, Par_file and every runner.
#
#   ./0_preflight.sh          modules, sbatch, scratch, mpi4py world size
#   ./1_build_specfem_cpu.sh  CPU build (or reuse the project tree's binaries)
#   tools/plan_mesh.py 1.5 --apply    choose + write the mesh (see README)
#   ./2_make_mesh.sh          xmeshfem3D at NPROC (SLURM job)
#   ./3_make_M0.sh            bake the tomo export of the 2 s final model -> M0
#   ./4_bench.sh              ONE event forward(+save): measures core rate, SU cost
#   ./smoke_test.sh           2 events through the whole chain
#   ./run_loop.sh             the autonomous loop (tmux on a login node)
# ---------------------------------------------------------------------------

## --- where everything lives ----------------------------------------------
GPU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # name kept: every lib/ helper reads GPU_ROOT
ANVIL_ROOT="$GPU_ROOT"

# WORKBASE: BIG + regenerable. On Anvil this MUST be on scratch
# (/anvil/scratch/$USER/...), never $HOME (25 GB quota) -- env.sh sets it.
WORKBASE="${WORKBASE:-$GPU_ROOT/work}"
# MODELBASE: the models. At NPROC=1024 a model is 3 x NPROC files (~6-8 GB at
# 1.5 s, ~19 GB at 1 s), so it also lives on scratch; back it up to the
# project space (models/ archive step in README) after every commit.
MODELBASE="${MODELBASE:-$GPU_ROOT/models}"
SPECFEM_DIR="${SPECFEM_DIR:-$GPU_ROOT/specfem3d}"

## --- iteration identity ---------------------------------------------------
ITER=0                                   # input model = M$ITER, output = M$((ITER+1))
ORIMODEL="$MODELBASE/M$ITER"
OUTMODEL="$MODELBASE/M$((ITER + 1))"

## --- mesh + parallelism (from the planner) ---------------------------------
# tools/plan_mesh.py --apply writes mesh_plan.sh with NEX_XI NEX_ETA NEX_Z
# NPROC_XI NPROC_ETA NPROC DT NSTEP MESH_TARGET_S, and the matching
# DATA/meshfem3D_files/{Mesh_Par_file,interfaces.dat}. The defaults below are
# the 1.5 s plan so the tree is runnable before the planner is invoked.
if [[ -r "$GPU_ROOT/mesh_plan.sh" ]]; then
  source "$GPU_ROOT/mesh_plan.sh"
else
  MESH_TARGET_S=1.5
  NEX_XI=144; NEX_ETA=256; NEX_Z=33
  NPROC_XI=16; NPROC_ETA=32; NPROC=512
  DT=0.0110; NSTEP=9091
fi
MESHDATA="$MODELBASE/meshdata"           # NPROC proc*_Database files from 2_make_mesh.sh
DBDIR="$WORKBASE/db"                     # per-model xgenerate_databases output (NPROC external_mesh.bin each)

## --- SLURM shape ------------------------------------------------------------
# Anvil wholenode = 128 cores / 256 GB. Whole nodes are charged whole, so
# NPROC should be a multiple of CORES_PER_NODE (the planner enforces it).
CORES_PER_NODE="${CORES_PER_NODE:-128}"
NODES_PER_EVENT=$(( (NPROC + CORES_PER_NODE - 1) / CORES_PER_NODE ))
ARRAY_THROTTLE="${ARRAY_THROTTLE:-4}"    # events in flight per array (x NODES_PER_EVENT nodes)
                                         # 4 x 8 nodes = 32 nodes: check the account's running-node cap
EVENT_WALL="${EVENT_WALL:-01:30:00}"     # per forward task; 4_bench.sh tells you the real number
ADJ_WALL="${ADJ_WALL:-04:00:00}"         # per adjoint task (forward+save, then kernel)
SERVICE_WALL="${SERVICE_WALL:-04:00:00}" # genmesh / smooth / caladj / post / update jobs
# partitions + account come from env.sh (SLURM_ACCOUNT, PART_COMPUTE, PART_HIGHMEM, PART_DEBUG)

## --- solver runtime -------------------------------------------------------
# DT/NSTEP come from the plan (above). Band code follows DT: both planned
# meshes have DT <= 0.0125 -> *.HXZ.semd. cal_adjoint's glob accepts [BH]XZ;
# NO 2 s-era window cache or adjoint source may be mixed in (docs/GOTCHAS.md, item 1).
SEISMO_FORMAT="${SEISMO_FORMAT:-bin}"    # bare float32 traces (validated 2026-09-11); ascii at
                                         # NSTEP~11-15k would be ~350 MB/event of formatted I/O on GPFS
# Par_file invariants this port forces (lib/solver.sh prepare_workdir):
#   GPU_MODE=.false.  NPROC=$NPROC  MODEL=gll  SAVE_MESH_FILES=.false.
#   USE_BINARY_FOR_SEISMOGRAMS per run (off in SIMULATION_TYPE=3)

## --- event sets -----------------------------------------------------------
SRC_SET="${SRC_SET:-$GPU_ROOT/events/s2_keep150}"
POOL_SRC="${POOL_SRC:-$GPU_ROOT/events/allforward_all}"
LS_SUBSET="${LS_SUBSET:-$GPU_ROOT/events/c1_ls40h}"

## --- forward stage --------------------------------------------------------
# Same decoupling as ite3_gpu: the forward stage writes seismograms only; the
# adjoint task re-runs forward(SAVE_FORWARD) then the kernel leg back to back
# in the event's own DATABASES_MPI and deletes the wavefield. On Anvil the
# reason is not one workstation disk but the SIZE of the Stacey traction
# store at these meshes (planner: ~180 GB/event at 1.5 s, ~740 GB at 1 s).
# ARRAY_THROTTLE events hold that much scratch transiently, so budget
# ARRAY_THROTTLE x wavefield GB on /anvil/scratch (100 TB quota, check `myquota`).
FRAME_KEEP=0
FRAME_DIR="$WORKBASE/FRAME"
EVENT_MIN_SEMD=700                       # 932 stations -> >=700 *Z.semd files = complete

## --- adjoint + kernels ----------------------------------------------------
ADJSRC="$WORKBASE/adjoint_sources"
KERNELS="$WORKBASE/KERNELS"
ORI_KERNELS="$WORKBASE/ORI_KERNELS"
KERNEL_SUM="$KERNELS/condi_sum"
KERNEL_SMOOTH="$KERNELS/smooth_sum"
KERNEL_ARCHIVE=1

## --- clip + receiver mask (unchanged) -------------------------------------
CLIP_VALUE=1E-5
MASK=1
SIGMA_H=1700
SIGMA_V=1000

## --- smoothing (CPU kd-tree path, NPROC ranks) ----------------------------
SMOOTH_GPU=0
SMOOTH_SIGMA_H="${SMOOTH_SIGMA_H:-${SMOOTH_SIGMA_H_PLAN:-1500}}"   # 2000 m at 2 s; the plan scales it
SMOOTH_SIGMA_V="${SMOOTH_SIGMA_V:-200}"
SMOOTH_KERNELS="alpha_kernel_clip beta_kernel_clip"
SMOOTH_COMBINED=1

## --- L-BFGS model update --------------------------------------------------
LBFGS_STATE="$MODELBASE/LBFGS_STATE"     # fresh: partition-dependent, never reused across meshes
MBFGS=5
MAXPERT=0.03
# lbfgs_update.py holds (2*MBFGS+6) float64 copies of the 2-parameter model
# vector in RAM: ~40 GB at 1.5 s, ~160 GB at 1 s -> UPDATE_PARTITION=highmem for 1 s.
UPDATE_PARTITION="${UPDATE_PARTITION:-}" # empty = PART_COMPUTE

## --- line search ----------------------------------------------------------
LS_STEPS="0.01 0.03 0.05"
LS_CONV=0.003

## --- cal_adjoint ----------------------------------------------------------
# One or two whole nodes of mpi4py ranks (128/node); capped at the source count.
CALADJ_NODES="${CALADJ_NODES:-1}"
CALADJ_NPROC="${CALADJ_NPROC:-0}"        # 0 = CALADJ_NODES x CORES_PER_NODE
CALADJ_DATA="${CALADJ_DATA:-$GPU_ROOT/obs_data}"
CALADJ_EXCLUDE="${CALADJ_EXCLUDE:-$GPU_ROOT/obs_data/exclude_zo_line_n81.txt}"
# Band selection for this stage (cal_adjoint/config.py BAND_TABLE keys). The
# 1 s mesh enables 1_3; the 1.5 s mesh 1.5_4 (label kept dotted on purpose --
# it is only ever a directory/file suffix). Regenerate the observed band dirs
# with cal_adjoint/0_newprocess.py (README step 4) BEFORE the first cal_adjoint.
CALADJ_BANDS="${CALADJ_BANDS:-${CALADJ_BANDS_PLAN:-1.5_4,2_6,3_10,6_18}}"

## --- autonomous loop ------------------------------------------------------
REWINDOW=1
REWINDOW_TAG=s3                          # window cache prefix: obs_data/windows/s3_iter<N>
MAX_UPDATES=4
POLL=30
KEEP_GOING_ON_EVENT_FAIL=1
MAX_EVENT_RETRIES=1
STAGE_RETRIES=3

## --- housekeeping ---------------------------------------------------------
LOGDIR="$GPU_ROOT/log"
NICE_LEVEL=0
