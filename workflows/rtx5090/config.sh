# ite3_gpu/config.sh -- single source of truth for the SINGLE-GPU (RTX 5090) FWI workflow.
#
# This is the LOCAL-WORKSTATION port of ite3 (which runs on Anvil under SLURM).
# Same science, same file formats, same models -- completely different execution model:
#
#            Anvil ite3                     |   local ite3_gpu
#   ---------------------------------------- + ----------------------------------------
#    NPROC = 1250 (10 nodes x 125 ranks)    |   NPROC = 1  (one MPI rank, one GPU)
#    sbatch / squeue / --wait / watchdogs   |   plain sequential bash, sentinel-resumable
#    N group jobs run CONCURRENTLY          |   events run ONE AT A TIME
#    FRAME kept for all events (2.8 TB)     |   FRAME is TRANSIENT per event (~30 GB peak)
#    CPU solver (AOCC/OpenMPI, znver3)      |   CUDA solver, sm_120 (Blackwell)
#
# Everything below is read by every orchestrator in this tree. Edit HERE, once.
# Machine-specific toolchain paths (CUDA, MPI, compilers) live in env.sh instead --
# that file is the ONLY thing you must touch when moving to a different box.
#
#   ./0_preflight.sh          sanity-check the machine before anything else
#   ./1_build_specfem_gpu.sh  build the CUDA solver
#   ./2_make_mesh.sh          mesh at NPROC=1
#   ./3_make_M0.sh            bake the tomo file onto the mesh -> M0
#   ./4_bench.sh              time ONE event, extrapolate the whole loop  <-- run this first!
#   ./run_loop.sh             the autonomous FWI loop
#
# ---------------------------------------------------------------------------

## --- where everything lives ----------------------------------------------
# GPU_ROOT is derived, not hardcoded: every script sources this file by absolute
# path, so the tree can be rsync'd anywhere (/home, /mnt/nvme, /data ...) and it
# still works. Do not set GPU_ROOT by hand.
GPU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# WORKBASE: the parent for everything that is BIG and REGENERATABLE
# (scratch wavefields, per-event kernels, forward results). Point this at your
# fastest large disk -- ideally NVMe, because the adjoint stage streams ~30 GB
# per event through it. Defaults to inside the tree; override if that disk is small.
WORKBASE="${WORKBASE:-$GPU_ROOT/work}"

# MODELBASE: the models (M0, M1, ...). SMALL (~150 MB per model at NPROC=1) and
# PRECIOUS -- keep on a disk you back up. NOT under WORKBASE on purpose.
MODELBASE="${MODELBASE:-$GPU_ROOT/models}"

# SPECFEM_DIR: your LOCAL copy of the SPECFEM3D 4.1.1 source tree, built for CUDA
# by 1_build_specfem_gpu.sh. This is a private copy -- unlike Anvil, nothing else
# on the machine shares it, so rebuilding is safe at any time.
SPECFEM_DIR="${SPECFEM_DIR:-$GPU_ROOT/specfem3d}"

## --- iteration identity ---------------------------------------------------
ITER=6                                   # this iteration: input model = M$ITER, output = M$((ITER+1))
ORIMODEL="$MODELBASE/M$ITER"             # model the kernels are computed on (GLL proc*_{vp,vs,rho}.bin)
OUTMODEL="$MODELBASE/M$((ITER + 1))"     # NEW model the L-BFGS update writes (+ .done sentinel)

## --- mesh + parallelism ---------------------------------------------------
# THE defining change of this port. One GPU == one MPI rank == one mesh partition.
# NPROC must equal NPROC_XI * NPROC_ETA in DATA/meshfem3D_files/Mesh_Par_file AND
# NPROC in DATA/Par_file. 2_make_mesh.sh enforces all three agree before meshing.
NPROC=1
NPROC_XI=1
NPROC_ETA=1
MESHDATA="$MODELBASE/meshdata"           # saved proc000000_Database (skips the mesher on later runs)

# Mesh size, for the sanity checks + the memory estimate printed by 0_preflight.sh.
# These MUST match Mesh_Par_file; 2_make_mesh.sh re-reads the file and errors on drift.
NEX_XI=104
NEX_ETA=208
NEX_Z=25                                 # vertical elements (interfaces.dat last field)

## --- GPU ------------------------------------------------------------------
GPU_ARCH=120                             # CUDA compute capability WITHOUT the dot.
                                         #   RTX 5090 / 5080 / 5070 (Blackwell GeForce) = 120
                                         #   RTX 4090 (Ada) = 89 | A100 = 80 | H100 = 90
                                         # 1_build_specfem_gpu.sh turns this into
                                         #   -gencode=arch=compute_$GPU_ARCH,code="sm_$GPU_ARCH,compute_$GPU_ARCH"
                                         # and OVERRIDES the Makefile's GENCODE (SPECFEM 4.1.1 ships
                                         # nothing above sm_90, so a stock --with-cuda=cuda12 build
                                         # produces Hopper cubins that only run on a 5090 via slow PTX JIT).
CUDA_MAJOR_MIN=12                        # minimum CUDA toolkit major
CUDA_MINOR_MIN=8                         # ... and minor. sm_120 needs >= 12.8; preflight enforces it.
GPU_DEVICE=0                             # which GPU to use (CUDA_VISIBLE_DEVICES). Single-card box: 0.
GPU_VRAM_MIN_MB=12000                    # refuse to start if the card has less than this free

## --- solver runtime -------------------------------------------------------
NSTEP=6900                               # timesteps per simulation (2s fine mesh)
DT=0.0145                               # s (2s fine mesh; still BX band code)
# Seismogram file format written by xspecfem3D (Par_file USE_BINARY_FOR_SEISMOGRAMS):
#   ascii : two-column "t u" text, one file per trace (the historical format;
#           writing 932 traces costs ~13 s of a 58 s forward on the 5090)
#   bin   : the same <net>.<sta>.<band>Z.semd NAME, but the bare float32 trace
#           (values only, no time column) -- SPECFEM's own switch, ~27 KB/trace.
#           The time axis is recovered from the seismo_meta.txt sidecar the
#           runner writes next to the traces (dt + start time from the solver
#           log). Every reader in this tree (cal_adjoint, grace_hopper/validate_forward,
#           eq_validation) detects the format per file, so mixed result dirs
#           are fine. Reference results (work/forward/result_M6) are ascii.
SEISMO_FORMAT="${SEISMO_FORMAT:-ascii}"
# Par_file settings this port FORCES (see doc/GPU_NOTES.md for why each one):
#   GPU_MODE            = .true.    <- the whole point
#   NPROC               = 1
#   SAVE_MOHO_MESH      = .false.   <- hard stop in GPU mode
#   USE_LDDRK           = .false.   <- hard stop in GPU mode
#   LTS_MODE            = .false.   <- hard stop in GPU mode
#   PML_CONDITIONS      = .false.   <- use STACEY instead (already the case)
# Everything else (ATTENUATION=.false., ANISOTROPY=.false., NGNOD=8, MODEL=gll,
# STACEY_ABSORBING_CONDITIONS=.true., APPROXIMATE_HESS_KL=.true.) is already
# GPU-legal in the ite3 Par_file and is carried over unchanged.

## --- event sets -----------------------------------------------------------
# Same layout as ite3: a directory of <src>_Z/ event dirs, each holding a
# FORCESOLUTION. The Anvil version pre-groups them into group_*/RUN/ for
# concurrent SLURM jobs; here there is nothing to group -- one GPU, one event at
# a time -- so the runners accept BOTH a flat <src>_Z/ dir and a grouped
# group_*/RUN/<src>_Z tree and just flatten it.
SRC_SET="${SRC_SET:-$GPU_ROOT/events/s2_keep150}"        # stage-2 top-150 by 3-band pairs (98.7%)
# NOT YET IMPLEMENTED in this port: ite3's `1_auto_forward.bash --add-pool` (diff
# the pool against the finished set, forward only the missing sources) has no
# equivalent here. To grow coverage manually, point SRC_SET at the pool and re-run
# run_forward.sh with the same --out: finished events are skipped by their
# .event_done sentinel, so it forwards only what is missing. Kept as a path for
# the source pool (staged by a data-transfer script not included in this repository).
POOL_SRC="${POOL_SRC:-$GPU_ROOT/events/allforward_all}"
LS_SUBSET="${LS_SUBSET:-$GPU_ROOT/events/c1_ls40h}" # 40-event subset for line-search trial forwards

## --- forward stage --------------------------------------------------------
# BIG DIFFERENCE FROM ANVIL. On Anvil the per-iteration forward runs with
# SAVE_FORWARD=.true. and parks the reconstruction wavefield in FRAME/ so the
# adjoint can skip its own forward. That store is ~29 GB PER EVENT (the Stacey
# absorbing-boundary traction history: 19350 faces x 25 GLL x 3 x 4 B x 5000 steps).
# For 99 events that is 2.8 TB -- fine on a Lustre scratch, absurd on a workstation.
#
# So here the two stages are decoupled:
#   forward stage : SAVE_FORWARD=.false.  -> seismograms only (~250 MB/event)
#   adjoint stage : per event, run forward(SAVE_FORWARD=.true.) then SIMULATION_TYPE=3
#                   back-to-back, then DELETE the wavefield before the next event.
# Costs one extra forward per event; on a GPU that is minutes, and it turns a
# 2.8 TB requirement into a ~30 GB high-water mark. FRAME_KEEP=1 restores the
# Anvil behaviour if you ever have the disk for it.
FRAME_KEEP=0
FRAME_DIR="$WORKBASE/FRAME"              # only used when FRAME_KEEP=1
# COMPLETENESS. The Anvil workflow infers "this event finished" from `du -sm` of
# the event dir against EVENT_MIN_MB=300. Do NOT carry that heuristic over: it is
# FILESYSTEM-DEPENDENT. Anvil's GPFS allocates in 512 KB blocks, so each 165 KB
# .semd occupies 512 KB and an event measures 381 MB -- but only 120 MB of it is
# real data. On a local ext4/xfs/zfs the same event measures ~120 MB, i.e. BELOW
# the 300 MB threshold, so every event would look unfinished forever: the runner
# would re-run finished events endlessly and --finalize would never fire.
# This port therefore uses an EXPLICIT SENTINEL instead -- the runner writes
# `.event_done` only after the solver exits 0 AND the expected seismogram count
# is present. Deterministic, filesystem-independent, and cheap to check.
EVENT_MIN_SEMD=700                       # minimum *.semd files for an event to count as complete.
                                         # 756 stations -> 756 files; 700 tolerates a few receivers
                                         # falling outside the mesh. Re-check after a STATIONS change.

## --- adjoint + kernels ----------------------------------------------------
ADJSRC="$WORKBASE/adjoint_sources"       # where cal_adjoint stage 4 writes <event>/{FORCESOLUTION,STATIONS_ADJOINT,SEM/*.adj}
KERNELS="$WORKBASE/KERNELS"              # per-event raw kernels
ORI_KERNELS="$WORKBASE/ORI_KERNELS"      # clipped/masked kernels, input to the sum
# These MUST match where run_post.sh actually writes -- xsmooth_sem and
# sum_precond.py put their output under $KERNELS, and run_update.sh defaults
# --grad to KERNEL_SMOOTH. Pointing them anywhere else silently breaks the
# standalone `./run_update.sh` entry point and smoke_test's last phase.
KERNEL_SUM="$KERNELS/condi_sum"          # summed + preconditioned gradient
KERNEL_SMOOTH="$KERNELS/smooth_sum"      # smoothed gradient (the search direction input)
KERNEL_ARCHIVE=1                         # 1 = keep each iteration's smoothed gradient as adjoint/m{x}to{x+1}/

## --- clip + receiver mask (unchanged from ite3) ---------------------------
CLIP_VALUE=1E-5                          # xclip_sem clamp magnitude
MASK=1                                   # 1 = apply per-event receiver taper, 0 = off
SIGMA_H=1700                             # receiver-taper horizontal width (m)
SIGMA_V=1000                             # receiver-taper vertical width (m)

## --- smoothing ------------------------------------------------------------
# xsmooth_sem is the ONE post-processing tool with a CUDA path in 4.1.1 -- it
# takes GPU_MODE as a 6th positional argument. On Anvil this was a 40-minute
# 10-node job; on the 5090 it should be minutes. Set SMOOTH_GPU=0 to fall back
# to the CPU (kd-tree) path if the GPU brute-force search runs out of memory.
# NB these are NOT the SIGMA_H/SIGMA_V above -- those are the receiver-taper widths
# used by the clip/mask step. The SMOOTHING sigmas are a different, larger pair,
# hardcoded as `2500 200` in ite3's run_smooth_group_anvil.slurm. Carried over
# verbatim so the gradient is smoothed identically to the Anvil runs.
SMOOTH_GPU=1
SMOOTH_SIGMA_H=2000                      # horizontal smoothing half-width (m) -- 2000 for the 2s fine mesh
SMOOTH_SIGMA_V=200                       # vertical   smoothing half-width (m)
SMOOTH_KERNELS="alpha_kernel_clip beta_kernel_clip"   # rho is derived via RHO_SCALING, never smoothed
SMOOTH_COMBINED=1                        # 1 = smooth all SMOOTH_KERNELS in ONE xsmooth_sem call
                                         # (it accepts a comma-delimited list and does the expensive
                                         # neighbour search once for all of them -- roughly half the
                                         # cost of two calls). Set 0 to go one at a time if memory
                                         # is tight; memory grows as NGLL3*NSPEC*n_kernels.

## --- L-BFGS model update --------------------------------------------------
LBFGS_STATE="$MODELBASE/LBFGS_STATE"     # persistent across iterations. FRESH for this tree:
                                         # curvature pairs are partition-dependent, so ite3's
                                         # 1250-rank state CANNOT be reused at NPROC=1.
MBFGS=5                                  # L-BFGS history length
MAXPERT=0.03                             # trust-region cap on |ln-model change| (only used without a line search)

## --- line search (parabolic step-length search) ---------------------------
LS_STEPS="0.01 0.03 0.05"                # trial step lengths (--no-clamp, so step == max|ln-model change|)
LS_CONV=0.003                            # lowered 2026-08-12: under REWINDOW a small alpha* self-heals next iteration (frozen-window-era 0.01 stopped stage-2 at alpha*=0.0084)

## --- cal_adjoint (offline adjoint-source pipeline) ------------------------
# Pure Python + mpi4py, no GPU. This is the one stage that WANTS many CPU cores,
# so it is the natural place to use the rest of the box while the GPU idles.
CALADJ_NPROC="${CALADJ_NPROC:-0}"        # 0 = auto (use all physical cores, capped at the event count)
CALADJ_DATA="${CALADJ_DATA:-$GPU_ROOT/obs_data}"   # observed CC / STATIONS / frozen windows
CALADJ_EXCLUDE="${CALADJ_EXCLUDE:-$GPU_ROOT/obs_data/exclude_zo_line_n81.txt}"   # ZO-line same-line pair filter (transect_diagnosis)

## --- autonomous loop ------------------------------------------------------
REWINDOW=1                               # 1 = fresh windows + LS group EVERY iteration (dual-measure gate)
REWINDOW_TAG=s2                          # window cache prefix: obs_data/windows/${TAG}_iter<N>
MAX_UPDATES=4                            # model updates per launch of run_loop.sh
POLL=10                                  # (currently unused -- there is nothing to poll without a
                                         # scheduler; progress is printed per event as it completes.
                                         # Kept so status.sh/config stay recognisable next to ite3.)
KEEP_GOING_ON_EVENT_FAIL=1               # 1 = a single event that crashes is logged + skipped, loop continues
                                         # 0 = any event failure aborts the whole iteration
MAX_EVENT_RETRIES=1                      # re-run a failed event this many times before giving up on it
STAGE_RETRIES=3                          # run_loop re-invokes a whole failed stage this many times.
                                         # Stages are strict (no .stage_done unless EVERY event finished,
                                         # because a missing event changes the misfit basis and would
                                         # corrupt the cross-iteration gate) and resumable (a re-invocation
                                         # retries only what failed) -- so retrying lets a transient
                                         # failure self-heal overnight while a reproducible one still stops.

## --- housekeeping ---------------------------------------------------------
LOGDIR="$GPU_ROOT/log"
NICE_LEVEL=0                             # `nice -n` for solver runs; raise if you use the box interactively
