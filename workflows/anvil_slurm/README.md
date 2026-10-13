# anvil_cpu — the ite3_gpu FWI workflow back on Purdue Anvil, for a 1.5 s / 1 s mesh

Self-contained subfolder of `ite3_gpu`. Nothing in the parent tree is modified
or imported at run time: the science code (`cal_adjoint/*.py`, `pyflex`,
`pyadjoint`, `adjoint/{clip_iso,sum_precond,lbfgs_update}.py`, `polyfit.py`,
`select_ls_group.py`) is **copied verbatim** from the parent as of 2026-09-18,
so rsync this directory alone to Anvil.

```
tools/plan_mesh.py 1.5             size the mesh: NEX, decomposition, DT, NSTEP, disk, SU
tools/plan_mesh.py 1.5 --apply     write mesh_plan.sh + DATA/{Mesh_Par_file,interfaces.dat,Par_file}
./0_preflight.sh                   modules, sbatch, account, scratch, venv, data
./1_build_specfem_cpu.sh [--reuse] CPU build (znver3) or reuse the project tree's binaries
./2_make_mesh.sh                   xmeshfem3D at NPROC        (1 SLURM job)
./3_make_M0.sh                     M6-export tomo -> M0       (1 SLURM job)
./4_bench.sh                       1 event fwd + fwd/save: core rate, wavefield GB, SU/iteration
./smoke_test.sh                    2 events through the chain, sandboxed
tmux new -s fwi ./run_loop.sh      the loop (login node; it only submits and waits)
./status.sh -w                     squeue + loop state
```

## 1. What changed from the single-GPU tree, and what did not

| | local `ite3_gpu` (5090) | `anvil_cpu` |
|---|---|---|
| ranks per event | 1 | `NPROC = NPROC_XI × NPROC_ETA` (512 at 1.5 s, 2048 at 1 s) |
| events at once | 1 | a **SLURM job array**, one event per task, `ARRAY_THROTTLE` in flight |
| drivers | plain bash | same scripts, each stage `sbatch --wait`s its job(s); `lib/slurm.sh` |
| solver dir | one per stage | one **per model** (`work/db/<model>`, genmesh once) + one **per event per stage** with NPROC symlinked mesh files |
| solver | CUDA `sm_120` | CPU, `-march=znver3`, `GPU_MODE=.false.` |
| smoothing | `xsmooth_sem` GPU | `xsmooth_sem` CPU kd-tree at NPROC ranks |
| seismograms | ascii | binary (`SEISMO_FORMAT=bin`, validated 2026-09-11) |
| lock | flock on the GPU | flock on the driver only; SLURM serialises the rest |

Kept exactly: sentinel resumability (`.event_done`, `.kernel_done`,
`.stage_done`, `.state/ite<N>`), the stale-kernel and stale-linesearch stamps,
the transient wavefield (forward+save then kernel in the same task, wavefield
deleted after), REWINDOW with the same-basis misfit gate and backward check,
the kernel-pair manifest line search, `LS_STEPS="0.01 0.03 0.05"`, the
zero-measurement guard. `run_loop.sh` is the parent's file with three lines
changed (preflight checks and the caladj script name).

Why the transient wavefield still, on a machine with 100 TB of scratch: the
Stacey traction store is `faces × 25 × 12 B × NSTEP` per event, which the
planner puts at **~170 GB per event at 1.5 s and ~620 GB at 1 s** (77 GB at
2 s, measured). Keeping it for 150 events is 25–90 TB per iteration; keeping it
for `ARRAY_THROTTLE` events for a few minutes is 0.7–1.3 TB. The forward leg is
repeated instead, as on the 5090.

## 2. Mesh plans (from the measured 2 s mesh, `tools/plan_mesh.py`)

Scaled from the stage-2 mesh that ran: 104×208×25, min period 1.966 s, DT
0.0145 (Courant 0.485), NSTEP 6900, 3.7 G element-timesteps per forward. Cost
uses an **effective 0.15 M element-timesteps/s per core** (wall based): the
dry run on this box's CPU measured 0.14 (16 ranks, 480k elements, 24 min per
forward); SPECFEM3D is memory-bandwidth bound, so Milan at 128 ranks per node
should land in 0.1–0.3. The 5090 does ~64 on the same mesh, so **one Anvil
node is worth roughly a third of the 5090** — the gain is the 16+ nodes, not
the cores. `4_bench.sh` measures the real rate; re-run the planner with
`--core-rate`.

| target | NEX | NPROC (nodes/event) | el/rank | DT / NSTEP | band | forward | wavefield/event | kernels/150 ev | L-BFGS RAM | SU / iteration | wall / iteration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1.5 s** | 144×256×33 = 1.22 M | 512 = 16×32 (4) | 2376 | 0.0110 / 9091 | HX | ~2.4 min | 173 GB | 365 GB | 39 GB | **~19,000** | ~9 h at 4 in flight (16 nodes) |
| **1.0 s** | 224×384×49 = 4.21 M | 2048 = 32×64 (16) | 2058 | 0.0070 / 14286 | HX | ~3.3 min | 624 GB | 1.26 TB | 135 GB | **~104,000** | ~25 h at 2 in flight (32 nodes) |

An iteration is 930 forward-equivalents (150 forwards + 150 × (1.0 save + 3.4
kernel) + 3 × 40 line search; the save and kernel factors are the dry run's
measured 24 / 24 / 80 min). Anvil `wholenode` caps a *job* at 16 nodes;
array tasks are separate jobs, but check the account's running-node limit
before setting `ARRAY_THROTTLE`. Both meshes put DT below 0.0125, so SPECFEM
writes `*.HXZ.semd`: **no 2 s-era window cache, adjoint source or result may
be mixed in** (docs/GOTCHAS.md, item 1); the loop's window prefix is `s3_`.

Recommendation: do 1.5 s first. It is a 3× step from what is proven, ~19k SU
per iteration (a 4-iteration stage ≈ 80k SU), and its data band (1.5–4 s) is
measured (below). The 1 s mesh is a 16× step, ~105k SU per iteration (a stage
≈ 400k SU), 2 events in flight at best, ~1 day of wall per iteration, and the
L-BFGS update wants a full node's RAM. Bench first: if Milan measures 0.3
instead of 0.15 every SU number halves.

## 3. The data gate (done 2026-09-18 on the 5090 box)

`tools/band_snr_survey.py` ran cal_adjoint stage 0's own per-pair SNR test on
2,000 random raw CC pairs (`obs_data/final_sac_new`, 20 Hz, 400 s):

| band | distance gate pass | SNR ≥ 3 | both | pass rate 10–20 / 20–40 / 40–80 km |
|---|---|---|---|---|
| 1_3 | 89.5 % | 33.1 % | **32.1 %** | 46 / 56 / 32 % |
| 1.5_4 | 86.2 % | 32.8 % | **31.8 %** | 46 / 56 / 40 % |
| 2_6 (control, carried stage 2) | 80.0 % | 27.8 % | 26.2 % | 41 / 46 / 44 % |
| 3_10 | 75.9 % | 29.1 % | 27.3 % | 45 / 56 / 41 % |

The short bands survive at least as often as 2_6 and hold to 40–80 km, so the
noise data supports the stage. The group-velocity gates in
`cal_adjoint/config.py: BAND_TABLE` (vsmin 0.5–0.6 km/s for basin Rayleigh
waves at 1–2 s) are a starting point to confirm on the stage-0 output.

## 4. Runbook on Anvil

### 4.1 Transfer (on the 5090 box)

```bash
cd workflows/anvil_slurm
./stage_to_anvil.sh                 # code, trimmed specfem3d, tomo export of models/M6, events, raw obs (3.6 GB)
rsync -avP bundle/anvil_<date>/ <user>@anvil.rcac.purdue.edu:/anvil/scratch/<user>/wb_s3/bundle/
```

The starting model is `models/M6` (2 s final = global M15) exported by
`tools/gll_to_tomo.py` on the parent's NPROC=1 mesh, so it survives the
re-mesh and re-partition (GLL models are welded to their mesh; L-BFGS state
is dropped — first update is preconditioned steepest descent, as always under
REWINDOW). Note for `3_make_M0.sh`: SPECFEM's tomography reader loads the
whole grid on **every rank** (68.4 M records at the 438 m export = 1.0 GB of
arrays per process, and the 5.4 GB text file is read NPROC times). 128 ranks
× 1 GB fits a 256 GB node; the read is a one-time GPFS cost. On the 30 GB
5090 box the same bake at 16 ranks was OOM-killed, which is why the dry run
uses the coarse 1.3 km campaign-C export instead.

### 4.2 Unpack, toolchain, build

```bash
mkdir -p ~/ite3_anvil && cd ~/ite3_anvil
B=/anvil/scratch/$USER/wb_s3/bundle; sha256sum -c $B/MANIFEST.sha256
tar xzf $B/code.tar.gz --strip-components=1        # anvil_cpu/* -> here
tar xzf $B/specfem3d.tar.gz                        # -> specfem3d/
mkdir -p DATA/tomo_files && gunzip -c $B/tomography_model_M6.xyz.gz > DATA/tomo_files/tomography_model.xyz
tar xzf $B/events.tar.gz                           # -> events/
mkdir -p /anvil/scratch/$USER/wb_s3 && tar -C /anvil/scratch/$USER/wb_s3 -xf $B/obs_data.tar
ln -s /anvil/scratch/$USER/wb_s3/obs_data obs_data
cp env.sh.anvil env.sh && $EDITOR env.sh           # [EDIT]: modules, SLURM_ACCOUNT, partitions
./0_preflight.sh
./1_build_specfem_cpu.sh          # or --reuse (/anvil/projects/x-<allocation>/specfem3d-4.1.1_wb/bin) if its MPI matches the modules
curl -LsSf https://astral.sh/uv/install.sh | sh && cal_adjoint/build_env_anvil.sh
```

`build_env_anvil.sh` compiles mpi4py against the loaded `mpicc` and runs a
2-rank `srun` test: it must print `world size = 2`. A PyPI mpi4py wheel with
its own MPI makes every rank believe it is rank 0 of 1 and the misfit comes
out wrong and *low*, with no error.

### 4.3 Mesh, model, observed bands, benchmark

```bash
tools/plan_mesh.py 1.5 --apply    # or 1.0; writes mesh_plan.sh (sourced by config.sh) + DATA files
./2_make_mesh.sh                  # 1 job, NPROC ranks, saves NPROC Database files to models/meshdata
./3_make_M0.sh                    # 1 job: MODEL=tomo bake + crustal-Vs acceptance -> models/M0
# observed data for the new band set (stage 0 regenerates ALL active bands with a fresh
# station-cover reorder; run once, ~1 h on a node):
sbatch --account=$SLURM_ACCOUNT -p wholenode -N1 -n128 -t 02:00:00 --wrap \
  "cd $PWD/cal_adjoint && CALADJ_BANDS=1.5_4,2_6,3_10,6_18 srun -n 128 .venv/bin/python 0_newprocess.py \
     --sac-dir ../obs_data/final_sac_new --out-dir ../obs_data/final_sac_wb_s3"
# then in env.sh:  export CALADJ_CCDIR=$PWD/obs_data/final_sac_wb_s3
./4_bench.sh                      # measured core rate -> tools/plan_mesh.py 1.5 --core-rate <x> to re-price
./smoke_test.sh                   # 2 events end to end (6 small jobs); PASS = plumbing
```

The first `cal_adjoint` picks windows (empty `obs_data/windows/s3_iter0` →
pick mode, automatic). `EVENT_WALL` / `ADJ_WALL` in `config.sh` should be set
from the bench numbers with margin: an array task that hits its wall time is
just a failed event that the next re-invocation retries, but it wastes the
allocation.

### 4.4 The loop

```bash
tmux new -s fwi './run_loop.sh'     # login node; only sbatch --wait + file ops
./status.sh -w                      # from another shell
touch STOP                          # graceful stop at the iteration boundary
```

Per iteration the driver submits: `fwd_M<x>` (array), `caladj_full`,
`caladj_misfit` (REWINDOW gate + backward check), `adj_M<x>` (array),
`post_clipsum`, `post_smooth`, per trial step `genmesh_pert_<a>` +
`fwd_pert_<a>` (array) + `caladj_misfit`, `update_M<x>`, `fwd_M<x+1>` (array).
Everything is resumable: kill the driver, relaunch, finished events and stages
are skipped. Each stage script also works standalone with the parent's flags
(`--limit`, `--force`, `--list`). Job logs: `log/slurm/<name>_<jobid>[_<task>].out`;
per-task solver logs: `log/forward_tasks.log`, `log/adjoint_tasks.log`.

### 4.5 Keep, back up, clean

`MODELBASE` and `WORKBASE` are on scratch (purged). After each commit copy
`models/M<x>`, `work/adjoint/m<x>to<x+1>` and `misfit_history.txt` to the
project space. Per-event kernels (`work/adjoint/group_0/RUN`, ~365 GB at
1.5 s, 1.3 TB at 1 s) and the clipped copies are regenerable; the parent's
keep-only-kernels policy applies (confirm before deleting result data).

## 5. Dry run on the 5090 box (done 2026-09-18, PASSED)

`tools/mock_slurm/sbatch` runs `--wrap` commands in-process, sequentially, with
the SLURM variables set, so the drivers can be exercised without a scheduler.
`env.sh.mock5090` puts it on PATH and sets `CORES_PER_NODE=16`, `MPIRUN=mpirun`.
The dry run used a copy of this tree (`mock_run/tree`: links to the parent's
CUDA binaries run with `GPU_MODE=.false.`, the coarse campaign-C tomo, obs
data, events, venv) at the 2 s resolution with NPROC=16
(`plan_mesh.py 2.0 --cores-per-node 16 --nproc-xi 4 --nproc-eta 4 --apply`:
100×192×25 = 480k elements, DT 0.0145, NSTEP 6897, BX), windows from
`s2_iter6`, bands 2_6/3_10/6_18. Log: `mock_run/chain2.log`.

| stage | result | wall on 16 cores |
|---|---|---|
| `2_make_mesh.sh` | 16 Database files, stamp written | 1 s |
| `3_make_M0.sh` | 48 model files, median Vs 3678 m/s accepted | 76 s |
| genmesh(M0) → `work/db/M0` | 16 `external_mesh.bin` + `values_from_mesher.h` | 61 s |
| forward array (2 events) | 932 binary traces + sidecar each, `.event_done` | 24 min per event |
| `cal_adjoint --full` (self-submitted, 2 ranks) | misfit 20.34 over 37 pairs, 36 `.adj` per event | 6 s |
| adjoint array (2 events) | save leg 24 min + 66.2 GB wavefield (planner: 70), kernel leg 80 min, 112 kernel files per event, wavefield dropped | 104 min per event |
| `run_post.sh` (clip+sum job, then smooth job at 16 ranks) | 32 smoothed files, finite, archived to `m0to1` | 3 s + 2 s + 6 min |
| `run_update.sh` (self-submitted) | 48 model files, max \|ln(M1/M0)\| = 0.01 | 3 s |

Two bugs found and fixed by the run: `CALADJ_BANDS` was not exported to the
Python stage (config.py silently used its default band list), and
`smoke_test.sh --keep` returned a false nonzero status. Measured factors now
in the planner: effective 0.14 M element-timesteps/s per core, save leg 1.0×
and kernel leg 3.4× a forward. The 66 GB wavefield per event is why the
transient design stays.

## 6. Status

Written on the 5090 box without access to Anvil. Statically verified: every
script passes `bash -n`; `config.py` band selection and the planner run; the
NPROC-file plumbing (per-model DB dir, per-event symlinked workdir, harvest,
kernel move at NPROC files, `xsmooth_sem` CPU call, `lbfgs_update.py --nproc`)
was exercised end to end by the mock dry run (section 5). Not exercised: the
line search driver (`linesearch.sh`, 3 edits from the parent) and `run_loop.sh`
itself, both pure drivers over the tested stages; and real
`sbatch` semantics (`--wait` on arrays, `--export=ALL` propagation of
`CALADJ_WINDIR`/`LS_SUBSET`, `srun --mpi=pmi2`), module names, account and
partition names, scratch quotas. `0_preflight.sh` and `smoke_test.sh` are
built to surface exactly those on day one.

See `doc/DESIGN_NOTES.md` for the per-file rationale and the traps carried
over from docs/GOTCHAS.md that still apply here.
