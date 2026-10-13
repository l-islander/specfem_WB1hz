# ite3_gpu — FWI on a single RTX 5090

The local-workstation port of `ite3`, the original Purdue Anvil version of this workflow (not included here). Same inversion, same
file formats, same models — no scheduler, no queue, one GPU.

Setup notes from the original port; the data staging script (stage_from_anvil.sh) and the data are not included in this repository.

```bash
cp env.sh.example env.sh && $EDITOR env.sh   # point at your CUDA + MPI
./0_preflight.sh          # is this machine ready?
./1_build_specfem_gpu.sh  # build the CUDA solver for sm_120
./2_make_mesh.sh          # mesh at NPROC=1
./3_make_M0.sh            # bake the tomo file onto it -> M0
./smoke_test.sh           # <-- 2 events through the WHOLE chain, in minutes
./4_bench.sh              # <-- TIME ONE EVENT BEFORE COMMITTING A NIGHT
./run_loop.sh             # the autonomous inversion
./status.sh -w            # (from another shell) live progress
```

`smoke_test.sh` and `4_bench.sh` are the two things to run before the loop.
The smoke test drives forward → cal_adjoint → kernels → clip/sum/smooth → update
on 2 events in an isolated sandbox, so a wiring problem surfaces in minutes
instead of after hours of forward simulations. The benchmark then tells you what
the full loop will cost.

---

## What changed, and why

|  | Anvil `ite3` | local `ite3_gpu` |
|---|---|---|
| Ranks | 1250 (10 nodes × 125) | **1** (one GPU) |
| Scheduling | `sbatch` / `squeue` / `--dependency` / watchdogs | plain sequential bash |
| Concurrency | 10–20 group jobs at once | one event at a time |
| Solver | CPU (AOCC, `-march=znver3`) | **CUDA, `sm_120`** |
| Wavefield store | `FRAME/` — 28 GB × 627 events ≈ **18 TB** | transient, **~30 GB** peak |
| Smoothing | 40-min 10-node MPI job | `xsmooth_sem` on the GPU |
| "Event finished?" | `du -sm ≥ 300` | explicit sentinel + file count |

Four of these are more than simplification, and each has its own reasoning:

**1. `NPROC=1250 → 1` forces a local rebuild of the mesh and M0.** A partitioned
mesh is not a file format, it is the domain physically cut into 1250 pieces, and
nothing merges them back. So Anvil's `meshdata/` and `M0/` are not transferable —
`2_make_mesh.sh` and `3_make_M0.sh` rebuild both from the same
`tomography_model.xyz` (152 MB, which *is* transferable). Same field, one piece.
This also means the L-BFGS state starts fresh: curvature pairs are stored as
concatenated per-rank vectors, so a 1250-rank state is unreadable at NPROC=1. The
first update is pure steepest descent — as ite3's first update was too.

**2. The wavefield store is gone.** With Stacey absorbing conditions, a
`SAVE_FORWARD` run stores boundary tractions for every timestep: ~29 GB per event
at this mesh and `NSTEP=5000` (measured: 28 GB). ite3 keeps that so the adjoint
can skip its own forward. On a workstation, 2.8 TB for 99 events is not a trade
worth making. Here the forward stage runs `SAVE_FORWARD=.false.` (seismograms
only) and the **adjoint stage runs forward-then-kernel back-to-back per event**,
with the wavefield living in `DATABASES_MPI` for the few minutes it is needed and
deleted immediately after. Costs one extra forward per event; saves 2.8 TB of
disk and 5.6 TB of copying per iteration.

**3. Completion is an explicit sentinel.** ite3's `du -sm ≥ 300 MB` test is
calibrated to Anvil's GPFS, which allocates in 512 KB blocks — a finished event
measures 381 MB there but holds only 120 MB of data. On local ext4/xfs it
measures ~120 MB, *below* the threshold, so every event would re-run forever.
This port writes `.event_done` / `.kernel_done` after verifying the solver exited
0 and the expected files exist.

**4. A real bug is fixed on the way through** — see *Band code* below.

Full technical detail, with source citations: **[`docs/GPU_NOTES.md`](../../docs/GPU_NOTES.md)**.

---

## Setup

### Requirements

| | |
|---|---|
| GPU | RTX 5090 (or any card — set `GPU_ARCH` in `config.sh` to its compute capability) |
| CUDA | **≥ 12.8** (`sm_120` needs it) |
| Compilers | `gfortran`, `gcc` ≤ 14 (nvcc rejects newer), an MPI (`mpif90`/`mpirun`) |
| RAM | ≥ 24 GB (genmesh at NPROC=1, and the L-BFGS vectors) |
| Disk | ~150 GB free on `WORKBASE`, ideally NVMe |
| Python | `uv` (for the cal_adjoint venv) |

`0_preflight.sh` checks all of it, including *compiling a trivial `sm_120` kernel*
to prove your toolkit really supports the card.

### Getting the data here

(Historical: the staging script and the data are not included in this repository.)

```bash
# 1. copy this tree from Anvil
rsync -av --info=progress2 \
  <user>@anvil.rcac.purdue.edu:/anvil/scratch/<user>/wb_inversion/ite3_gpu/ \
  ~/fwi/ite3_gpu/

# 2. pull the inputs
cd ~/fwi/ite3_gpu
./stage_from_anvil.sh --user <user> --all     # not included in this repository
```

**~13 GB total**, of which 11.6 GB is the observed correlation data:

| what | size | why |
|---|---|---|
| SPECFEM3D source | ~1.1 GB | build artefacts excluded |
| `tomography_model.xyz` + mesh params | ~155 MB | M0 is rebuilt from this |
| event sets | ~2 MB | just `FORCESOLUTION` files |
| observed data (`3_10/`, `6_18/`) | ~11.6 GB | the misfit inputs |
| window cache | ~3 MB | |

Deliberately **not** transferred: `FRAME/` (18 TB, regenerable and unused here),
`M0/` + `meshdata/` (wrong partitioning), `result_M0*` (~240 GB of synthetics for
a partitioning that does not apply), `data/cc/` (11 GB, stage-0 output that
stages 1–4 never open), `data/final_sac/` (41 GB of raw SAC — only stage 0 needs
it, and stage 0 is model-independent and already done).

### Configure

`env.sh` — machine paths only (CUDA, MPI, compilers). The one file to edit per machine.

`config.sh` — the science and the workflow. Everything else reads it. Notable knobs:

| var | meaning |
|---|---|
| `GPU_ARCH` | compute capability without the dot (5090 → `120`) |
| `WORKBASE` | where the big transient data goes — **point this at your fastest large disk** |
| `MODELBASE` | where the models live — small and precious, keep it backed up |
| `ITER` | current iteration; the loop bumps it |
| `MAX_UPDATES` | how many model updates before stopping |
| `LS_STEPS` | line-search trial step lengths |
| `SMOOTH_GPU` | `1` = smooth on the GPU; set `0` if it OOMs |
| `FRAME_KEEP` | `1` restores Anvil's wavefield-store behaviour (needs the disk) |

---

## Running

### Benchmark first

`./4_bench.sh` runs one event and extrapolates the whole loop. **Do this before
`run_loop.sh`** — it is roughly two forward simulations and it tells you whether
an iteration is four hours or four days. It reports measured throughput
(M element-timesteps/s), peak VRAM, the transient wavefield size, and a projected
per-iteration and total wall time.

Measured on the 2 s production mesh (540,800 elements): 18.2 GPU-hours per iteration on the RTX
5090 (top-level README); everything downstream scales linearly from the per-event throughput.

### The loop

```bash
tmux new -s fwi './run_loop.sh'      # survives a dropped ssh session
tail -f log/run_loop.log

./status.sh          # where am I? one snapshot
./status.sh -w       # live view, refreshes every 20s
```

`status.sh` is read-only and safe to run at any time. It reports which iteration
and stage the loop is on, per-stage progress counts (events forwarded, kernels
computed, trial forwards done), GPU utilisation and VRAM, disk headroom, the
misfit series with its trend, and the last few log lines. It is the thing to run
when you walk back to the machine.

Per iteration, for model `M_x`:

1. **forward(M_x)** → seismograms *(GPU)*
2. **cal_adjoint(M_x)** → misfit + adjoint sources *(CPU, uses every core while the GPU idles)*
   → **misfit gate**: if misfit rose vs `M_{x-1}`, stop
3. **adjoint(M_x)** → per-event kernels → clip/mask → preconditioned sum → smooth *(GPU)*
4. **line search** → `alpha*` *(GPU; `LS_STEPS × LS_SUBSET` forwards — usually the most expensive step)*
   → if `alpha* ≤ 0`, no descending step exists: stop **without** committing
5. **commit** `M_{x+1}` at `alpha*` *(CPU)*
6. **forward(M_{x+1})**, then bump `ITER` and loop

Stops on: `MAX_UPDATES` reached, misfit rise, `alpha* ≤ 0`, convergence
(`alpha* < LS_CONV`), `touch ./STOP`, or any step failure.

**Everything is resumable.** There is no step counter — each step is skipped when
its on-disk product exists. Kill it whenever; re-run to continue. To force a
redo, delete that step's product (the log names it).

Graceful stop: `touch STOP` (halts at the next iteration boundary).
Immediate stop: Ctrl-C — safe, just re-run.

### Or run stages by hand

```bash
./forward/run_forward.sh --model $MODELBASE/M0 --limit 3   # smoke test
./cal_adjoint/run_caladj_local.sh --full
./adjoint/run_adjoint.sh --limit 2
./adjoint/run_post.sh
./adjoint/run_update.sh --step 0.02
```

Most accept `--limit N` (first N events), `--list` (show the plan), `--force`
(redo completed work). A single GPU lock (`flock`) prevents two solver runs
colliding.

---

## Layout

```
ite3_gpu/
├── config.sh                    SINGLE SOURCE OF TRUTH (science + workflow)
├── env.sh                       your machine (CUDA, MPI, compilers)  [you create this]
├── 0_preflight.sh               validate the machine
├── 1_build_specfem_gpu.sh       CUDA build with the sm_120 override
├── 2_make_mesh.sh               mesh at NPROC=1
├── 3_make_M0.sh                 tomo -> M0  (3 guards; see below)
├── 4_bench.sh                   time one event, extrapolate the loop
├── smoke_test.sh                2 events through the whole chain (sandboxed)
├── status.sh                    read-only progress view  (-w to watch)
├── run_loop.sh                  the autonomous driver
├── stage_from_anvil.sh          pull inputs from Anvil (not included in this repository)
├── lib/
│   ├── common.sh                logging, sentinels, gpu_run, locking, Par_file helpers
│   └── solver.sh                workdir staging, genmesh cache, per-event runners
├── forward/run_forward.sh       sequential forward over an event set
├── adjoint/
│   ├── run_adjoint.sh           per-event forward+kernel, transient wavefield
│   ├── run_post.sh              clip/mask -> sum -> smooth (GPU)
│   ├── run_update.sh            L-BFGS update
│   ├── lbfgs_update.py  lib/    carried over verbatim from ite3
│   ├── clip_iso.py  sum_precond.py
│   └── forward_update/linesearch.sh   parabolic step-length search
├── cal_adjoint/
│   ├── build_env_local.sh       one-time venv build (mpi4py from source)
│   ├── run_caladj_local.sh      stages 1-4 + misfit; --full / --misfit / --repick
│   ├── config.py                localized: every path from the environment
│   └── pyflex/  pyadjoint/      vendored, unchanged (not included in this repository)
├── DATA/                        Par_file, Mesh_Par_file, STATIONS, tomo
└── ../../docs/GPU_NOTES.md      the technical reference
```

Runtime data lives under `WORKBASE` (default `./work`) and models under
`MODELBASE` (default `./models`) — both outside the script tree, so the tree
stays rsync-able.

---

## Things worth knowing

### Band code — a bug this port fixes

SPECFEM derives the seismogram channel code from `DT`. ite2 ran `DT=0.005` →
`*.HXZ.semd`; **ite3 runs `DT=0.02` → `*.BXZ.semd`**. But
`cal_adjoint/1_process_data.py` hardcoded an `HXZ` glob.

Verified on Anvil: `forward/result_M0_p1/3A.20050_Z/` holds 756 `*.BXZ.semd` and
zero `HXZ`. So on ite3 that glob matches nothing — no windows measured, no `.adj`
written — **and the job still exits 0 reporting `NA 0` as the misfit**, which an
automated loop reads as success.

Fixed here two ways: the glob accepts any band code, and zero measurements is
treated as a hard failure rather than a misfit of zero.

**This is live in the Anvil ite3 tree and needs the same fix there.** Two other
Anvil-side issues are documented in [`docs/GPU_NOTES.md §8`](../../docs/GPU_NOTES.md):
`forward/result` is a dangling symlink, and `WINDIR` points at a cache that does
not exist.

### The M0 guards

`3_make_M0.sh` refuses to build if:

1. **the tomo z column is in kilometres.** This is the bug that invalidated all of
   ite2's M0–M3: x/y in metres, z in km, so SPECFEM put the whole 30 km mesh
   below the grid and clamped every point to the bottom layer (100 km, mantle,
   Vs ≈ 4.3 km/s).
2. **the mesh stamp disagrees** with `Mesh_Par_file`. `xgenerate_databases` takes
   geometry from `proc*_Database`, not from `Mesh_Par_file`, so editing the latter
   without re-meshing silently does nothing.
3. **the resulting median Vs looks like mantle** (> 4100 m/s) — the check that
   would have caught the km/m bug on day one. Fails without writing `.done`, so a
   bad M0 can never be used.

### Windows

The first `cal_adjoint` run must **pick** windows, not reuse them — the cache
`WINDIR` points at does not exist yet. `run_caladj_local.sh` detects an empty
cache and switches to picking automatically. Do not point `WINDIR` at
`windows/r2`: those are ite2's picks against the mantle-clamped model, and the
arrival times do not transfer.

### If the line search returns `alpha* ≤ 0`

No trial step reduced the misfit, so committing would give `M_{x+1} == M_x`. The
loop stops without committing. Re-tune the windows
(`./cal_adjoint/run_caladj_local.sh --repick`, adjusting `config.py`), delete
`work/adjoint/m{x}to{x+1}/`, and relaunch.

### Comparing against Anvil

The misfit series from this tree is **not** directly comparable to ite3's — the
starting models are rebuilt independently and the window cache differs. Compare
*trends* (does misfit fall, does the line search find sane steps), not absolute
values. Within this tree the series is self-consistent, which is what the misfit
gate needs.

---

## Troubleshooting

Ordered by how likely you are to hit them on a first run.

**`nvcc REJECTS -arch=sm_120`** — the toolkit predates Blackwell. Needs CUDA ≥ 12.8.
`nvidia-smi` showing a new driver is not enough; the *toolkit* is what compiles.

**`nvcc rejects host compiler gcc <N>`** — nvcc hard-errors on a gcc newer than it
supports (CUDA 12.8 → gcc ≤ 14). Either install an older gcc and set `CC`/`CXX`
in `env.sh`, or set `NVCC_EXTRA="-allow-unsupported-compiler"` there.

**Build succeeds but `cuobjdump` reports no `sm_120`** — the `GENCODE` override did
not take. Check the `GENCODE =` line in `specfem3d/Makefile`. Do **not** work
around it by running `autoreconf` — this tree's `m4/` is empty and that breaks
`configure` permanently (see [`docs/GPU_NOTES.md §1`](../../docs/GPU_NOTES.md)).

**`configure did NOT enable CUDA`** — `CUDA_INC`/`CUDA_LIB` are wrong. The build
script prints both paths and whether it found `cuda_runtime.h` and `libcudart`.

**Solver dies immediately with `stop 'GPU mode ...'`** — a Par_file setting the GPU
path rejects. `assert_gpu_parfile` should catch these first; if one slips
through, [`docs/GPU_NOTES.md §2`](../../docs/GPU_NOTES.md) has the full list.

**cal_adjoint reports zero measurements** — the guard fired, which is the intended
behaviour (Anvil's version would have silently reported `NA 0` as success). In
order of likelihood: synthetics not found (band code — check
`work/forward/result_*/<src>_Z/SYN/` really has `*.BXZ.semd`); observed data not
staged for those sources; window cache empty or non-overlapping. Per-rank detail
is in `cal_adjoint/log/`.

**mpi4py sees world size 1 under `mpirun -np N`** — the wheel's bundled MPI does not
match the launcher. `build_env_local.sh` rebuilds mpi4py from source against your
`mpicc` specifically to prevent this, and tests for it. If it warns, fix the MPI
install before trusting a misfit: the pipeline would do 1/N of the work N times
and report a wrong, low misfit with no error.

**`xsmooth_sem` out of memory** — the GPU path uses a brute-force neighbour search.
Try `SMOOTH_COMBINED=0` (one kernel at a time), then `SMOOTH_GPU=0` (CPU kd-tree,
much slower but far less memory).

**"another ite3_gpu run holds the GPU lock"** — one GPU, one solver. Check with
`ps -ef | grep -E 'xspecfem3D|run_loop'`; if nothing is running, delete `.gpu.lock`.

**Line search returns `alpha* ≤ 0`** — not a crash. No trial step reduced the
misfit, so the loop stops without committing (see above).

**Everything is resumable.** Almost any failure is recoverable by fixing the cause
and re-running the same command — finished work is skipped.

---

## Status

Written and statically verified on Anvil, which has **no GPU and no CUDA** — so
the CUDA build and every GPU run are, necessarily, unexercised. What *was*
verified:

- all 19 shell files pass `bash -n`; all 48 Python files compile under 3.11
- every entry script initializes cleanly (`--help` reached with no unbound-variable
  errors under `set -u`)
- **all 16 cross-script call sites checked programmatically** — every flag passed
  between scripts is one the callee's parser actually accepts
- `par_get`/`par_set` round-trip correctly against the real Par_file, preserving
  SPECFEM's exact column alignment (verified by diff against the original)
- `band_code` matches SPECFEM's `band_instrument_code` at DT = 0.02 / 0.005 / 0.5
- `list_events` handles both flat and grouped layouts, returning the right counts
  against the real Anvil event sets (99 / 40 / 536)
- completion-sentinel logic rejects empty and partial event dirs
- the band-code patch parses and tries the right codes in order
- `cal_adjoint/config.py` resolves correctly with defaults *and* with the env
  overrides `run_caladj_local.sh` sets
- `retry_stage` succeeds on a flaky command and gives up on a persistent one
- the kernel-filename chain verified end to end against the actual Python:
  `clip_iso.py` → `_clip.bin` → `sum_precond.py` → `condi_sum/` → `xsmooth_sem`
  → `_clip_smooth.bin` → `lbfgs_direction.py`'s `SUFFIX`
- `xsmooth_sem`'s argument order and its optional 6th GPU argument confirmed
  against `smooth_sem.F90:225-274`
- `0_preflight.sh` runs end to end and classifies this machine correctly (fails on
  GPU/CUDA, which is right — Anvil has neither)

The tree then went through an adversarial read-only review, which found and I
fixed **three bugs that would have blocked the loop outright**:

1. **Nested GPU lock deadlocked every line search.** `linesearch.sh` took the
   `flock`, then spawned `run_forward.sh`, which took it again in a new process.
   flock locks are per open-file-description, so the child's `exec 9>` made a new
   one and conflicted with its own parent. The first trial forward died → the
   line search died → the loop died, every iteration. `take_lock` is now
   re-entrant via an exported marker (nested calls pass, unrelated shells are
   still refused — both directions tested).
2. **Iteration 1 would have silently reused iteration 0's kernels.** The adjoint
   work dir and its `.kernel_done` sentinels were model-independent, so at ITER=1
   every event tested "done", the adjoint was skipped entirely, and M0's gradient
   was re-smoothed and archived as `m1to2`. Wrong science, no error. Now guarded
   by a `.kernel_model` stamp that wipes stale kernels, mirroring the
   `.ls_model` stamp the line search already had.
3. **The line search's step-0 fallback built a symlink farm that cal_adjoint
   rejected** — `find -type d` does not match a symlink-to-directory, so it
   reported zero sources and died. Now `find -L`.

Plus: `KERNEL_SUM`/`KERNEL_SMOOTH` pointed at directories nothing creates (which
broke the standalone `./run_update.sh` and smoke-test phase 5); a resume race
where being killed between the ITER bump and the `ls_alpha.txt` delete would make
the next iteration reuse the previous step length; an empty `alpha` reading as
"converged"; and the forward stage harvesting all three seismogram components
where the inversion uses only Z (3× the disk, ~74 GB → ~24 GB per iteration).

The first things to break will now be environmental — CUDA version, host-compiler
compatibility, MPI launch. `0_preflight.sh` is built to catch exactly those, and
`1_build_specfem_gpu.sh` verifies with `cuobjdump` that the binary really contains
`sm_120` code before you trust a timing.
