# GPU notes — SPECFEM3D 4.1.1 on an RTX 5090

Reference material for this port. Every claim here was checked against the
SPECFEM source tree at `/anvil/projects/x-<allocation>/specfem3d-4.1.1_wb`.

---

## 1. The `sm_120` problem, and the fix

SPECFEM3D 4.1.1 hardcodes a GPU architecture per `--with-cuda` value:

| `--with-cuda=` | architecture | card |
|---|---|---|
| cuda9  | `sm_70` | V100 |
| cuda10 | `sm_75` | T4, RTX 2080 |
| cuda11 | `sm_80` | A100, RTX 3080 |
| **cuda12** | **`sm_90`** | **H100 — the highest available** |

`Makefile.in:218`:
```make
@COND_CUDA_TRUE@@COND_CUDA12_TRUE@GENCODE = $(GENCODE_90) $(FC_DEFINE)GPU_DEVICE_Hopper
```

An RTX 5090 is Blackwell, **`sm_120`**, which does not appear anywhere in the
tree. A stock `--with-cuda=cuda12` build therefore produces Hopper cubins. The
5090 cannot execute those natively; it falls back to JIT-compiling the embedded
`compute_90` PTX at every process start.

**The fix** (`1_build_specfem_gpu.sh`): `GENCODE` is a plain recursive
assignment in the generated `Makefile`, so a make command-line override wins and
propagates to both the compile rule (`NVCC_FLAGS`) and the device-link rule
(`NVCCLINK`):

```bash
./configure --with-cuda=cuda12 CUDA_INC=... CUDA_LIB=... FC=gfortran CC=gcc MPIFC=mpif90 MPICC=mpicc
make GENCODE='-gencode=arch=compute_120,code="sm_120,compute_120"' -j$(nproc)
```

Overriding both rules together matters: a mismatch between compile and device-link
architectures surfaces as missing-symbol link errors, not as a clear message.

Losing `-DGPU_DEVICE_Hopper` in the override is harmless. Its entire body in
`src/gpu/mesh_constants_cuda.h:102-107` is `#undef USE_LAUNCH_BOUNDS`, and
`USE_LAUNCH_BOUNDS` is commented out by default anyway (`:58`).

The build script verifies the result with `cuobjdump`, which lists the GPU code
images actually embedded in the binary. If it does not report `sm_120`, the
override did not take.

### Two things that will break the build

**Do not run `autoreconf` / `autogen.sh`.** This tree's `m4/` is **empty** — the
`CIT_CUDA_CONFIG` macro exists only pre-expanded inside the shipped `./configure`
(around line 9235). Regenerating leaves it undefined.

**Do not pass `--with-cuda=cuda13`** or any unrecognised value. Configure still
sets `COND_CUDA` true but leaves every `COND_CUDAn` *and* `COND_CUDA_PLUS` false,
which silently: falls back to `GENCODE_20`, drops `-dc` from the compile line,
and leaves `NVCCLINK_BASE` **undefined**. Use `cuda12` and override `GENCODE`.

### CUDA version

`sm_120` requires **CUDA toolkit ≥ 12.8**. `0_preflight.sh` checks the version
*and* proves it by compiling a trivial kernel with `-arch=sm_120`. It also checks
that `nvcc` accepts your host `gcc` — nvcc hard-errors on a compiler newer than it
supports (CUDA 12.8 wants gcc ≤ 14).

---

## 2. Par_file settings the GPU path rejects

Hard `stop`s in `src/specfem3D/initialize_simulation.F90:565-578` and
`src/shared/read_parameter_file.F90`:

| Setting | Required | Source |
|---|---|---|
| `GPU_MODE` | `.true.` | the only GPU flag in Cartesian's Par_file |
| `SAVE_MOHO_MESH` | `.false.` | `initialize_simulation.F90:572` |
| `USE_LDDRK` | `.false.` | `read_parameter_file.F90:943` |
| `LTS_MODE` | `.false.` | `read_parameter_file.F90:1027` |
| `PML_CONDITIONS` | `.false.` | use `STACEY_ABSORBING_CONDITIONS` instead |
| `CUSTOM_REAL` | `4` | do **not** `--enable-double-precision` |
| `N_SLS` | `3` if `ATTENUATION=.true.` | moot here, attenuation is off |

`lib/common.sh:assert_gpu_parfile` checks these before every launch, turning a
three-minute mystery crash into one line.

The ite3 Par_file was already compliant on every count except `GPU_MODE`; the only
substantive change this port makes is `GPU_MODE=.true.` and `NPROC=1`.

Note the reverse case: `GRAVITY` is *only* supported in GPU mode
(`initialize_simulation.F90:356`). Not used here.

**Noise tomography is broken on GPU for kernel runs**
(`compute_add_sources_viscoelastic.F90:825`: `stop 'for NOISE simulations,
backward GPU routine is not implemented yet'`). This workflow sets
`NOISE_TOMOGRAPHY = 0` and uses `FORCESOLUTION` virtual sources, so it is
unaffected — but do not turn that flag on.

---

## 3. What is GPU-accelerated

Confirmed present in `src/gpu/` (26 `.cu` files + 41 kernels):

- forward elastic, `SIMULATION_TYPE=1`, with `SAVE_FORWARD`
- adjoint/kernel, `SIMULATION_TYPE=3`, including `APPROXIMATE_HESS_KL`
- `STACEY_ABSORBING_CONDITIONS`
- `MODEL=gll`, `NGNOD=8`, `ANISOTROPY=.false.`, `ATTENUATION=.false.`

**Post-processing is CPU-only with one exception.** Only three binaries link the
GPU objects: `xspecfem3D`, `xinverse_problem_for_model`, and **`xsmooth_sem`**.

`xsmooth_sem` takes GPU mode as a **6th positional argument**:

```
xsmooth_sem SIGMA_H SIGMA_V KERNEL_NAME INPUT_DIR OUTPUT_DIR [GPU_MODE]
```

This is a real win — smoothing was a 40-minute 10-node MPI job on Anvil. Caveat
(`smooth_sem.F90:280`): the GPU path uses a **brute-force** neighbour search
rather than the CPU kd-tree, so it is compute-heavy and memory-hungry. If it
OOMs, set `SMOOTH_GPU=0` in `config.sh`.

`xclip_sem`, `xsum_preconditioned_kernels`, and `xadd_model_iso` are CPU-only —
and this workflow does not use them anyway; ite3 replaced them with
`clip_iso.py` / `sum_precond.py` / `lbfgs_update.py`, which are carried over
verbatim.

---

## 4. `NPROC=1` is supported

`src/meshfem3D/meshfem3D.F90:320`:
```fortran
if (NPROC_XI < 1) call exit_MPI(myrank,'NPROC_XI must be greater than 1')
```
The **message is wrong** — the test is `< 1`, so `NPROC_XI = NPROC_ETA = 1`
passes. `read_mesh_parameter_file.f90:448` in fact *requires* it for a serial run.

All MPI-assembly code is guarded (`save_databases.F90:409` etc.).
`xgenerate_databases`' only constraint is that the rank count matches Par_file
`NPROC`.

### Why Anvil's mesh and models cannot be reused

`meshdata/proc000000_Database … proc001249_Database` is not a file-format detail,
it is a **domain decomposition** — the mesh is physically cut into 1250 pieces.
One GPU runs one rank and needs one piece containing the whole domain, and there
is no tool that merges partitions. The same applies to `M0` (`proc*_{vp,vs,rho}.bin`)
and to the L-BFGS state (concatenated per-rank vectors).

So `2_make_mesh.sh` and `3_make_M0.sh` rebuild both locally from the same
`tomography_model.xyz`. The resulting M0 is the same physical field, sampled onto
a one-piece mesh. The L-BFGS state starts fresh, meaning the first update is pure
steepest descent — exactly as ite3's first update was.

---

## 5. Memory and disk at this mesh size

Mesh: `75 × 150 × 18 = 202,500` spectral elements, ≈ 13.2 M unique global points.

**GPU memory**, single precision:

| array group | size |
|---|---|
| displ/veloc/accel (×3 components) | ~475 MB |
| backward field (`SIMULATION_TYPE=3`) | ~475 MB |
| Jacobian + `xix…gammaz` (9 arrays) | ~910 MB |
| kappa/mu/rho stores | ~300 MB |
| `ibool` | ~100 MB |
| kernels (4–5 arrays, type 3 only) | ~500 MB |
| `epsilondev` (fwd + bwd) | ~1000 MB |

Forward ≈ 2–3 GB, kernel run ≈ 4–6 GB. On a 32 GB 5090 there is a lot of room —
this mesh is not memory-limited. `4_bench.sh` measures the real peak.

**Disk — the number that actually constrains the design.** With
`STACEY_ABSORBING_CONDITIONS` and `UNDO_ATTENUATION_AND_OR_PML=.false.`, a
`SAVE_FORWARD` run stores boundary tractions for **every timestep**:

```
faces = 2·(75·18) + 2·(150·18) + 75·150 = 19,350
bytes = 19,350 × 25 GLL × 3 × 4 B × 5000 steps ≈ 29 GB per event
```

Measured on Anvil: `du -sh FRAME/3A.20050_Z` → **28 GB**. For 99 events that is
2.8 TB, and ite3's `FRAME/` currently holds 627 events ≈ **18 TB**.

That is why this port decouples the stages: the per-iteration forward runs with
`SAVE_FORWARD=.false.` (seismograms only, ~250 MB/event), and the adjoint stage
runs forward-then-kernel back-to-back per event with the wavefield transient in
`DATABASES_MPI`, deleted immediately after. Peak disk ≈ 30 GB instead of 2.8 TB,
and the 5.6 TB of per-iteration copying disappears. Cost: one extra forward per
event, which on a GPU is minutes.

---

## 6. Filesystem-dependent completion checks — a portability trap

ite3 decides an event is finished with `du -sm <dir> >= 300`. On Anvil's GPFS
(512 KB allocation unit) each 165 KB `.semd` occupies 512 KB, so a finished event
measures **381 MB** while containing only **120 MB** of data.

On a local ext4/xfs the same event measures ~120 MB — **below the threshold**. The
consequences would be silent and severe: every event re-runs forever, `--finalize`
never fires, `--resume` resubmits everything.

This port uses explicit sentinels (`.event_done`, `.kernel_done`) plus a file
count. Deterministic and filesystem-independent.

The same class of bug is live in ite3's adjoint: `run_forward_n_adjoint_anvil.slurm`
skips an event when `du -h KERNEL | grep -c '[3,4]\.[0-9]G'` matches. That was
calibrated on ite2's 60-layer mesh (`NSPEC=540`, kernels ≈ 3.4 GB). ite3's
18-layer mesh gives `NSPEC=162` and kernels ≈ **1.0 GB**, so the pattern **never
matches** and every resubmit recomputes finished events.

---

## 7. The band-code bug (affects Anvil ite3 too)

SPECFEM derives the seismogram channel band code from `DT`
(`write_seismograms.f90:band_instrument_code`):

| DT range | code |
|---|---|
| `> 0.1` | `MX` |
| `0.0125 < DT ≤ 0.1` | **`BX`** |
| `0.004 < DT ≤ 0.0125` | **`HX`** |
| `≤ 0.004` | `CX` / `FX` |

ite2 ran `DT=0.005` → `*.HXZ.semd`. ite3 runs `DT=0.02` → `*.BXZ.semd`.

`cal_adjoint/1_process_data.py:69` hardcoded:
```python
syn_candidates = list(syndir.glob(f"{net_sta}.{sta}.HXZ.semd"))
```

Verified on disk: `forward/result_M0_p1/3A.20050_Z/` contains **756 `*.BXZ.semd`**
files and zero `HXZ`. So on ite3 this glob matches nothing, every pair logs
"Synthetic file not found", no windows are measured, no `.adj` are written — and
the job still **exits 0**, reporting `NA 0` as the misfit.

That is the worst possible failure mode for an automated loop: `run_loop.sh`
reads it as success.

This port fixes it two ways:
1. the glob accepts any band code, and falls back to `*Z.semd`;
2. `run_caladj_local.sh` treats **zero measurements as a hard failure**, not a
   misfit of zero.

**This bug is present in the Anvil ite3 tree right now** and would need the same
fix there before the loop can run.

---

## 8. Other live issues found in ite3 (Anvil-side, FYI)

- **`forward/result` is a dangling symlink** → `result_M0`, which does not exist.
  Only `result_M0_p1/` (536 events) and `result_M0_p2/` (91) do. `cal_adjoint`'s
  `DATADIR` points at that symlink, so it currently resolves to nothing, and
  `run_loop.sh` would `die` with "forward(M0) appears dead".
- **`cal_adjoint/config.py` `WINDIR` = `windows/ite3_r1` does not exist.** With
  `--reuse` that stages zero windows → the `NA 0` failure above. The first run
  must pick windows. This port detects an empty cache and picks automatically.
- **Do not point `WINDIR` at `windows/r2`** — those are ite2's picks, made against
  a model whose crust was clamped to mantle velocity. The arrival times do not
  transfer.

## Seismogram output format (2026-09-11)

`config.sh: SEISMO_FORMAT=ascii|bin` -> Par_file `USE_BINARY_FOR_SEISMOGRAMS`.
Writing 932 ascii traces is CPU-side formatted Fortran I/O and cost 12.8-17.2 s
of every 57-73 s forward on the 5090 (`output_solver.txt`: "Writing the
seismograms in parallel took"). With `bin` the same event wrote in 0.24 s
(solver total 73.0 -> 49.5 s; 217 -> 27 MB per event). Binary traces are the
bare float32 `tr(:)` array (write_output_ASCII_or_binary.f90), no time column;
the runner writes `seismo_meta.txt` (format, nstep, dt, start_time, band) next
to them, `start_time` parsed from the solver's own "start time:" line
(= -1.5 x half duration, -0.10875 s for this DT). Readers:
`cal_adjoint/utility.py:read_semd`, `workflows/grace_hopper/validate_forward.py`,
`eq_validation/compare_eq.py` (not included). Validation: traces identical to the ascii
reference within the run-to-run floor, misfit identical. The GH200 template
enables `bin`; the x86 campaign default stays `ascii`.

**Kernel-leg restriction.** `compute_add_sources_viscoelastic.F90:206` stops
any SIMULATION_TYPE=3 run that has `USE_BINARY_FOR_SEISMOGRAMS = .true.`
("Adjoint simulations not supported with .bin format, please use SU format
instead") -- the flag doubles as the adjoint-source *input* format selector
there. The workflow therefore sets it per run (`lib/common.sh:
seismo_format_par`): on for the two forward legs under SEISMO_FORMAT=bin, off
for the kernel leg, whose seismogram output is disabled anyway
(SAVE_SEISMOGRAMS_IN_ADJOINT_RUN=.false.). Validated 2026-09-11: the production
`adjoint/run_adjoint.sh --limit 1` under SEISMO_FORMAT=bin ran forward+save
and the kernel leg cleanly. The same applies to ASDF_FORMAT:
use it on forward runs only; the adjoint input side has its own switch
(READ_ADJSRC_ASDF, see doc/ASDF_NOTES.md).

