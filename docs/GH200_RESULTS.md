# Grace Hopper (GH200) port: measured results

Measured September 30 – October 7, 2026 on an OSU research-computing node with one NVIDIA GH200 480GB
superchip (72 Arm Neoverse-V2 cores, 477 GiB LPDDR5X, Hopper GPU with 95 GiB HBM3, Rocky Linux 9.6
aarch64 with a 64 KiB-page kernel). Reference machine: an x86 workstation with one RTX 5090 (32 GB,
sm_120) under Ubuntu 22.04/WSL2. Tools: `workflows/grace_hopper/validate_forward.py` and
`workflows/grace_hopper/time_forward.sh`. All runs were launched directly on the node, not through the
batch scheduler. Some timing runs used a stand-in 1D model on the production mesh; solver cost does not
depend on model values (26.29 s on both).

## 1. Build (no source changes)

| item | x86 / RTX 5090 | aarch64 / GH200 |
|---|---|---|
| compilers / MPI / CUDA | gfortran 11.4, Open MPI 4.1.2, CUDA 12.8 | gfortran 11.5, Open MPI 4.1.7 (HPC-X 2.23), CUDA 12.9 (NVHPC 25.7) |
| SPECFEM3D 4.1.1 configure | `--with-cuda=cuda12` + GENCODE override to sm_120 | `--with-cuda=cuda12`, stock sm_90 GENCODE |
| source changes | none | none |
| build time | — | 28 s (`-j 32`) |
| mesher output | reference | byte-for-byte identical (`cmp`) |
| GLL model files written on x86 | read directly | read directly, no conversion |
| Python stages (ObsPy, mpi4py, pyflex/pyadjoint) | uv venv | uv venv, CPython 3.10; obspy and mpi4py built from source; 2-rank and 72-rank MPI tests pass |

Site-specific environment only (in `env.sh`): explicit `CUDA_HOME`, `module load hpcx`, Open MPI
transport flags for the 64 KiB-page kernel (`--mca pml ob1 --mca osc ^ucx --mca coll ^hcoll,ucc`;
default HPC-X settings hang at 72 ranks), node-local `WORKBASE`, and a Python 3.10 pin for the vendored
pyflex. Note: the bundled Scotch sources must not carry x86 build products (`make realclean` first).

## 2. Forward validation against the x86 reference (final model, three virtual sources, 932 traces each; the GH200 rerun row used a stand-in 1D model, the control row source Z4.16063)

| comparison | max relative amplitude difference | zero-lag cc (min) | traces with non-zero lag |
|---|---|---|---|
| x86 rerun vs x86 reference (same card, fresh solve) | 8.3e-6 | 1.0000000 | 0 |
| GH200 rerun vs GH200 (same card) | 1.7e-6 | 1.0000000 | 0 |
| GH200 vs x86, source 3A.20050 | 7.2e-6 | 1.0000000 | 0 |
| GH200 vs x86, source 3A.20052 | 7.0e-6 | 1.0000000 | 0 |
| GH200 vs x86, source 3A.20055 | 7.1e-6 | 1.0000000 | 0 |
| control: two different models (M0 vs M6) | 2.6 | -0.70 | 898 |

The GH200 differs from the x86 reference by less than the x86 card differs from itself on a rerun
(GPU reductions are not bitwise reproducible); the difference is white round-off.

## 3. Cost (production 2 s mesh: 540,800 elements, NSTEP 6900)

| metric | GH200 (binary seismograms) | RTX 5090 (ascii, 150-source campaign mean) |
|---|---|---|
| solver time loop per forward | 26.2 s (3.79 ms/step) | 45.1 s (6.54 ms/step) |
| wall per source incl. startup | 45 s | ~75 s |
| seismogram write (932 traces) | 0.12 s (binary) / 14 s (ascii) | 17.2 s (ascii) / 0.24 s (binary) |
| adjoint simulation per source (forward+save, then kernel) | 62 s + 132 s = 194 s wall (timing run with synthetic adjoint sources); kernel solver = 3.8x a forward | — |
| full iteration (150 forwards + 150 adjoints + 3 x 40 line-search forwards) | 11.5 GPU-hours summed from measured per-source legs (10.9 ratio-scaled); no full iteration was run on the GH200 | 18.2 GPU-hours (measured) |
| 150-source forward pass | 1 h 52 min, 0.3 % spread, no failures | — |
| GPU memory, forward / kernel | 6.1 GB / 12.7 GB | 5.4 GB / 13.7 GB |
| GPU duty cycle during forwards | 57–59 % (startup-bound) | ~73 % (ascii-write-bound) |

**1 Hz mesh (4.33 M elements, DT 0.007 s, 14,300 steps):** one forward 425.8 s solver (7.1 min),
43.2 GB GPU memory, 90 % utilization: 16x the 2 s forward, as expected from 8x elements x 2x steps.
Two concurrent 1 Hz forwards peak at 86.5 GB; the 1 Hz kernel leg (projected ~95–100 GB; not run) would not fit one GH200.

**Concurrency on one GPU:** 1/2/4/8 simultaneous 2 s forwards give an aggregate 1.0/1.09/1.16/1.20x
(the time loop already saturates the GPU); MPS adds 3–4 %. One simulation per GPU is the right
granularity, and more GPUs, not more processes per GPU, are what scale the campaign.

**Mesh-database generation:** 16 min per model on Grace, of which 13 min is a quadratic adjacency setup
in SPECFEM3D 4.1.1; a 12-line fix produces a bitwise-identical database in 98 s (1 Hz mesh: 44 min
instead of a projected >20 h).

**I/O:** per-source startup (~20 s at 2 s, ~45 s at 1 Hz) is dominated by reading the mesh database
(5.4 GB at 2 s, ~46 GB at 1 Hz). The adjoint's boundary wavefield is 78 GB per source at 2 s (~620 GB projected
at 1 Hz), written and re-read within minutes; a page-faulting CPU job on the same superchip slowed the
GPU time loop 1.3–1.8x, a pure memory-bandwidth load did not.

## 4. What this means for Vera Rubin NVL72

- Same CPU–GPU lineage (Arm CPU + NVIDIA GPU over NVLink-C2C). The port to Rubin is a new
  compute-capability target: the same one-line GENCODE override used for the RTX 5090, CUDA 13, and, from a
  source check (not yet built with CUDA 13), one removed device-property field (`deviceOverlap`) to patch.
- Events are independent: the rack will run the same NPROC=1 binaries, one source per GPU, at 1.5 s; at
  1 Hz the adjoint is split over a few GPUs or keeps its wavefield in host memory (see
  `multi_gpu/README.md`).
- The measured 1.7x over the RTX 5090 on 2.2x the memory bandwidth (~75 % yield) is the basis for the
  4–8x per-GPU planning factor for Rubin (19.2 TB/s).
