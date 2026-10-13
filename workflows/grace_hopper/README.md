# Grace Hopper (GH200) workflow

The Grace Hopper workflow **is the RTX 5090 workflow** (`../rtx5090/`) with a different machine file.
Nothing in the science code, the orchestration loop or SPECFEM3D's source changed: the port consisted
of an `env.sh` for an aarch64 + Hopper node, a guard fix in `1_build_specfem_gpu.sh` so that
`GPU_ARCH=90` is accepted, and two measurement harnesses used to validate and time the port. Measured
results: `../../docs/GH200_RESULTS.md`.

| file | purpose |
|---|---|
| `env.sh.gh200` | machine file for an aarch64 + Hopper node: `GPU_ARCH=90`, CUDA/MPI detection, Open MPI 4 vs 5 shared-memory transport name, `CALADJ_NPROC` from the core count, binary seismograms on |
| `validate_forward.py` | compares one source's seismograms with a reference run (ascii or binary, mixed allowed): max relative amplitude difference, relative L2, zero-lag correlation, best lag; PASS/WARN/FAIL bands |
| `time_forward.sh` | runs N sources through the production forward runner with a 1 s GPU utilization/memory sampler and prints solver time loop, total, per-step time, wall per source, throughput, duty cycle and the projected iteration cost next to the RTX 5090 column |
| `gh200_forward.sbatch` | Slurm wrapper for `time_forward.sh` (one superchip, one GPU) |
| `asdf_to_semd.py` | unpacks a SPECFEM ASDF seismogram file (`synthetic.h5`) into per-trace binary files so the validator can compare ASDF output with the reference |

## Sequence (adapted to this repository layout)

```bash
cp workflows/grace_hopper/env.sh.gh200 workflows/rtx5090/env.sh     # then set CUDA_HOME, modules, WORKBASE
cd workflows/rtx5090
./0_preflight.sh                       # compute capability 9.0, nvcc accepts sm_90, MPI, Python
make -C specfem3d/external_libs/scotch_5.1.12b/src realclean   # only if the SPECFEM tree came from another architecture
./1_build_specfem_gpu.sh -j 32         # SPECFEM3D 4.1.1, stock Hopper code generation, 28 s
./2_make_mesh.sh                       # mesher output byte-identical to the x86 run
cal_adjoint/build_env_local.sh         # uv venv on aarch64, mpi4py built against the site MPI
../grace_hopper/time_forward.sh --n 3                      # forward the final model for 3 sources
python3 ../grace_hopper/validate_forward.py --ref <x86 result>/3A.20050_Z --test <GH200 result>/3A.20050_Z
../grace_hopper/time_forward.sh --n 40 --events events/lsauto_iter9 --tag ls40   # 40-source timing
```

Model files (GLL binaries), observed data and reference seismograms are not part of this repository.
