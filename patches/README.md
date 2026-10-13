# Changes to SPECFEM3D Cartesian 4.1.1

The workflow uses the upstream SPECFEM3D Cartesian 4.1.1 source
(https://github.com/SPECFEM/specfem3d) without modification for every production run. The items
below are build-time settings and fixes made while porting and scaling it. Patches are derived from
GPL-3.0 code and carry SPECFEM3D's license.

| change | where | status |
|---|---|---|
| **GENCODE override for new GPUs.** 4.1.1 ships code generation up to sm_90 (Hopper). The RTX 5090 (sm_120) needs `make GENCODE=-gencode=arch=compute_120,code=[sm_120,compute_120]` (bracket form: the stock quoted form is mangled by the shell when passed on the make line). Vera Rubin will need the same one-line override for its compute capability. | `workflows/rtx5090/1_build_specfem_gpu.sh` | in production (RTX 5090); not needed on GH200 |
| **CUDA 13: removed `deviceOverlap` property.** `src/gpu/initialize_gpu.cu` prints `deviceProp.deviceOverlap`, which CUDA 13 removed; a one-line change is needed to build with Rubin's toolkit. | to do (Q1) | planned |
| **Quadratic adjacency setup in mesh-database generation.** `setup_mesh_adjacency.f90` resets two full arrays inside the per-element loop, making it O(NSPEC x NGLOB): 13 of 16 min per model at 2 s, a projected >20 h at 1 Hz. A 12-line fix (clear only the entries that were set) gives a bitwise-identical database in 98 s (2 s) and 44 min (1 Hz). | patch not yet in this repository (to be added with the upstream report) | validated on GH200 |
| **ASDF adjoint-source input in GPU mode** (`specfem3d-4.1.1_asdf_adjsrc.patch`): (1) the GPU routine built the adjoint-source name as `NET.STA` while the ASDF reader and the CPU routine use `NET_STA`, so kernel runs aborted; (2) the ASDF branch copied the samples into a local array that nothing reads, so ASDF adjoint sources were silently zero (all kernels exactly 0, exit code 0). With both fixes, ASDF-fed kernels match ASCII-fed kernels within the run-to-run floor. | this folder; `build_specfem_asdf.sh` builds the ASDF variant against parallel HDF5 + asdf-library | validated on RTX 5090 |
| **Binary seismogram output** for rack-scale I/O (SPECFEM's `USE_BINARY_FOR_SEISMOGRAMS`, applied per run: SPECFEM refuses binary output in adjoint runs). Per-source writes 17 s -> 0.2 s, identical misfit. | `workflows/rtx5090/lib/common.sh` (`seismo_format_par`), `cal_adjoint/utility.py` (`read_semd`) | available option (default ascii); on in the GH200 machine file |

Apply the ASDF patch from the SPECFEM source root:

```bash
patch -p0 < /path/to/specfem_WB1hz/patches/specfem3d-4.1.1_asdf_adjsrc.patch
```

Not patched (CPU path, not used here): the CPU routine's `READ_ADJSRC_ASDF` branch calls
`compute_arrays_adjoint_source` once more after its receiver loop with
`irec_local = nadj_rec_local + 1`. Both ASDF fixes are candidates for an upstream report.
