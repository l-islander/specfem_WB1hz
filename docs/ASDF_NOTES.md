# ASDF (Adaptable Seismic Data Format) for SPECFEM3D -- investigation notes

> 2026-09-11. Question: is SPECFEM's ASDF support worth adopting for the
> rack-scale runs, and is it available on Arm?
> Short answer: **yes, nothing in the chain is architecture-specific**, and the
> whole dependency stack ships for linux-aarch64 from conda-forge. The catch is
> that neither our x86 build nor the stock 4.1.1 configure does anything for
> ASDF out of the box; it is a rebuild plus a Python rewrite of two stages.
>
> **Update, same day -- SPECFEM side tried on the 5090 (section 6): both
> directions work, but the adjoint-input path needed two source fixes
> (`patches/`). cal_adjoint is unchanged so far; that is the next step.**

## 1. What SPECFEM3D 4.1.1 offers

| direction | Par_file switch | what it does | source |
|---|---|---|---|
| seismogram **output** | `ASDF_FORMAT = .true.` | one `OUTPUT_FILES/synthetic.h5` per run: all traces + QuakeML + StationXML + provenance | `src/specfem3D/write_output_ASDF.f90` |
| adjoint-source **input** | `READ_ADJSRC_ASDF = .true.` (SIMULATION_TYPE 2/3) | reads `AuxiliaryData/AdjointSources/<net>_<sta>_<comp>` from an ASDF file instead of `SEM/*.adj` | `src/specfem3D/read_adjoint_sources_ASDF.f90` |

Constraints found in the source:
- Needs `./configure --with-asdf ASDF_LIBS="..."`. Configure does **no**
  detection; it only sets `COND_ASDF` and appends to the link line
  (`Makefile.in:413`): `-lasdf -lhdf5hl_fortran -lhdf5_hl -lhdf5 -lstdc++`.
  Without it both switches stop at startup with
  "Not configured to compile or run with ASDF." (`src/shared/asdf_manager_stubs.f90`).
- `ASDF_FORMAT` and `SU_FORMAT` are mutually exclusive; ASDF output allows
  only ONE saved type (we save displacement only -- fine).
- The check "ASDF_FORMAT requires WRITE_SEISMOGRAMS_BY_MAIN" is commented out
  in `read_parameter_file.F90:990` -- i.e. less exercised than the plain
  binary path we validated today. Treat as needing its own validation run.
- STATIONS_ADJOINT still selects the adjoint stations; a listed station
  missing from the ASDF file is a hard `exit_MPI`.
- GPU_MODE is orthogonal: adjoint sources are read on the host and
  transferred exactly as with `.adj` files.

## 2. The dependency chain and ARM status

| component | what it is | aarch64 status (checked 2026-09-11) |
|---|---|---|
| **asdf-library** (SeismicData/asdf-library) | ~2,500 lines of plain C + Fortran-binding shim (`ASDF_f.c`), CMake; last commit 2020-06 | no x86 intrinsics, no endian/asm code (grepped). Builds anywhere HDF5+MPI build. |
| **parallel HDF5** (>= 1.8, with Fortran bindings) | asdf-library + SPECFEM link dep | conda-forge ships `hdf5 2.2.0 mpi_openmpi_*` and `mpi_mpich_*` for **linux-aarch64**, with `libhdf5_fortran`/`libhdf5_hl_fortran`. Any GH200 site stack will also have one. |
| **Boost headers** (uuid, for provenance ids) | asdf-library build dep, header-only | `libboost-headers` on conda-forge, noarch-equivalent |
| **pyasdf** 0.8.2 (Aug 2025) | Python side: numpy, obspy >= 1.2.1, h5py, colorama, prov, dill; mpi4py optional | pure Python (conda-forge `noarch`) |
| **h5py** 3.16 | pyasdf dep | manylinux_2_28 **aarch64** wheels for CPython 3.10-3.14 on PyPI |

## 3. Dry-run build on the x86 box (proves the recipe; replay verbatim on ARM)

```bash
conda create -p ./asdfenv -c conda-forge --override-channels \
    cmake "hdf5=*=mpi_openmpi*" openmpi c-compiler cxx-compiler make libboost-headers
conda activate ./asdfenv
git clone --depth 1 https://github.com/SeismicData/asdf-library && cd asdf-library
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j8                       # -> lib/libasdf.a, examples/write_ASDF
mpirun -np 2 examples/write_ASDF && h5dump -n examples/synthetic.h5
```

Result: `libasdf.a` built (HDF5 2.2.0 parallel, OpenMPI 5.0.10, gcc 15.3),
the C MPI example wrote a valid `synthetic.h5` (Waveforms/, AuxiliaryData/,
Provenance/, QuakeML). Two gotchas, neither ARM-related:

1. **CMake >= 4** refuses the library's `cmake_minimum_required(VERSION 2.8.7)`;
   pass `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
2. The library's **Fortran example** fails on gfortran >= 10 with a kind
   mismatch (INTEGER(8)/INTEGER(4)) inside the example itself; the library
   and SPECFEM's interface go through `ASDF_f.c` (iso_c_binding) and are not
   affected. `-fallow-argument-mismatch` compiles the example if wanted.
3. **Library naming**: SPECFEM's `Makefile.in` links `-lhdf5hl_fortran`
   (Debian/Ubuntu spelling); conda-forge and most HPC builds name it
   `libhdf5_hl_fortran`. Either symlink it or override via `ASDF_LIBS`.
   (Kept as a note; not yet exercised since SPECFEM was not rebuilt with ASDF.)


## 4. What adopting it would take

| step | effort | notes |
|---|---|---|
| asdf-library + parallel HDF5 on the target (GH200 now, Vera Rubin later) | small | conda-forge recipe above, or the site's `hdf5-parallel` module + the same cmake |
| rebuild SPECFEM `--with-asdf ASDF_LIBS="-L... -lasdf"` | small | plus the `-lhdf5hl_fortran` naming fix; then re-run the forward validation for the ASDF output path |
| cal_adjoint stage 1 reader (`.semd`/binary -> per-station SAC) | medium | read `synthetic.h5` via pyasdf, keep writing SAC downstream; `read_semd` stays for legacy dirs |
| cal_adjoint stage 4 writer (`SEM/*.adj`) | medium | write `AuxiliaryData/AdjointSources/<net>_<sta>_<comp>` via pyasdf; keep STATIONS_ADJOINT |
| runner (`lib/solver.sh`) | small | harvest `synthetic.h5` instead of `*Z.sem*`; completion sentinel = file present + trace count from h5 |
| validation | one forward + one adjoint event | ASDF traces vs ascii reference with `workflows/grace_hopper/validate_forward.py` (needs an h5 branch); misfit equality as done for the binary switch |

What it buys over the binary-trace switch already in place: one file per
event on both sides of the loop (932 -> 1 for seismograms; the `.adj` ASCII
read in the adjoint leg is the last formatted-I/O cost), self-describing
metadata for long-term archives, and a format the community tooling
(pyasdf/ObsPy) reads directly. What it costs: a build dependency on every
target and a Python rewrite of two stages.

## 5. Sources

- asdf-library: https://github.com/SeismicData/asdf-library (cloned, grepped, built)
- pyasdf setup.py install_requires: https://github.com/SeismicData/pyasdf
- conda-forge hdf5 files (linux-aarch64 mpi variants): https://anaconda.org/conda-forge/hdf5/files
- h5py 3.16 aarch64 wheels: https://pypi.org/project/h5py/#files
- pyasdf on conda-forge (noarch, 0.8.2): https://anaconda.org/conda-forge/pyasdf

## 6. SPECFEM-side trial on the RTX 5090 (2026-09-11)

Everything lives beside the production tree and does not touch it:
`ext/hdf5` (HDF5 1.14.6, `--enable-parallel --enable-fortran`, built against
the SYSTEM OpenMPI 4.1.2 -- the conda-forge HDF5 is linked to conda's OpenMPI 5
and cannot be mixed with our gfortran/nvcc build), `ext/asdf` (libasdf.a built
against it; Boost headers borrowed from the conda env), `specfem3d_asdf/`
(source copy, 52 MB), `bin_asdf/`, `patches/build_specfem_asdf.sh` (same configure
as production + `--with-asdf ASDF_LIBS=...`; builds the workflow's targets only,
the bundled Scotch has no Makefile in a fresh copy), `work/asdf_test/` (solver
workdir on the M6-baked mesh, event 3A.20050_Z).

### 6.1 Output side -- `ASDF_FORMAT = .true.` : works out of the box

| item | result |
|---|---|
| run | forward, GPU_MODE, NSTEP 6900; exit 0; solver total 56.5 s |
| file | `OUTPUT_FILES/synthetic.h5`, 82 MB: 2,796 traces (932 stations x BXE/BXN/BXZ, float32) + one StationXML per station + QuakeML; no AuxiliaryData group (ASDF_OUTPUT_PROVENANCE is off in constants.h) |
| write time | "Writing the seismograms in parallel took 0.60 s" (ascii: 17.2 s, bin: 0.24 s for Z only vs all 3 components here) |
| trace names | `Waveforms/3A.20051/3A.20051.S3.BXZ__1999-12-31T00:59:59__...__synthetic` (location code "S3", placeholder origin 2000-01-01 minus t0 since FORCESOLUTION has no date) |
| validation | `workflows/grace_hopper/asdf_to_semd.py` unpacked Z into binary traces + sidecar (start time recovered from the file's own QuakeML origin = -0.10875 s); `validate_forward.py` vs the ascii reference: **max rel. amp 7.7e-6, lags 0, cc 1.0 -> PASS** (run-to-run floor 8.3e-6) |
| run-dir needs | `setup/constants.h` and `DATA/Par_file` are read from the CURRENT DIRECTORY at write time (stored in the file) -- the runner must stage `setup/` into each workdir |
| caveats | ASDF output is forward-only in practice: SIMULATION_TYPE=3 must have it off (same rule as USE_BINARY, section 6.3); `SAVE_ALL_SEISMOS_IN_ONE_FILE` irrelevant; pyasdf 0.8.2 warns once on close (harmless) |

### 6.2 Input side -- `READ_ADJSRC_ASDF = .true.` : works only with two source fixes

Test: `tools/adj_to_asdf.py` converted the event's 39 `SEM/*.adj` (13 stations x
3 comps, from a real cal_adjoint `--full` run on s2_iter6 windows) into
`SEM/adjoint.h5` (`AuxiliaryData/AdjointSources/NET_STA_BXZ`, float32, 6900
samples; values equal to the ascii column). Then, on the same saved forward
wavefield: kernel run fed from ascii `.adj` twice (run-to-run floor) and from
`adjoint.h5` once; compare `proc000000_*_kernel.bin`.

| run | result |
|---|---|
| stock 4.1.1, GPU mode | aborts: "Adjoint source 3A.20111_BXE has wrong length" -- the GPU routine builds `NET.STA` (dot) while the ASDF existence check and the CPU routine use `NET_STA`; the read then asks HDF5 for a name that does not exist |
| + fix 1 (name) | runs to completion, exit 0, **all seven kernels exactly zero**: the ASDF branch of `compute_arrays_adjoint_source` copies the samples into a local array that nothing reads (the ascii branch writes `source_adjoint` directly) |
| + fix 2 (store into `source_adjoint`) | **kernels match the ascii-fed run: 4.6e-5 (beta) .. 2.5e-4 (hess) relative; ascii-vs-ascii run-to-run floor 5.3e-5 .. 1.7e-4** -> identical within GPU accumulation noise |

Both fixes are in `patches/specfem3d-4.1.1_asdf_adjsrc.patch` (applies
cleanly with `patch -p0` from a stock 4.1.1 source root; notes in
`patches/README.md`). They are bugs in SPECFEM3D 4.1.1's ASDF adjoint
path as shipped; worth checking against `devel` and reporting upstream.
Kernel-run cost with ASDF input: 161 s vs 162 s with ascii (the `.adj` read is
not a measurable cost at 13 stations; it would be at 700+).

### 6.3 Side finding: USE_BINARY_FOR_SEISMOGRAMS must be off in kernel runs

`compute_add_sources_viscoelastic.F90:206/625` stops any SIMULATION_TYPE=3 run
with `USE_BINARY_FOR_SEISMOGRAMS = .true.` ("Adjoint simulations not supported
with .bin format"). The workflow's SEISMO_FORMAT=bin switch (added earlier
today) set the flag per workdir and would have broken the kernel leg; it now
sets it per run (`lib/common.sh:seismo_format_par`: forward legs on, kernel
leg off). Production check: `SEISMO_FORMAT=bin adjoint/run_adjoint.sh --model M6
--limit 1` -> genmesh 20m36s, forward+save 1m27s (binary traces), kernels OK
(7 files, 2m59s), ADJOINT DONE; the kernel run's Par_file had the flag off.

### 6.4 Status / next

- [x] parallel HDF5 + asdf-library built against the system MPI (recipe: `patches/build_specfem_asdf.sh` header; HDF5 configure line in this section)
- [x] SPECFEM `--with-asdf` build, GPU, sm_120 (`bin_asdf/`), production `bin/` untouched
- [x] ASDF seismogram output validated vs ascii reference
- [x] ASDF adjoint-source input validated vs ascii-fed kernels (with the 2 patches)
- [x] kernel leg under SEISMO_FORMAT=bin through the production runner (2026-09-11, event 3A.20050_Z)
- [ ] cal_adjoint: stage 1 reads `synthetic.h5` (pyasdf) instead of per-station `.semd`; stage 4 writes `SEM/adjoint.h5` (the `tools/adj_to_asdf.py` logic) -- next step
- [ ] runner: harvest `synthetic.h5`, stage `setup/constants.h`, sentinel on trace count in the h5
- [ ] GH200: rebuild `ext/hdf5` + `ext/asdf` + `specfem3d_asdf` with the site MPI; same patch
