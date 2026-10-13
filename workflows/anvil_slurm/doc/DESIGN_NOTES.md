# anvil_cpu design notes

## Per-file map (what is new, what is copied)

| file | origin | note |
|---|---|---|
| `config.sh` | rewritten | sources `mesh_plan.sh`; SLURM shape; CPU smoothing; `CALADJ_BANDS`; `SEISMO_FORMAT=bin` |
| `env.sh.anvil` | new | modules, `MPIRUN=srun`, account/partitions, scratch paths |
| `env.sh.mock5090` | new | dry-run machine file for the 5090 box |
| `mesh_plan.sh` | generated | by `tools/plan_mesh.py --apply` |
| `lib/common.sh` | adapted | `gpu_run`→`mpi_run` (srun/mpirun), driver lock, `assert_cpu_parfile` |
| `lib/slurm.sh` | new | `sbatch_wait`, `sbatch_array`, `array_line`, `job_summary` |
| `lib/solver.sh` | rewritten | per-model DB dir + per-event workdir at NPROC; `run_forward_event`/`run_kernel_event` logic unchanged |
| `lib/genmesh_job.sh` | new | in-job `xgenerate_databases` |
| `forward/run_forward.sh` + `fwd_worker.sh` | rewritten | driver + array task |
| `adjoint/run_adjoint.sh` + `adj_worker.sh` | rewritten | driver + array task (fwd/save + kernel + drop) |
| `adjoint/run_post.sh` + `post_job.sh` + `smooth_job.sh` | rewritten | 1-node clip/sum job, NPROC-rank smooth job |
| `adjoint/run_update.sh` | adapted | self-submitting single-node job (`UPDATE_PARTITION`) |
| `adjoint/forward_update/linesearch.sh` | parent + 3 edits | sources slurm.sh, wipes `db/pert_*`, calls `run_caladj.sh` |
| `adjoint/{clip_iso,sum_precond,lbfgs_update}.py`, `adjoint/lib/*`, `polyfit.py` | **verbatim** | they always handled NPROC>1 |
| `cal_adjoint/run_caladj.sh` | parent + self-submit | ranks = `CALADJ_NODES × 128` capped at sources |
| `cal_adjoint/config.py` | parent + `BAND_TABLE`/`CALADJ_BANDS` | science parameters untouched |
| `cal_adjoint/{0..4}_*.py, utility.py, mpi_progress.py, pyflex/, pyadjoint/` | **verbatim** | |
| `cal_adjoint/build_env_anvil.sh` | adapted | srun 2-rank test |
| `run_loop.sh` | parent + 3 edits | preflight lines, script name |
| `0_preflight.sh`, `1_build_specfem_cpu.sh`, `2_make_mesh.sh`, `3_make_M0.sh`, `4_bench.sh`, `smoke_test.sh`, `status.sh` | rewritten | SLURM-shaped equivalents |
| `tools/plan_mesh.py` | new | mesh/decomposition/cost planner |
| `tools/band_snr_survey.py` | new | data gate for 1–1.5 s |
| `tools/mock_slurm/` | new | dry-run `sbatch`/`sacct` |
| `tools/select_ls_group.py`, `tools/gll_to_tomo.py` | verbatim | (gll_to_tomo is used on the PARENT tree by `stage_to_anvil.sh`) |
| `stage_to_anvil.sh` | new | bundle builder |
| `DATA/` | parent Par_file (NPROC/DT/NSTEP/GPU_MODE rewritten by the planner), STATIONS, FORCESOLUTION template, `interface2.dat`, `Mesh_Par_file.template` | |

## Why job arrays, not ite3's group jobs

ite3 pre-grouped events into `group_N/RUN/` and ran one 1250-rank job per
group with `du -sm` completion tests and InfiniBand watchdogs. A job array
gives the same concurrency with one line, per-event accounting in `sacct`,
per-event retry (re-invoking the driver resubmits only the events whose
sentinel is missing), and no filesystem-size heuristics. The `group_0/RUN`
directory name survives only because `clip_iso.py` globs it.

## Why one DB dir per model

`xgenerate_databases` bakes the GLL model into `proc*_external_mesh.bin`; the
solver reads only those files (plus `OUTPUT_FILES/values_from_mesher.h`). With
several events of the same model in flight, the mesh files are shared
read-only through symlinks and each event's `DATABASES_MPI` receives only its
own transient wavefield and kernels. Trial models of the line search get their
own DB dirs (`db/pert_<a>`), wiped by the stale-scratch guard together with
`MODEL_SCAN`.

## Sizes at NPROC=512 / 2048

Model = 3 × NPROC files (1.8 GB / 6.3 GB); mesh Database files 512 / 2048;
per-event kernels 4 × NPROC files (2.4 GB / 8.4 GB); every `cp`/`ln` of
`proc*` files goes through `find -print0 | xargs -0` so ARG_MAX never matters.

## Traps carried over from docs/GOTCHAS.md that still apply

1. Band code follows DT: both plans are HX. Never mix BX-era windows, `.adj`
   files or results into this stage (rc=30 "file has wrong length" kernel
   aborts if stale `work/adjoint_sources` dirs survive a basis change).
2. Basis-change reset checklist (windows/band/source-set change): archive
   `misfit_history.txt`, `rm .loop_marks/caladj.M<x>`, wipe `work/adjoint/*`,
   `work/KERNELS`, `work/ORI_KERNELS`, `work/linesearch`, `work/db/pert_*`,
   `.state/ite<x>`, `models/LBFGS_STATE`, stale `DATAP_*/SYNP_*/OPT_*/WIN_*`
   in the result dir, stale `work/adjoint_sources/*`.
3. Sentinels live in `$GPU_ROOT/.state/ite$ITER`, not under `work/`.
4. `SAVE_MESH_FILES` must stay `.false.` in solver runs (models are copied into
   the DB dir, never symlinked, so even a mistake cannot overwrite `models/`).
5. `USE_BINARY_FOR_SEISMOGRAMS` must be off in `SIMULATION_TYPE=3` runs; the
   runners set it per run.
6. Launch multi-hour drivers under `tmux` (or `setsid nohup`); a driver killed
   mid-`sbatch --wait` leaves the array running — it finishes on its own and
   the relaunched driver picks the sentinels up.

## Options deliberately not implemented

* `UNDO_ATTENUATION_AND_OR_PML=.true.` would replace the Stacey traction
  store with chunked forward re-computation (snapshots every
  `NT_DUMP_ATTENUATION` steps) — much less scratch at 1 s, same compute, but
  an untested code path for this workflow. Worth a smoke-test comparison of
  kernels if the 1 s stage's 600 GB/event proves awkward on GPFS.
* Several events per job (ite3's grouping): no benefit over arrays.
* ASDF output: the parent's `bin_asdf/` build is a separate track.
