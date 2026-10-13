# Hard-won gotchas

Each item below cost real debugging time during the 2026 Willamette Valley campaigns. Scripts in
`workflows/` cite them as "GOTCHAS item N". The numbering is stable.

1. **Band code follows DT.** SPECFEM derives the channel band code from the time step: DT > 0.0125 s
   gives `BX`, DT <= 0.0125 s gives `HX`. Every file name keys on it (`.semd`, `.adj`, `.win`). Never
   mix window caches, adjoint sources or results across a DT change; the 1.5 s and 1 s meshes
   (DT ~0.011 and ~0.007 s) flip to `HX` and need a completely fresh measurement basis.

2. **Basis-change reset checklist.** When the windows, band, source set or misfit definition change:
   archive `misfit_history.txt`; clear the loop marks for the current model; wipe kernel scratch
   (`work/adjoint/group_*`, `.kernel_model`, `.stage_done`, `work/KERNELS`, `work/ORI_KERNELS`,
   `work/linesearch`) **and** the iteration-keyed post stamps in `.state/ite<x>/`; wipe the L-BFGS
   state; remove stale `DATAP_* / SYNP_* / OPT_* / WIN_*` directories inside the result directory (the
   misfit sums every `.adjinfo` it finds); and remove stale event directories from
   `work/adjoint_sources` (old adjoint sources with a different NSTEP make the kernel run abort with
   "file has wrong length").

3. **MPI slots.** Open MPI counts physical cores as slots; with SMT on, `mpirun -np <threads>` is
   refused. Set `CALADJ_NPROC` to the number of physical cores (16 on the RTX 5090 workstation, 72 on
   a Grace superchip).

4. **Misfit summation.** Never `xargs awk` over `.adjinfo` files: xargs may split the list across
   several awk processes and concatenate their partial sums. The robust single-awk `getline` form is
   in `run_caladj_local.sh` and `linesearch.sh`; `.adjinfo` files are written with a trailing newline.

5. **Mesh-database generation inputs.** `xgenerate_databases` opens `DATA/FORCESOLUTION` (staged by
   `lib/solver.sh:prepare_workdir`), and the solver needs `OUTPUT_FILES/values_from_mesher.h`.

6. **Mesh-database generation cost.** At NPROC=1 the database step took 16–25 min per model on the 2 s
   mesh and recurs for every model (four times per iteration with the line search). Most of it is a
   quadratic adjacency setup in SPECFEM3D 4.1.1 (`setup_mesh_adjacency.f90`); a 12-line fix gives a
   bitwise-identical database in 98 s (see `patches/README.md`).

7. **Workstation filesystem.** On WSL2, never put `WORKBASE` on a `/mnt/*` (9p) mount: the adjoint
   stage streams tens of GB per source through it. Host sleep kills runs (they resume, but keep the
   host awake).

8. **Data safety.** Confirm before deleting large result or model directories; park instead of
   deleting when in doubt. Several results were regenerated at real GPU cost after a cleanup.

9. **Window cache staging** never clears stale window directories, and `find -type d` misses
   symlinked event directories: use real copies in event sets.

10. **Line-search basis.** Pair decimation is deterministic but depends on the pool, so subset
    `--misfit` runs can keep pairs the full-pool run never measured, biasing every trial against the
    step-0 anchor. `linesearch.sh` therefore sums every trial over one kernel-pair manifest, and the
    manifest is wiped whenever the model changes. Under per-iteration re-windowing the objective
    changes every iteration, so L-BFGS curvature pairs mix objective drift into curvature; the loop
    therefore runs preconditioned steepest descent plus line search in that mode.

11. **Stale line-search result after an early stop.** The CONVERGED branch must remove
    `work/linesearch/ls_alpha.txt` before exiting, or the next launch silently reuses the old step
    length. `polyfit.py` falls back to the lowest measured point when the fitted parabola is concave.
