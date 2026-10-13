# ite3_gpu/lib/solver.sh -- SPECFEM solver plumbing for the single-GPU workflow.
#
# On Anvil each stage got its own SLURM job with its own staged copy of bin/,
# DATA/ and DATABASES_MPI/ -- 10 groups x 20 MB of binaries, per stage. With one
# GPU running one event at a time, all of that collapses into a single reusable
# workdir per stage. These functions build and drive it.
#
# The workdir layout (identical in shape to what SPECFEM expects anywhere):
#
#   <wd>/
#     bin              -> symlink to $GPU_ROOT/bin
#     DATA/
#       Par_file         REAL file, rewritten per run (SIMULATION_TYPE etc.)
#       FORCESOLUTION    REAL file, replaced per event
#       STATIONS         copy
#       STATIONS_ADJOINT copy, adjoint runs only
#       meshfem3D_files/ copy
#     DATABASES_MPI/
#       proc*_external_mesh.bin     from genmesh
#       proc*_{vp,vs,rho}.bin       the GLL model (COPIED, see note below)
#       .gll_model                  which model is staged -> genmesh cache key
#       proc*_save_forward_arrays.bin |  written by a SAVE_FORWARD run,
#       proc*_absorb_field.bin        |  consumed by the SIMULATION_TYPE=3 run,
#                                     |  deleted straight after (~29 GB)
#     OUTPUT_FILES/                 seismograms land here, harvested per event
#     SEM -> <event>/SEM            adjoint sources, adjoint runs only
#
# MODEL FILES ARE COPIED, NOT SYMLINKED. ite3 symlinks them, which is safe there
# only because SAVE_MESH_FILES=.false. -- if it were ever .true., genmesh would
# write proc*_vp.bin back THROUGH the symlink and silently destroy the canonical
# model. At NPROC=1 the model is 3 files / ~300 MB, so copying costs a second and
# removes the hazard entirely. We also assert SAVE_MESH_FILES=.false. anyway.

[[ -n "${_ITE3GPU_SOLVER_LOADED:-}" ]] && return 0
_ITE3GPU_SOLVER_LOADED=1
: "${_ITE3GPU_COMMON_LOADED:?solver.sh must be sourced after lib/common.sh}"

# ---------------------------------------------------------------------------
# prepare_workdir <wd>
# Build (or refresh) a solver workdir. Safe to call repeatedly; it never touches
# DATABASES_MPI, so an existing mesh/model staging survives.
# ---------------------------------------------------------------------------
prepare_workdir() {
  local wd="$1"
  mkdir -p "$wd/DATA" "$wd/DATABASES_MPI" "$wd/OUTPUT_FILES"
  ln -sfn "$GPU_ROOT/bin" "$wd/bin"

  # meshfem3D_files: genmesh reads it even under MODEL=gll (the Mesh_Par_file
  # declares the material as `-1 tomography elastic tomography_model.xyz`, so the
  # tomo file must be resolvable or genmesh aborts with "could not open
  # tomography file"). Symlinked: read-only, and the tomo file is 152 MB.
  ln -sfn "$GPU_ROOT/DATA/meshfem3D_files" "$wd/DATA/meshfem3D_files"
  # tomo_files only exists after staging the tomography file (not distributed with this repository). Check rather
  # than leave a dangling symlink -- genmesh's failure mode for a missing tomo
  # file is an opaque "could not open tomography file" deep in Fortran.
  [[ -d "$GPU_ROOT/DATA/tomo_files" ]] \
    || die "missing $GPU_ROOT/DATA/tomo_files -- stage tomography_model.xyz there (not distributed with this repository)
     (genmesh needs it even under MODEL=gll: Mesh_Par_file declares the material
      as '-1 tomography elastic tomography_model.xyz', so the file must resolve)"
  ln -sfn "$GPU_ROOT/DATA/tomo_files"      "$wd/DATA/tomo_files"

  # STATIONS is read-only per run -> copy once (real file; SPECFEM opens it on
  # rank 0 and a dangling link here is a confusing failure).
  cp -f "$GPU_ROOT/DATA/STATIONS" "$wd/DATA/STATIONS"

  # FORCESOLUTION: xgenerate_databases (run by ensure_genmesh below) OPENS
  # DATA/FORCESOLUTION even though the mesh+model it bakes into external_mesh.bin
  # does not depend on the source -- at NPROC=1 there is one partition and source
  # location is irrelevant to it. Without a source file it dies with
  # "Error opening source SOLUTION file: ./DATA/FORCESOLUTION" before any event
  # is staged. Stage the template so genmesh always has one; the per-event
  # runners (stage_event / the adjoint stager) cp -f the real event file over it
  # for the actual simulation. 2_make_mesh.sh and 3_make_M0.sh already do exactly
  # this for their standalone genmesh; prepare_workdir was missing it, so bench,
  # forward and adjoint all failed at genmesh. Only copy if absent, so a
  # per-event FORCESOLUTION already staged is never clobbered.
  [[ -f "$wd/DATA/FORCESOLUTION" ]] \
    || cp -f "$GPU_ROOT/DATA/FORCESOLUTION" "$wd/DATA/FORCESOLUTION" 2>/dev/null \
    || die "no DATA/FORCESOLUTION template in $GPU_ROOT/DATA -- genmesh needs a source file"

  # Par_file is REWRITTEN every run -> must be a real, private file.
  [[ -f "$wd/DATA/Par_file" ]] || cp "$GPU_ROOT/DATA/Par_file" "$wd/DATA/Par_file"

  # These are the invariants for every run in this tree; set them once here so
  # no stage can drift.
  par_set "$wd/DATA/Par_file" NPROC            "$NPROC"
  par_set "$wd/DATA/Par_file" GPU_MODE         .true.
  par_set "$wd/DATA/Par_file" MODEL            gll
  par_set "$wd/DATA/Par_file" SAVE_MESH_FILES  .false.
  par_set "$wd/DATA/Par_file" NSTEP            "$NSTEP"
  par_set "$wd/DATA/Par_file" DT               "$DT"
  par_set "$wd/DATA/Par_file" SAVE_ALL_SEISMOS_IN_ONE_FILE .false.
  # USE_BINARY_FOR_SEISMOGRAMS (config.sh SEISMO_FORMAT) is set PER RUN by
  # sim_forward/sim_forward_save/sim_kernel in lib/common.sh: SPECFEM refuses a
  # SIMULATION_TYPE=3 run with it on ("Adjoint simulations not supported with
  # .bin format"), so the kernel leg must always turn it off.
  case "${SEISMO_FORMAT:-ascii}" in ascii|bin) ;; *) die "SEISMO_FORMAT must be ascii or bin (got '${SEISMO_FORMAT}')" ;; esac
}

# ---------------------------------------------------------------------------
# write_seismo_meta <event_result_dir>
# Sidecar describing the traces in that dir: format, NSTEP, DT, band code and
# the solver's start time (t0). Binary traces carry no time column, so readers
# rebuild t = start_time + i*dt from this file; for ascii it is informative.
# dt/NSTEP come from config.sh (exact decimals); start_time is parsed from the
# solver's own log line ("start time: -0.108750001 seconds"), never re-derived.
# ---------------------------------------------------------------------------
write_seismo_meta() {
  local d="$1" log="$1/output_solver.txt" st
  st=$(sed -n 's/^ *start time: *\([-+0-9.Ee]*\).*/\1/p' "$log" 2>/dev/null | head -1)
  [[ -n "$st" ]] || { warn "$(basename "$d"): no 'start time' in output_solver.txt -- seismo_meta.txt incomplete"; st="NA"; }
  {
    echo "format=${SEISMO_FORMAT:-ascii}"
    echo "nstep=$NSTEP"
    echo "dt=$DT"
    echo "start_time=$st"
    echo "band=$(band_code)"
    echo "written=$(date -Is)"
  } > "$d/seismo_meta.txt"
}

# ---------------------------------------------------------------------------
# stage_model <wd> <model_dir>
# Copy the GLL model into DATABASES_MPI and record which model it is.
# ---------------------------------------------------------------------------
stage_model() {
  local wd="$1" model="$2" n
  [[ -d "$model" ]] || die "model directory not found: $model"
  for p in vp vs rho; do
    n=$(count_procfiles "$model" "_$p.bin")
    [[ "$n" -eq "$NPROC" ]] || die "model $model has $n proc*_$p.bin, expected $NPROC"
  done
  rm -f "$wd"/DATABASES_MPI/proc*_vp.bin "$wd"/DATABASES_MPI/proc*_vs.bin "$wd"/DATABASES_MPI/proc*_rho.bin
  cp "$model"/proc*_vp.bin "$model"/proc*_vs.bin "$model"/proc*_rho.bin "$wd/DATABASES_MPI/"
  printf '%s\n' "$model" > "$wd/DATABASES_MPI/.gll_model"
}

# ---------------------------------------------------------------------------
# ensure_genmesh <wd> <model_dir> [--force]
# Run xgenerate_databases if (and only if) the staged model changed. This is the
# NPROC=1 analogue of ite3's top-level genmesh: external_mesh.bin has the model
# baked into it, so it must be regenerated whenever the model does.
# ---------------------------------------------------------------------------
ensure_genmesh() {
  local wd="$1" model="$2" force="${3:-}"
  local have nmesh
  have=$(cat "$wd/DATABASES_MPI/.gll_model" 2>/dev/null || echo "")
  nmesh=$(count_procfiles "$wd/DATABASES_MPI" "_external_mesh.bin")

  if [[ "$force" != "--force" && "$nmesh" -eq "$NPROC" && "$have" == "$model" ]]; then
    log "genmesh: reusing existing mesh for $(basename "$model")"
    return 0
  fi

  log "genmesh: building databases for model $(basename "$model")"
  # Fresh Database links + model copy, then run.
  rm -f "$wd"/DATABASES_MPI/proc*_Database "$wd"/DATABASES_MPI/proc*_external_mesh.bin
  ln -sf "$MESHDATA"/proc*_Database "$wd/DATABASES_MPI/"
  stage_model "$wd" "$model"

  local t0 t1
  t0=$(date +%s)
  ( cd "$wd" && gpu_run "$NPROC" "$GPU_ROOT/bin/xgenerate_databases" ) >>"$LOGFILE" 2>&1 || {
    tail -30 "$wd/OUTPUT_FILES/output_generate_databases.txt" 2>/dev/null | tee -a "$LOGFILE"
    die "xgenerate_databases FAILED in $wd"
  }
  t1=$(date +%s)

  nmesh=$(count_procfiles "$wd/DATABASES_MPI" "_external_mesh.bin")
  [[ "$nmesh" -eq "$NPROC" ]] || die "genmesh produced $nmesh external_mesh.bin, expected $NPROC"
  log "genmesh OK ($(human_time $((t1 - t0))))"
}

# ---------------------------------------------------------------------------
# clean_output <wd> -- wipe per-event solver output without touching the mesh
# ---------------------------------------------------------------------------
clean_output() {
  local wd="$1"
  find "$wd/OUTPUT_FILES" -maxdepth 1 -type f \
       \( -name '*.sem*' -o -name 'output*' -o -name '*.vtk' -o -name 'sr.vtk' \) -delete 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# drop_wavefield <wd> -- delete the reconstruction wavefield (~29 GB)
# ---------------------------------------------------------------------------
drop_wavefield() {
  local wd="$1"
  rm -f "$wd"/DATABASES_MPI/proc*_save_forward_arrays.bin \
        "$wd"/DATABASES_MPI/proc*_absorb_field.bin 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# run_forward_event <wd> <event_src> <dest> <save_forward:0|1>
#
#   event_src : dir holding FORCESOLUTION (from the source set)
#   dest      : where seismograms are collected (usually == event_src's copy)
#   save_forward : 1 -> also write the reconstruction wavefield into
#                       DATABASES_MPI for a following SIMULATION_TYPE=3 run.
#                       Costs ~29 GB of disk and extra write time; only the
#                       adjoint stage needs it.
# Returns 0 on success. Writes dest/.event_done on success.
# ---------------------------------------------------------------------------
run_forward_event() {
  local wd="$1" src="$2" dest="$3" save="${4:-0}"
  local name; name=$(basename "$src")

  [[ -f "$src/FORCESOLUTION" ]] || { warn "$name: no FORCESOLUTION in $src -- skipping"; return 1; }
  mkdir -p "$dest"
  cp -f "$src/FORCESOLUTION" "$wd/DATA/FORCESOLUTION"

  if [[ "$save" -eq 1 ]]; then sim_forward_save "$wd/DATA/Par_file"
  else                         sim_forward      "$wd/DATA/Par_file"; fi
  assert_gpu_parfile "$wd/DATA/Par_file"

  clean_output "$wd"
  cp "$wd/DATA/Par_file" "$wd/DATA/STATIONS" "$wd/DATA/FORCESOLUTION" "$wd/OUTPUT_FILES/" 2>/dev/null || true

  local t0 t1 rc
  t0=$(date +%s)
  ( cd "$wd" && gpu_run "$NPROC" "$GPU_ROOT/bin/xspecfem3D" ) >>"$LOGFILE" 2>&1
  rc=$?
  t1=$(date +%s)
  if [[ $rc -ne 0 ]]; then
    warn "$name: xspecfem3D forward FAILED (rc=$rc)"
    tail -20 "$wd/OUTPUT_FILES/output_solver.txt" 2>/dev/null | tee -a "$LOGFILE"
    return $rc
  fi

  # Harvest the VERTICAL COMPONENT ONLY, matching ite3's Anvil script.
  # SPECFEM writes all three components (SAVE_SEISMOGRAMS_DISPLACEMENT is a
  # single flag, there is no per-component switch), but this is an ambient-noise
  # ZZ inversion: cal_adjoint reads only <net>.<sta>.<band>Z.semd, and stage 4
  # writes the N/E adjoint components as zeros. Keeping all three would triple
  # the on-disk result (~125 MB -> ~375 MB per event, 12 GB -> 37 GB per
  # iteration) for data nothing reads. The rest are dropped by clean_output()
  # on the next event.
  local moved=0 f
  for f in "$wd"/OUTPUT_FILES/*Z.sem*; do
    [[ -e "$f" ]] || continue
    mv "$f" "$dest/"; moved=$((moved + 1))
  done
  for f in "$wd"/OUTPUT_FILES/output_list_stations.txt "$wd"/OUTPUT_FILES/output_list_sources.txt \
           "$wd"/OUTPUT_FILES/output_solver.txt "$wd"/OUTPUT_FILES/sr.vtk \
           "$wd"/OUTPUT_FILES/STATIONS "$wd"/OUTPUT_FILES/FORCESOLUTION; do
    [[ -e "$f" ]] && mv "$f" "$dest/"
  done

  if [[ "$moved" -lt "${EVENT_MIN_SEMD:-700}" ]]; then
    warn "$name: only $moved seismograms (expected >= ${EVENT_MIN_SEMD:-700}) -- NOT marking done"
    return 1
  fi
  write_seismo_meta "$dest"

  : > "$dest/.event_done"
  log "  $name: forward OK, $moved seismograms, $(human_time $((t1 - t0)))"
  return 0
}

# ---------------------------------------------------------------------------
# run_kernel_event <wd> <adj_event_dir>
#
# The SIMULATION_TYPE=3 leg. Requires that a SAVE_FORWARD run for the SAME event
# and SAME model just left its wavefield in <wd>/DATABASES_MPI.
#
#   adj_event_dir must contain: FORCESOLUTION, STATIONS_ADJOINT, SEM/*.adj
# Kernels are moved to <adj_event_dir>/KERNEL/.
# ---------------------------------------------------------------------------
run_kernel_event() {
  local wd="$1" ev="$2"
  local name; name=$(basename "$ev")

  for f in FORCESOLUTION STATIONS_ADJOINT; do
    [[ -f "$ev/$f" ]] || { warn "$name: missing $f -- skipping"; return 1; }
  done
  [[ -d "$ev/SEM" ]] || { warn "$name: missing SEM/ -- skipping"; return 1; }

  cp -f "$ev/FORCESOLUTION"    "$wd/DATA/FORCESOLUTION"
  cp -f "$ev/STATIONS_ADJOINT" "$wd/DATA/STATIONS_ADJOINT"
  # SPECFEM reads adjoint sources from ./SEM relative to the run dir.
  rm -f "$wd/SEM"; ln -sfn "$ev/SEM" "$wd/SEM"

  sim_kernel "$wd/DATA/Par_file"
  assert_gpu_parfile "$wd/DATA/Par_file"

  clean_output "$wd"
  cp "$wd/DATA/Par_file" "$wd/DATA/STATIONS" "$wd/DATA/FORCESOLUTION" \
     "$wd/DATA/STATIONS_ADJOINT" "$wd/OUTPUT_FILES/" 2>/dev/null || true

  local t0 t1 rc
  t0=$(date +%s)
  ( cd "$wd" && gpu_run "$NPROC" "$GPU_ROOT/bin/xspecfem3D" ) >>"$LOGFILE" 2>&1
  rc=$?
  t1=$(date +%s)
  rm -f "$wd/SEM"
  if [[ $rc -ne 0 ]]; then
    warn "$name: xspecfem3D kernel run FAILED (rc=$rc)"
    tail -20 "$wd/OUTPUT_FILES/output_solver.txt" 2>/dev/null | tee -a "$LOGFILE"
    return $rc
  fi

  mkdir -p "$ev/KERNEL" "$ev/OUTPUT_FILES"
  rm -f "$ev"/KERNEL/*
  local nk=0 f
  for f in "$wd"/DATABASES_MPI/*kernel.bin "$wd"/DATABASES_MPI/*source.bin; do
    [[ -e "$f" ]] || continue
    mv "$f" "$ev/KERNEL/"; nk=$((nk + 1))
  done
  # clip_iso.py locates each event's receivers via
  # group_*/RUN/<event>/OUTPUT_FILES/output_list_stations.txt -- keep that file.
  for f in "$wd"/OUTPUT_FILES/output_list_stations.txt "$wd"/OUTPUT_FILES/output_solver.txt \
           "$wd"/OUTPUT_FILES/sr.vtk; do
    [[ -e "$f" ]] && mv "$f" "$ev/OUTPUT_FILES/"
  done

  # The four the post-processing chain actually consumes.
  local missing=""
  for k in alpha_kernel beta_kernel rho_kernel hess_kernel; do
    [[ -e "$ev/KERNEL/proc000000_${k}.bin" ]] || missing="$missing $k"
  done
  if [[ -n "$missing" ]]; then
    warn "$name: kernel run finished but missing:$missing"
    return 1
  fi

  : > "$ev/.kernel_done"
  log "  $name: kernels OK ($nk files, $(human_time $((t1 - t0))))"
  return 0
}

# ---------------------------------------------------------------------------
# kernel_event_complete <adj_event_dir>
# Explicit sentinel + file check. Replaces ite3's `du | grep '[3,4]\.[0-9]G'`
# heuristic, which was calibrated on ite2's 60-layer mesh (3.4 GB kernels) and
# silently never matches ite3's 18-layer mesh (~1.0 GB) -- so every resubmit
# recomputed already-finished events.
# ---------------------------------------------------------------------------
kernel_event_complete() {
  local ev="$1"
  [[ -f "$ev/.kernel_done" ]] || return 1
  for k in alpha_kernel beta_kernel rho_kernel hess_kernel; do
    [[ -e "$ev/KERNEL/proc000000_${k}.bin" ]] || return 1
  done
  return 0
}
