# anvil_cpu/lib/solver.sh -- SPECFEM plumbing at NPROC ranks, N events in flight.
#
# The local port had ONE reusable solver workdir per stage because one GPU ran
# one event at a time. With a job array running ARRAY_THROTTLE events at once,
# every event needs its own LOCAL_PATH: the SAVE_FORWARD leg writes
# proc*_save_forward_arrays.bin / proc*_absorb_field.bin into it and the kernel
# leg writes proc*_*_kernel.bin there. So the layout is split in two:
#
#   $DBDIR/<model>/                   ONE per model (built by ensure_genmesh_model,
#     DATA/Par_file ...                 a NPROC-rank xgenerate_databases job)
#     DATABASES_MPI/proc*_Database  -> $MESHDATA (symlinks)
#     DATABASES_MPI/proc*_{vp,vs,rho}.bin   COPIED model (SAVE_MESH_FILES hazard, as before)
#     DATABASES_MPI/proc*_external_mesh.bin the baked mesh+model, READ-ONLY from here on
#     OUTPUT_FILES/values_from_mesher.h     xspecfem3D reads it at start-up
#     .gll_model / .done
#
#   $WORKBASE/<stage>/ev/<event>/     ONE per event per stage, made by the array task
#     bin -> $GPU_ROOT/bin
#     DATA/{Par_file,STATIONS,FORCESOLUTION[,STATIONS_ADJOINT]}   real files
#     DATABASES_MPI/proc*_external_mesh.bin -> $DBDIR/<model>/... (NPROC symlinks)
#     DATABASES_MPI/proc*_save_forward_arrays.bin | transient, this event only,
#     DATABASES_MPI/proc*_absorb_field.bin        | deleted by drop_wavefield
#     OUTPUT_FILES/                  seismograms / kernels land here or in DATABASES_MPI
#     SEM -> <event>/SEM             adjoint runs
#
# Symlinking NPROC mesh files per event costs nothing; copying them would be
# NPROC x ~10 MB x ARRAY_THROTTLE of scratch churn per stage for no benefit.

[[ -n "${_ITE3GPU_SOLVER_LOADED:-}" ]] && return 0
_ITE3GPU_SOLVER_LOADED=1
: "${_ITE3GPU_COMMON_LOADED:?solver.sh must be sourced after lib/common.sh}"

# db_dir <model_dir> -> the per-model database dir
db_dir() { printf '%s/%s' "$DBDIR" "$(basename "$1")"; }

# ---------------------------------------------------------------------------
# write_parfile <dest Par_file> -- the tree's Par_file with this port's invariants
# ---------------------------------------------------------------------------
write_parfile() {
  local f="$1"
  cp -f "$GPU_ROOT/DATA/Par_file" "$f"
  par_set "$f" NPROC            "$NPROC"
  par_set "$f" GPU_MODE         .false.
  par_set "$f" MODEL            gll
  par_set "$f" SAVE_MESH_FILES  .false.
  par_set "$f" NSTEP            "$NSTEP"
  par_set "$f" DT               "$DT"
  par_set "$f" SAVE_ALL_SEISMOS_IN_ONE_FILE .false.
  case "${SEISMO_FORMAT:-ascii}" in ascii|bin) ;; *) die "SEISMO_FORMAT must be ascii or bin" ;; esac
}

# ---------------------------------------------------------------------------
# stage_model <db> <model_dir> -- copy the GLL model into the DB dir
# ---------------------------------------------------------------------------
stage_model() {
  local db="$1" model="$2" n p
  [[ -d "$model" ]] || die "model directory not found: $model"
  for p in vp vs rho; do
    n=$(count_procfiles "$model" "_$p.bin")
    [[ "$n" -eq "$NPROC" ]] || die "model $model has $n proc*_$p.bin, expected $NPROC"
  done
  find "$db/DATABASES_MPI" -maxdepth 1 \( -name 'proc*_vp.bin' -o -name 'proc*_vs.bin' -o -name 'proc*_rho.bin' \) -delete
  # cp with a glob of 3*NPROC files is fine (ARG_MAX is 2 MB); find|xargs for safety at NPROC=2048
  find "$model" -maxdepth 1 \( -name 'proc*_vp.bin' -o -name 'proc*_vs.bin' -o -name 'proc*_rho.bin' \) -print0 \
    | xargs -0 cp -t "$db/DATABASES_MPI/"
  printf '%s\n' "$model" > "$db/DATABASES_MPI/.gll_model"
}

# ---------------------------------------------------------------------------
# ensure_genmesh_model <model_dir>   (DRIVER side: submits a job unless in_slurm)
# Builds $DBDIR/<model>/ once per model. Prints nothing; dies on failure.
# ---------------------------------------------------------------------------
ensure_genmesh_model() {
  local model="$1" db; db=$(db_dir "$model")
  local have; have=$(cat "$db/.gll_model" 2>/dev/null || echo "")
  if [[ -f "$db/.done" && "$have" == "$model" \
        && "$(count_procfiles "$db/DATABASES_MPI" "_external_mesh.bin")" -eq "$NPROC" ]]; then
    log "genmesh: reusing $db for $(basename "$model")"
    return 0
  fi
  [[ -f "$MESHDATA/.mesh_stamp" ]] || die "no mesh at $MESHDATA -- run ./2_make_mesh.sh"
  [[ "$(count_procfiles "$MESHDATA" "_Database")" -eq "$NPROC" ]] \
    || die "$MESHDATA has $(count_procfiles "$MESHDATA" "_Database") Database files, expected $NPROC (re-mesh after a plan change)"
  [[ -d "$GPU_ROOT/DATA/tomo_files" ]] || die "missing $GPU_ROOT/DATA/tomo_files (genmesh resolves the Mesh_Par_file material line even under MODEL=gll)"

  log "genmesh: building $db for $(basename "$model")"
  rm -rf "$db"; mkdir -p "$db/DATA" "$db/DATABASES_MPI" "$db/OUTPUT_FILES"
  ln -sfn "$GPU_ROOT/bin" "$db/bin"
  ln -sfn "$GPU_ROOT/DATA/meshfem3D_files" "$db/DATA/meshfem3D_files"
  ln -sfn "$GPU_ROOT/DATA/tomo_files"      "$db/DATA/tomo_files"
  cp -f "$GPU_ROOT/DATA/STATIONS" "$db/DATA/STATIONS"
  cp -f "$GPU_ROOT/DATA/FORCESOLUTION" "$db/DATA/FORCESOLUTION"   # genmesh opens it; content irrelevant
  write_parfile "$db/DATA/Par_file"
  find "$MESHDATA" -maxdepth 1 -name 'proc*_Database' -print0 | xargs -0 ln -sf -t "$db/DATABASES_MPI/"
  stage_model "$db" "$model"
  printf '%s\n' "$model" > "$db/.gll_model"

  if in_slurm; then
    ( cd "$db" && mpi_run "$NPROC" "$GPU_ROOT/bin/xgenerate_databases" ) >>"$LOGFILE" 2>&1 \
      || die "xgenerate_databases FAILED in $db"
  else
    source "$GPU_ROOT/lib/slurm.sh"
    sbatch_wait "genmesh_$(basename "$model")" "$NODES_PER_EVENT" "$NPROC" "$SERVICE_WALL" "${PART_COMPUTE:?}" -- \
      bash "$GPU_ROOT/lib/genmesh_job.sh" "$db" \
      || die "genmesh job for $(basename "$model") failed -- $LOGDIR/slurm/"
  fi
  [[ "$(count_procfiles "$db/DATABASES_MPI" "_external_mesh.bin")" -eq "$NPROC" ]] \
    || die "genmesh produced $(count_procfiles "$db/DATABASES_MPI" "_external_mesh.bin") external_mesh.bin, expected $NPROC"
  [[ -f "$db/OUTPUT_FILES/values_from_mesher.h" ]] || die "genmesh left no values_from_mesher.h in $db"
  : > "$db/.done"
  log "genmesh OK -> $db"
}

# ---------------------------------------------------------------------------
# prepare_event_workdir <wd> <db> -- per-event, per-stage solver dir
# ---------------------------------------------------------------------------
prepare_event_workdir() {
  local wd="$1" db="$2"
  [[ -f "$db/.done" ]] || die "database $db is not built (ensure_genmesh_model first)"
  mkdir -p "$wd/DATA" "$wd/DATABASES_MPI" "$wd/OUTPUT_FILES"
  ln -sfn "$GPU_ROOT/bin" "$wd/bin"
  ln -sfn "$GPU_ROOT/DATA/meshfem3D_files" "$wd/DATA/meshfem3D_files"
  cp -f "$db/DATA/STATIONS" "$wd/DATA/STATIONS"
  cp -f "$db/DATA/Par_file" "$wd/DATA/Par_file"
  cp -f "$db/OUTPUT_FILES/values_from_mesher.h" "$wd/OUTPUT_FILES/values_from_mesher.h"
  # NPROC read-only mesh links; any stale wavefield/kernel from a killed task goes first
  find "$wd/DATABASES_MPI" -maxdepth 1 -type f -delete
  find "$db/DATABASES_MPI" -maxdepth 1 -name 'proc*_external_mesh.bin' -print0 | xargs -0 ln -sf -t "$wd/DATABASES_MPI/"
  [[ "$(count_procfiles "$wd/DATABASES_MPI" "_external_mesh.bin")" -eq "$NPROC" ]] \
    || die "event workdir $wd: mesh link count != $NPROC"
}

write_seismo_meta() {
  local d="$1" log="$1/output_solver.txt" st
  st=$(sed -n 's/^ *start time: *\([-+0-9.Ee]*\).*/\1/p' "$log" 2>/dev/null | head -1)
  [[ -n "$st" ]] || { warn "$(basename "$d"): no 'start time' in output_solver.txt"; st="NA"; }
  { echo "format=${SEISMO_FORMAT:-ascii}"; echo "nstep=$NSTEP"; echo "dt=$DT"
    echo "start_time=$st"; echo "band=$(band_code)"; echo "written=$(date -Is)"; } > "$d/seismo_meta.txt"
}

clean_output() {
  find "$1/OUTPUT_FILES" -maxdepth 1 -type f \
       \( -name '*.sem*' -o -name 'output_*' -o -name '*.vtk' -o -name 'sr.vtk' \) -delete 2>/dev/null || true
}
drop_wavefield() {
  find "$1/DATABASES_MPI" -maxdepth 1 -type f \
       \( -name 'proc*_save_forward_arrays.bin' -o -name 'proc*_absorb_field.bin' \) -delete 2>/dev/null || true
}
wavefield_gb() {   # what the transient store measures right now
  du -scm "$1"/DATABASES_MPI/proc*_save_forward_arrays.bin "$1"/DATABASES_MPI/proc*_absorb_field.bin 2>/dev/null \
    | tail -1 | awk '{printf "%.1f", $1/1024}'
}

# ---------------------------------------------------------------------------
# run_forward_event <wd> <event_src> <dest> <save_forward:0|1>   (inside a job)
# ---------------------------------------------------------------------------
run_forward_event() {
  local wd="$1" src="$2" dest="$3" save="${4:-0}"
  local name; name=$(basename "$src")
  [[ -f "$src/FORCESOLUTION" ]] || { warn "$name: no FORCESOLUTION in $src -- skipping"; return 1; }
  mkdir -p "$dest"
  cp -f "$src/FORCESOLUTION" "$wd/DATA/FORCESOLUTION"
  if [[ "$save" -eq 1 ]]; then sim_forward_save "$wd/DATA/Par_file"; else sim_forward "$wd/DATA/Par_file"; fi
  assert_cpu_parfile "$wd/DATA/Par_file"
  clean_output "$wd"
  cp "$wd/DATA/Par_file" "$wd/DATA/STATIONS" "$wd/DATA/FORCESOLUTION" "$wd/OUTPUT_FILES/" 2>/dev/null || true

  local t0 t1 rc; t0=$(date +%s)
  ( cd "$wd" && mpi_run "$NPROC" "$GPU_ROOT/bin/xspecfem3D" ) >>"$LOGFILE" 2>&1; rc=$?
  t1=$(date +%s)
  if [[ $rc -ne 0 ]]; then
    warn "$name: xspecfem3D forward FAILED (rc=$rc)"
    tail -20 "$wd/OUTPUT_FILES/output_solver.txt" 2>/dev/null | tee -a "$LOGFILE"
    return $rc
  fi
  # Z component only (ambient-noise ZZ inversion), as in ite3/ite3_gpu.
  local moved=0 f
  for f in "$wd"/OUTPUT_FILES/*Z.sem*; do [[ -e "$f" ]] || continue; mv "$f" "$dest/"; moved=$((moved + 1)); done
  for f in output_list_stations.txt output_list_sources.txt output_solver.txt sr.vtk STATIONS FORCESOLUTION; do
    [[ -e "$wd/OUTPUT_FILES/$f" ]] && mv "$wd/OUTPUT_FILES/$f" "$dest/"
  done
  if [[ "$moved" -lt "${EVENT_MIN_SEMD:-700}" ]]; then
    warn "$name: only $moved seismograms (expected >= ${EVENT_MIN_SEMD:-700}) -- NOT marking done"; return 1
  fi
  write_seismo_meta "$dest"
  : > "$dest/.event_done"
  log "  $name: forward OK, $moved seismograms, $(human_time $((t1 - t0)))$( [[ "$save" -eq 1 ]] && echo ", wavefield $(wavefield_gb "$wd") GB")"
  return 0
}

# ---------------------------------------------------------------------------
# run_kernel_event <wd> <adj_event_dir>   (inside a job, right after a save leg)
# ---------------------------------------------------------------------------
run_kernel_event() {
  local wd="$1" ev="$2"
  local name; name=$(basename "$ev")
  for f in FORCESOLUTION STATIONS_ADJOINT; do [[ -f "$ev/$f" ]] || { warn "$name: missing $f"; return 1; }; done
  [[ -d "$ev/SEM" ]] || { warn "$name: missing SEM/"; return 1; }
  cp -f "$ev/FORCESOLUTION"    "$wd/DATA/FORCESOLUTION"
  cp -f "$ev/STATIONS_ADJOINT" "$wd/DATA/STATIONS_ADJOINT"
  rm -f "$wd/SEM"; ln -sfn "$ev/SEM" "$wd/SEM"
  sim_kernel "$wd/DATA/Par_file"
  assert_cpu_parfile "$wd/DATA/Par_file"
  clean_output "$wd"
  cp "$wd/DATA/Par_file" "$wd/DATA/STATIONS" "$wd/DATA/FORCESOLUTION" "$wd/DATA/STATIONS_ADJOINT" "$wd/OUTPUT_FILES/" 2>/dev/null || true

  local t0 t1 rc; t0=$(date +%s)
  ( cd "$wd" && mpi_run "$NPROC" "$GPU_ROOT/bin/xspecfem3D" ) >>"$LOGFILE" 2>&1; rc=$?
  t1=$(date +%s)
  rm -f "$wd/SEM"
  if [[ $rc -ne 0 ]]; then
    warn "$name: xspecfem3D kernel run FAILED (rc=$rc)"
    tail -20 "$wd/OUTPUT_FILES/output_solver.txt" 2>/dev/null | tee -a "$LOGFILE"
    return $rc
  fi
  mkdir -p "$ev/KERNEL" "$ev/OUTPUT_FILES"
  find "$ev/KERNEL" -maxdepth 1 -type f -delete
  local nk
  nk=$(find "$wd/DATABASES_MPI" -maxdepth 1 -type f \( -name '*kernel.bin' -o -name '*source.bin' \) | wc -l)
  find "$wd/DATABASES_MPI" -maxdepth 1 -type f \( -name '*kernel.bin' -o -name '*source.bin' \) -print0 \
    | xargs -0 -r mv -t "$ev/KERNEL/"
  for f in output_list_stations.txt output_solver.txt sr.vtk; do
    [[ -e "$wd/OUTPUT_FILES/$f" ]] && mv "$wd/OUTPUT_FILES/$f" "$ev/OUTPUT_FILES/"
  done
  local missing="" k
  for k in alpha_kernel beta_kernel rho_kernel hess_kernel; do
    [[ "$(count_procfiles "$ev/KERNEL" "_${k}.bin")" -eq "$NPROC" ]] || missing="$missing $k"
  done
  if [[ -n "$missing" ]]; then warn "$name: kernel run finished but incomplete:$missing"; return 1; fi
  : > "$ev/.kernel_done"
  log "  $name: kernels OK ($nk files, $(human_time $((t1 - t0))))"
  return 0
}

kernel_event_complete() {
  local ev="$1" k
  [[ -f "$ev/.kernel_done" ]] || return 1
  for k in alpha_kernel beta_kernel rho_kernel hess_kernel; do
    [[ -e "$ev/KERNEL/$(printf 'proc%06d' $((NPROC - 1)))_${k}.bin" ]] || return 1
  done
  return 0
}
