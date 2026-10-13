#!/bin/bash
# anvil_cpu/2_make_mesh.sh -- xmeshfem3D at NPROC ranks (one SLURM job) -> $MESHDATA.
#
# Same guards as ite3_gpu: the partitioning in Mesh_Par_file / Par_file is
# forced to config (from mesh_plan.sh), NEX must divide by NPROC_XI/ETA, the
# geometry is stamped so 3_make_M0.sh can refuse a stale mesh. NPROC Database
# files are saved (1024-2048 files, ~1-3 GB) -- a MODEL-INDEPENDENT product
# built once per plan. Outside SLURM the script submits itself.
#
# Usage: ./2_make_mesh.sh [--force]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=mesh; source "$ROOT/lib/common.sh"; source "$ROOT/lib/slurm.sh"
FORCE=0; [[ "${1:-}" == "--force" ]] && FORCE=1
MPF="$ROOT/DATA/meshfem3D_files/Mesh_Par_file"; PAR="$ROOT/DATA/Par_file"
[[ -f "$MPF" ]] || die "missing $MPF -- run tools/plan_mesh.py <period> --apply first"
[[ -x "$ROOT/bin/xmeshfem3D" ]] || die "xmeshfem3D not built -- ./1_build_specfem_cpu.sh"
[[ $((NPROC_XI * NPROC_ETA)) -eq "$NPROC" ]] || die "NPROC_XI*NPROC_ETA != NPROC in config"
nex_xi=$(par_get "$MPF" NEX_XI); nex_eta=$(par_get "$MPF" NEX_ETA)
[[ "$nex_xi" == "$NEX_XI" && "$nex_eta" == "$NEX_ETA" ]] || die "Mesh_Par_file NEX $nex_xi x $nex_eta != config $NEX_XI x $NEX_ETA (re-run plan_mesh.py --apply)"
[[ $((nex_xi % NPROC_XI)) -eq 0 && $((nex_eta % NPROC_ETA)) -eq 0 ]] || die "NEX not divisible by NPROC_XI/ETA"
for kv in "NPROC_XI $NPROC_XI" "NPROC_ETA $NPROC_ETA"; do set -- $kv; [[ "$(par_get "$MPF" "$1")" == "$2" ]] || par_set "$MPF" "$1" "$2"; done
[[ "$(par_get "$PAR" NPROC)" == "$NPROC" ]] || par_set "$PAR" NPROC "$NPROC"
nz=$(grep -vE '^\s*#' "$MPF" | awk 'NF==7 && $7 ~ /^-?[0-9]+$/ {print $6; exit}')
STAMP="NEX_XI=$nex_xi NEX_ETA=$nex_eta NZ=$nz NPROC=$NPROC"
banner "mesh generation: $STAMP  ($((nex_xi * nex_eta * nz)) elements)"
[[ "$nz" == "$NEX_Z" ]] || warn "Mesh_Par_file NZ=$nz but config NEX_Z=$NEX_Z"
if [[ -f "$MESHDATA/.mesh_stamp" && "$FORCE" -eq 0 && "$(cat "$MESHDATA/.mesh_stamp")" == "$STAMP" \
      && "$(count_procfiles "$MESHDATA" "_Database")" -eq "$NPROC" ]]; then
  log "mesh already current -- nothing to do (--force to rebuild)"; exit 0
fi
WD="$WORKBASE/mesh"
if ! in_slurm; then
  rm -rf "$WD"; mkdir -p "$WD/DATABASES_MPI" "$WD/OUTPUT_FILES" "$WD/DATA"
  cp -rL "$ROOT/DATA/meshfem3D_files" "$WD/DATA/"; cp "$PAR" "$WD/DATA/Par_file"
  cp "$ROOT/DATA/STATIONS" "$ROOT/DATA/FORCESOLUTION" "$WD/DATA/" 2>/dev/null || true
  ln -sfn "$ROOT/bin" "$WD/bin"
  sbatch_wait "mesh" "$NODES_PER_EVENT" "$NPROC" "01:00:00" "${PART_COMPUTE:?}" -- bash "$0" --in-job \
    || die "xmeshfem3D job failed -- $LOGDIR/slurm/ and $WD/OUTPUT_FILES/output_meshfem3D.txt"
  n=$(count_procfiles "$WD/DATABASES_MPI" "_Database")
  [[ "$n" -eq "$NPROC" ]] || die "expected $NPROC proc*_Database, got $n"
  mkdir -p "$MESHDATA"; find "$MESHDATA" -maxdepth 1 -name 'proc*_Database' -delete
  find "$WD/DATABASES_MPI" -maxdepth 1 -name 'proc*_Database' -print0 | xargs -0 cp -t "$MESHDATA/"
  printf '%s\n' "$STAMP" > "$MESHDATA/.mesh_stamp"
  log "saved $n Database files -> $MESHDATA ($(du -sh "$MESHDATA" | cut -f1))"
  grep -iE 'Max element size|Min element size|suggested' "$WD/OUTPUT_FILES/output_meshfem3D.txt" | sed 's/^/    /' | tee -a "$LOGFILE"
  hr; log "MESH COMPLETE -- next: ./3_make_M0.sh"; hr; exit 0
fi
cd "$WD" || die "cannot cd $WD"
mpi_run "$NPROC" "$ROOT/bin/xmeshfem3D" || { tail -40 OUTPUT_FILES/output_meshfem3D.txt; die "xmeshfem3D FAILED"; }
