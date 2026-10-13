#!/bin/bash
# ite3_gpu/2_make_mesh.sh -- build the mesh at NPROC=1 (one GPU, one partition).
#
# WHY THIS MUST RUN LOCALLY, AND WHY ANVIL'S meshdata IS USELESS HERE
# -------------------------------------------------------------------
# Anvil's saved mesh is 1250 files: meshdata/proc000000_Database ... proc001249.
# That is not a file format difference, it is a DOMAIN DECOMPOSITION -- the mesh
# is physically cut into 1250 pieces, one per MPI rank. A single GPU runs one
# rank, so it needs one piece containing the whole domain. There is no tool that
# merges 1250 partitions into 1; the mesh must be regenerated. Same for the M0
# model (proc*_{vp,vs,rho}.bin), which is per-partition too -- hence 3_make_M0.sh.
#
# The good news: the mesh is small (75 x 150 x 18 = 202,500 spectral elements)
# and the mesher at NPROC=1 takes minutes, not hours. Only the geometry inputs
# (Mesh_Par_file, interfaces.dat, the tomo file) need to come from Anvil, and
# those total ~152 MB.
#
# This script:
#   1. rewrites Mesh_Par_file NPROC_XI/NPROC_ETA and Par_file NPROC to config's
#      values (and refuses if NEX is not divisible by them)
#   2. runs xmeshfem3D on one rank
#   3. saves proc000000_Database to $MESHDATA with a geometry stamp
#   4. echoes the mesher's resolution/Courant advice, which is what tells you
#      whether DT and NSTEP are still sane
#
# Idempotent: if $MESHDATA/.mesh_stamp matches the current geometry, it exits 0.
# Force with --force.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=mesh
source "$ROOT/lib/common.sh"

FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done

banner "mesh generation at NPROC=$NPROC"
take_lock

MPF="$ROOT/DATA/meshfem3D_files/Mesh_Par_file"
PAR="$ROOT/DATA/Par_file"
[[ -f "$MPF" ]] || die "missing $MPF"
[[ -f "$PAR" ]] || die "missing $PAR"
[[ -x "$ROOT/bin/xmeshfem3D" ]] || die "xmeshfem3D not built -- run ./1_build_specfem_gpu.sh"

# ---------------------------------------------------------------------------
# 1. force the partitioning to match config, with a divisibility check
# ---------------------------------------------------------------------------
[[ $((NPROC_XI * NPROC_ETA)) -eq "$NPROC" ]] \
  || die "config inconsistency: NPROC_XI($NPROC_XI) * NPROC_ETA($NPROC_ETA) != NPROC($NPROC)"

nex_xi=$(par_get "$MPF" NEX_XI)
nex_eta=$(par_get "$MPF" NEX_ETA)
[[ $((nex_xi % NPROC_XI)) -eq 0 ]]  || die "NEX_XI=$nex_xi is not a multiple of NPROC_XI=$NPROC_XI"
[[ $((nex_eta % NPROC_ETA)) -eq 0 ]] || die "NEX_ETA=$nex_eta is not a multiple of NPROC_ETA=$NPROC_ETA"

cur_xi=$(par_get "$MPF" NPROC_XI); cur_eta=$(par_get "$MPF" NPROC_ETA)
if [[ "$cur_xi" != "$NPROC_XI" || "$cur_eta" != "$NPROC_ETA" ]]; then
  cp "$MPF" "$MPF.bak_nproc${cur_xi}x${cur_eta}"
  par_set "$MPF" NPROC_XI  "$NPROC_XI"
  par_set "$MPF" NPROC_ETA "$NPROC_ETA"
  log "Mesh_Par_file: NPROC_XI/ETA ${cur_xi}/${cur_eta} -> ${NPROC_XI}/${NPROC_ETA}  (backup: $(basename "$MPF").bak_nproc${cur_xi}x${cur_eta})"
fi
cur_np=$(par_get "$PAR" NPROC)
if [[ "$cur_np" != "$NPROC" ]]; then
  par_set "$PAR" NPROC "$NPROC"
  log "Par_file: NPROC $cur_np -> $NPROC"
fi

# vertical element count: 6th field of the single region line in Mesh_Par_file
nz=$(grep -vE '^\s*#' "$MPF" | awk 'NF==7 && $7 ~ /^-?[0-9]+$/ {print $6; exit}')
[[ -n "$nz" ]] || die "could not parse NZ_END from the region line in $MPF"
STAMP="NEX_XI=$nex_xi NEX_ETA=$nex_eta NZ=$nz NPROC=$NPROC"
log "geometry: $STAMP  ($((nex_xi * nex_eta * nz)) spectral elements)"

if [[ "$nz" != "$NEX_Z" ]]; then
  warn "Mesh_Par_file NZ=$nz but config.sh NEX_Z=$NEX_Z -- update config.sh (preflight estimates use it)"
fi

# ---------------------------------------------------------------------------
# 2. idempotence gate
# ---------------------------------------------------------------------------
if [[ -f "$MESHDATA/.mesh_stamp" && "$FORCE" -eq 0 ]]; then
  have=$(cat "$MESHDATA/.mesh_stamp")
  ndb=$(count_procfiles "$MESHDATA" "_Database")
  if [[ "$have" == "$STAMP" && "$ndb" -eq "$NPROC" ]]; then
    log "mesh already current ($ndb Database file(s), stamp matches) -- nothing to do"
    log "force a rebuild with: ./2_make_mesh.sh --force"
    exit 0
  fi
  log "stamp differs (have: $have) or Database count $ndb != $NPROC -- remeshing"
fi

# ---------------------------------------------------------------------------
# 3. run the mesher
# ---------------------------------------------------------------------------
WD="$WORKBASE/mesh"
rm -rf "$WD"
mkdir -p "$WD/DATABASES_MPI" "$WD/OUTPUT_FILES" "$WD/DATA"
cp -rL "$ROOT/DATA/meshfem3D_files" "$WD/DATA/"
cp "$PAR" "$WD/DATA/Par_file"
cp "$ROOT/DATA/STATIONS" "$WD/DATA/" 2>/dev/null || true
[[ -f "$ROOT/DATA/FORCESOLUTION" ]] && cp "$ROOT/DATA/FORCESOLUTION" "$WD/DATA/"
ln -sfn "$ROOT/bin" "$WD/bin"

log "mesher workdir: $WD"
cd "$WD" || die "cannot cd $WD"

t0=$(date +%s)
if ! gpu_run "$NPROC" "$ROOT/bin/xmeshfem3D" >>"$LOGFILE" 2>&1; then
  log "--- last 40 lines of output_meshfem3D.txt ---"
  tail -40 OUTPUT_FILES/output_meshfem3D.txt 2>/dev/null | tee -a "$LOGFILE"
  die "xmeshfem3D FAILED -- see $LOGFILE and $WD/OUTPUT_FILES/"
fi
t1=$(date +%s)
log "xmeshfem3D finished in $(human_time $((t1 - t0)))"

n=$(count_procfiles "$WD/DATABASES_MPI" "_Database")
[[ "$n" -eq "$NPROC" ]] || die "expected $NPROC proc*_Database, got $n -- check $WD/OUTPUT_FILES/output_meshfem3D.txt"

# ---------------------------------------------------------------------------
# 4. save + stamp
# ---------------------------------------------------------------------------
mkdir -p "$MESHDATA"
rm -f "$MESHDATA"/proc*_Database
cp "$WD/DATABASES_MPI"/proc*_Database "$MESHDATA/"
printf '%s\n' "$STAMP" > "$MESHDATA/.mesh_stamp"
log "saved $n Database file(s) -> $MESHDATA  ($(du -sh "$MESHDATA" | cut -f1))"

# The mesher prints the estimated resolvable period, suggested time step and
# Courant number. If DT or NSTEP drifted out of range this is where you find out.
log ""
log "--- mesher resolution report (check DT=$DT, NSTEP=$NSTEP against this) ---"
grep -iE 'resolved|suggested|time step|frequency|Courant|element size|maximum number' \
  "$WD/OUTPUT_FILES/output_meshfem3D.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$LOGFILE"

hr
log "MESH COMPLETE"
log "next:  ./3_make_M0.sh"
hr
