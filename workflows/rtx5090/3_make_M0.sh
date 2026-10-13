#!/bin/bash
# ite3_gpu/3_make_M0.sh -- bake the tomography file onto the NPROC=1 mesh -> M0.
#
# Anvil's Modelfile_ite3/M0 is 1250 per-partition files and cannot be reused at
# NPROC=1 (see the note in 2_make_mesh.sh). M0 is rebuilt here from the SAME
# tomography_model.xyz, so the starting model is numerically the same field,
# just sampled onto a single-partition mesh.
#
# Mechanism (identical to ite3's 0_make_M0.sh):
#   MODEL=tomo + SAVE_MESH_FILES=.true.  ->  xgenerate_databases interpolates the
#   tomo grid onto every GLL point and, because SAVE_MESH_FILES is on, writes the
#   result out as proc*_{vp,vs,rho}.bin. Those files ARE the GLL model. Harvest
#   them, then restore MODEL=gll so every later run reads the model instead of
#   re-interpolating.
#
# Three guards, all carried over from ite3 because each one caught a real failure:
#   1. TOMO Z-UNITS. ite2 lost an entire inversion to a tomo file whose z column
#      was in kilometres while x/y were in metres. SPECFEM reads z as metres, so
#      the whole 30 km mesh sat below the grid and every GLL point was clamped to
#      the bottom layer (100 km depth, mantle, Vs ~ 4.3 km/s). Refuse to build.
#   2. MESH STAMP. xgenerate_databases takes geometry from proc*_Database, not
#      from Mesh_Par_file -- so editing the latter without remeshing silently
#      does nothing. Refuse if the stamp disagrees.
#   3. CRUSTAL VS ACCEPTANCE. After the build, sample the actual Vs and fail if
#      the median looks like mantle. This is the check that would have caught the
#      ite2 bug on day one.
#
# Idempotent via $M0/.done. Force with --force.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=make_M0
source "$ROOT/lib/common.sh"

FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done

M0="$MODELBASE/M0"
TOMO="$ROOT/DATA/tomo_files/tomography_model.xyz"
MPF="$ROOT/DATA/meshfem3D_files/Mesh_Par_file"
PAR="$ROOT/DATA/Par_file"

banner "build M0 from tomography (NPROC=$NPROC)"
take_lock

[[ -f "$TOMO" ]] || die "tomography file not found: $TOMO"
[[ -x "$ROOT/bin/xgenerate_databases" ]] || die "xgenerate_databases not built -- run ./1_build_specfem_gpu.sh"

# ---------------------------------------------------------------------------
# guard 0: already done?
# ---------------------------------------------------------------------------
if [[ -f "$M0/.done" && "$FORCE" -eq 0 ]]; then
  n=$(count_procfiles "$M0" "_vp.bin")
  log "M0 already built ($n vp file(s), .done present) -- nothing to do"
  log "force a rebuild with: ./3_make_M0.sh --force"
  exit 0
fi

# ---------------------------------------------------------------------------
# guard 1: tomo z units must be METRES
# ---------------------------------------------------------------------------
log "checking tomography z units..."
# Read z_min from the HEADER (line 1: ORIG_X ORIG_Y ORIG_Z END_X END_Y END_Z) --
# it IS the grid definition SPECFEM reads, and unlike a capped data-row scan it
# is independent of grid density. (The old first-500k-rows scan broke on fine
# grids: at 300x570 nodes/slab, 500k rows only reaches z=-501 m, which then
# looks like kilometres and false-trips this guard.)
zmin=$(awk 'NR==1{ printf "%.0f", ($3 < $6 ? $3 : $6); exit }' "$TOMO")
if [[ -z "$zmin" ]]; then
  die "could not parse a z column from $TOMO"
elif [[ "$zmin" -gt -1000 ]]; then
  die "tomo z_min=$zmin -- that is KILOMETRES, not metres.
     x/y are in metres, so SPECFEM would place the whole mesh below the grid and
     clamp every point to the bottom layer. This is exactly the bug that
     invalidated ite2's M0-M3. Rescale the z column to metres before continuing."
fi
log "  z_min=${zmin} m -- units look correct"

# ---------------------------------------------------------------------------
# guard 2: mesh stamp
# ---------------------------------------------------------------------------
nex_xi=$(par_get "$MPF" NEX_XI); nex_eta=$(par_get "$MPF" NEX_ETA)
nz=$(grep -vE '^\s*#' "$MPF" | awk 'NF==7 && $7 ~ /^-?[0-9]+$/ {print $6; exit}')
WANT="NEX_XI=$nex_xi NEX_ETA=$nex_eta NZ=$nz NPROC=$NPROC"
[[ -f "$MESHDATA/.mesh_stamp" ]] || die "no mesh stamp at $MESHDATA/.mesh_stamp -- run ./2_make_mesh.sh first"
HAVE=$(cat "$MESHDATA/.mesh_stamp")
[[ "$HAVE" == "$WANT" ]] || die "mesh stamp mismatch.
     mesh was built as : $HAVE
     Mesh_Par_file says: $WANT
     Run ./2_make_mesh.sh --force, then retry."
ndb=$(count_procfiles "$MESHDATA" "_Database")
[[ "$ndb" -eq "$NPROC" ]] || die "expected $NPROC proc*_Database in $MESHDATA, found $ndb"
log "  mesh stamp OK: $HAVE"

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
# Clear .done FIRST so an interrupted build can never leave a half-written M0
# marked complete.
rm -f "$M0/.done"
mkdir -p "$M0"

WD="$WORKBASE/genmesh_M0"
rm -rf "$WD"
mkdir -p "$WD/DATABASES_MPI" "$WD/OUTPUT_FILES" "$WD/DATA"
cp -rL "$ROOT/DATA/meshfem3D_files" "$WD/DATA/"
cp "$ROOT/DATA/STATIONS" "$WD/DATA/" 2>/dev/null || true
[[ -f "$ROOT/DATA/FORCESOLUTION" ]] && cp "$ROOT/DATA/FORCESOLUTION" "$WD/DATA/"
# tomo_files is read-only input; symlink rather than copy the 152 MB file
ln -sfn "$ROOT/DATA/tomo_files" "$WD/DATA/tomo_files"
ln -sfn "$ROOT/bin" "$WD/bin"
ln -sf "$MESHDATA"/proc*_Database "$WD/DATABASES_MPI/"

# A LOCAL Par_file set to tomo mode. The shared $ROOT/DATA/Par_file is never
# left in tomo mode -- that was a real hazard in the Anvil version, where an
# interrupted run could leave MODEL=tomo behind and confuse the next stage.
cp "$PAR" "$WD/DATA/Par_file"
par_set "$WD/DATA/Par_file" MODEL tomo
par_set "$WD/DATA/Par_file" SAVE_MESH_FILES .true.
par_set "$WD/DATA/Par_file" TOMOGRAPHY_PATH ./DATA/tomo_files/
log "workdir $WD  (MODEL=tomo, SAVE_MESH_FILES=.true.)"

cd "$WD" || die "cannot cd $WD"
t0=$(date +%s)
if ! gpu_run "$NPROC" "$ROOT/bin/xgenerate_databases" >>"$LOGFILE" 2>&1; then
  log "--- last 40 lines of output_generate_databases.txt ---"
  tail -40 OUTPUT_FILES/output_generate_databases.txt 2>/dev/null | tee -a "$LOGFILE"
  die "xgenerate_databases FAILED -- see $LOGFILE and $WD/OUTPUT_FILES/"
fi
t1=$(date +%s)
log "xgenerate_databases finished in $(human_time $((t1 - t0)))"

# ---------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------
for p in vp vs rho; do
  n=$(count_procfiles "$WD/DATABASES_MPI" "_$p.bin")
  [[ "$n" -eq "$NPROC" ]] || die "expected $NPROC proc*_$p.bin, got $n -- did SAVE_MESH_FILES take effect?"
done
cp "$WD/DATABASES_MPI"/proc*_vp.bin "$WD/DATABASES_MPI"/proc*_vs.bin "$WD/DATABASES_MPI"/proc*_rho.bin "$M0/"
log "harvested $((NPROC * 3)) model file(s) -> $M0  ($(du -sh "$M0" | cut -f1))"

# Keep the mesh too: every later stage needs proc*_external_mesh.bin for THIS
# model, and regenerating it costs another genmesh run.
mkdir -p "$MODELBASE/mesh_M0"
cp "$WD/DATABASES_MPI"/proc*_external_mesh.bin "$MODELBASE/mesh_M0/" 2>/dev/null || true
cp "$WD/OUTPUT_FILES/values_from_mesher.h" "$MODELBASE/mesh_M0/" 2>/dev/null || true

# ---------------------------------------------------------------------------
# guard 3: crustal Vs acceptance test
# ---------------------------------------------------------------------------
log "acceptance test: sampling Vs..."
python3 - "$M0" <<'PY' | tee -a "$LOGFILE"
import sys, glob, array, statistics, os
m0 = sys.argv[1]
vals = []
for f in sorted(glob.glob(os.path.join(m0, "proc*_vs.bin")))[:30]:
    with open(f, "rb") as fh:
        raw = fh.read()
    # Fortran sequential-unformatted: [int32 nbytes][data][int32 nbytes]
    a = array.array("f")
    a.frombytes(raw[4:len(raw) - 4 - ((len(raw) - 8) % 4)])
    vals.extend(v for v in a if 100.0 < v < 10000.0)
if not vals:
    print("  ACCEPT-FAIL: no finite Vs values read"); sys.exit(1)
vals.sort()
med = statistics.median(vals)
p10 = vals[len(vals)//10]; p90 = vals[9*len(vals)//10]
print(f"  Vs samples={len(vals)}  p10={p10:.0f}  median={med:.0f}  p90={p90:.0f} m/s")
if med > 4100:
    print(f"  ACCEPT-FAIL: median Vs {med:.0f} m/s looks like MANTLE, not crust.")
    print("               This is the signature of the km/m tomo bug.")
    sys.exit(1)
print(f"  ACCEPT-OK: median Vs {med:.0f} m/s is crustal")
PY
rc=${PIPESTATUS[0]}
[[ "$rc" -eq 0 ]] || die "M0 acceptance test FAILED -- M0 left WITHOUT a .done sentinel (will not be used)"

: > "$M0/.done"
log "M0 accepted -> $M0/.done"

hr
log "M0 COMPLETE"
log "next:  ./4_bench.sh   (time one event before launching the full loop)"
hr
