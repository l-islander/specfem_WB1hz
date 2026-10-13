#!/bin/bash
# anvil_cpu/3_make_M0.sh -- bake DATA/tomo_files/tomography_model.xyz onto the new mesh -> M0.
#
# The starting model of this stage is the 2 s campaign's FINAL model (models/M6
# = global M15), exported to a tomography file on the 5090 box by
# tools/gll_to_tomo.py (stage_to_anvil.sh does it, 438 m grid, 5.4 GB) and
# shipped here. MODEL=tomo + SAVE_MESH_FILES=.true. makes xgenerate_databases
# write the interpolated GLL model; harvest it, done. The three ite3 guards
# (tomo z in metres, mesh stamp, crustal-Vs acceptance) are kept. One SLURM job
# at NPROC ranks; outside SLURM the script submits itself.
#
# Usage: ./3_make_M0.sh [--force]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=make_M0; source "$ROOT/lib/common.sh"; source "$ROOT/lib/slurm.sh"
FORCE=0; [[ "${1:-}" == "--force" ]] && FORCE=1
M0="$MODELBASE/M0"; TOMO="$ROOT/DATA/tomo_files/tomography_model.xyz"
MPF="$ROOT/DATA/meshfem3D_files/Mesh_Par_file"; PAR="$ROOT/DATA/Par_file"; WD="$WORKBASE/genmesh_M0"

if in_slurm && [[ "${1:-}" == "--in-job" ]]; then
  cd "$WD" || die "cannot cd $WD"
  mpi_run "$NPROC" "$ROOT/bin/xgenerate_databases" || { tail -40 OUTPUT_FILES/output_generate_databases.txt; die "xgenerate_databases FAILED"; }
  exit 0
fi
banner "build M0 from tomography (NPROC=$NPROC)"
[[ -f "$TOMO" ]] || die "tomography file not found: $TOMO"
if [[ -f "$M0/.done" && "$FORCE" -eq 0 ]]; then log "M0 already built -- nothing to do (--force to rebuild)"; exit 0; fi
zmin=$(awk 'NR==1{ printf "%.0f", ($3 < $6 ? $3 : $6); exit }' "$TOMO")
[[ -n "$zmin" && "$zmin" -le -1000 ]] || die "tomo z_min=${zmin:-?} -- that is KILOMETRES, not metres (the ite2 bug). Refusing."
nex_xi=$(par_get "$MPF" NEX_XI); nex_eta=$(par_get "$MPF" NEX_ETA)
nz=$(grep -vE '^\s*#' "$MPF" | awk 'NF==7 && $7 ~ /^-?[0-9]+$/ {print $6; exit}')
WANT="NEX_XI=$nex_xi NEX_ETA=$nex_eta NZ=$nz NPROC=$NPROC"
[[ -f "$MESHDATA/.mesh_stamp" ]] || die "no mesh stamp -- ./2_make_mesh.sh first"
[[ "$(cat "$MESHDATA/.mesh_stamp")" == "$WANT" ]] || die "mesh stamp mismatch: have '$(cat "$MESHDATA/.mesh_stamp")' want '$WANT' -- ./2_make_mesh.sh --force"
log "  z_min=${zmin} m OK; mesh stamp OK"

rm -f "$M0/.done"; mkdir -p "$M0"
rm -rf "$WD"; mkdir -p "$WD/DATABASES_MPI" "$WD/OUTPUT_FILES" "$WD/DATA"
cp -rL "$ROOT/DATA/meshfem3D_files" "$WD/DATA/"
cp "$ROOT/DATA/STATIONS" "$ROOT/DATA/FORCESOLUTION" "$WD/DATA/" 2>/dev/null || true
ln -sfn "$ROOT/DATA/tomo_files" "$WD/DATA/tomo_files"; ln -sfn "$ROOT/bin" "$WD/bin"
find "$MESHDATA" -maxdepth 1 -name 'proc*_Database' -print0 | xargs -0 ln -sf -t "$WD/DATABASES_MPI/"
cp "$PAR" "$WD/DATA/Par_file"
par_set "$WD/DATA/Par_file" NPROC "$NPROC"; par_set "$WD/DATA/Par_file" GPU_MODE .false.
par_set "$WD/DATA/Par_file" MODEL tomo; par_set "$WD/DATA/Par_file" SAVE_MESH_FILES .true.
par_set "$WD/DATA/Par_file" TOMOGRAPHY_PATH ./DATA/tomo_files/
par_set "$WD/DATA/Par_file" NSTEP "$NSTEP"; par_set "$WD/DATA/Par_file" DT "$DT"
sbatch_wait "make_M0" "$NODES_PER_EVENT" "$NPROC" "02:00:00" "${PART_COMPUTE:?}" -- bash "$0" --in-job \
  || die "M0 bake job failed -- $LOGDIR/slurm/"
for p in vp vs rho; do
  n=$(count_procfiles "$WD/DATABASES_MPI" "_$p.bin"); [[ "$n" -eq "$NPROC" ]] || die "expected $NPROC proc*_$p.bin, got $n"
done
find "$WD/DATABASES_MPI" -maxdepth 1 \( -name 'proc*_vp.bin' -o -name 'proc*_vs.bin' -o -name 'proc*_rho.bin' \) -print0 | xargs -0 cp -t "$M0/"
log "harvested $((NPROC * 3)) model files -> $M0 ($(du -sh "$M0" | cut -f1))"
grep -E 'Minimum period resolved|Maximum suggested time step|Max stability' "$WD/OUTPUT_FILES/output_generate_databases.txt" | sed 's/^/    /' | tee -a "$LOGFILE"
resolve_python
"$PYBIN" - "$M0" <<'PY' | tee -a "$LOGFILE"
import sys, glob, os, numpy as np
m0 = sys.argv[1]; vals = []
for f in sorted(glob.glob(os.path.join(m0, "proc*_vs.bin")))[::max(1, len(glob.glob(os.path.join(m0, "proc*_vs.bin")))//40)]:
    raw = open(f, "rb").read(); a = np.frombuffer(raw[4:4 + (len(raw) - 8)//4*4], dtype=np.float32)
    vals.append(a[(a > 100) & (a < 10000)])
v = np.concatenate(vals); med = float(np.median(v))
print(f"  Vs samples={v.size}  p10={np.percentile(v,10):.0f}  median={med:.0f}  p90={np.percentile(v,90):.0f} m/s")
sys.exit(1 if med > 4100 else 0)
PY
[[ "${PIPESTATUS[0]}" -eq 0 ]] || die "M0 acceptance FAILED (median Vs looks like mantle) -- no .done written"
: > "$M0/.done"; log "M0 accepted -> $M0/.done"
hr; log "M0 COMPLETE -- next: ./4_bench.sh"; hr
