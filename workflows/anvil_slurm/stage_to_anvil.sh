#!/bin/bash
# anvil_cpu/stage_to_anvil.sh -- assemble the transfer set for Anvil (run on the 5090 box).
#
# Produces $OUT/ with tarballs + MANIFEST.sha256:
#   code.tar.gz        this anvil_cpu/ tree (no work/, log/, .state, venv, bin/)
#   specfem3d.tar.gz   trimmed SPECFEM3D 4.1.1 source (configure, src/, setup/, external_libs/)
#   tomo_M6.xyz.gz     models/M6 (2 s final = global M15) exported by tools/gll_to_tomo.py
#                      on the parent's NPROC=1 mesh -- the starting model of this stage
#   events.tar.gz      events/{s2_keep150,c1_ls40h,allforward_all,...}
#   obs_data.tar       STATIONS, exclude list, and final_sac_new (3.6 GB RAW stage-0 CC data:
#                      the new band must be produced by 0_newprocess.py, so the raw
#                      traces ship, not the 2 s-era band folders)
# Unpack on Anvil into ~/ite3_anvil (code) and $SCRATCH/wb_s3 (data) -- README step 1.
#
# Usage: ./stage_to_anvil.sh [--out DIR] [--model DIR] [--dry-run]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PARENT="$(dirname "$HERE")"
OUT="$HERE/bundle/anvil_$(date +%Y%m%d)"; MODEL="$PARENT/models/M6"; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in --out) OUT="$2"; shift ;; --model) MODEL="$2"; shift ;; --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;; *) echo "unknown option: $1" >&2; exit 1 ;; esac; shift
done
say() { printf '[stage_to_anvil] %s\n' "$*"; }
run() { if [[ "$DRY" -eq 1 ]]; then printf '   DRY: %q ' "$@"; echo; else "$@"; fi; }
[[ -d "$MODEL" ]] || { echo "model $MODEL missing" >&2; exit 1; }
PY="$PARENT/cal_adjoint/.venv/bin/python"
mkdir -p "$OUT"
say "-> $OUT"

say "code.tar.gz"
run tar -C "$PARENT" --exclude='anvil_cpu/work' --exclude='anvil_cpu/log' --exclude='anvil_cpu/.state' \
    --exclude='anvil_cpu/bin' --exclude='anvil_cpu/bundle' --exclude='anvil_cpu/cal_adjoint/.venv' \
    --exclude='anvil_cpu/cal_adjoint/log' --exclude='__pycache__' --exclude='anvil_cpu/models' \
    --exclude='anvil_cpu/DATA/tomo_files' --exclude='anvil_cpu/obs_data' --exclude='anvil_cpu/events' \
    --exclude='anvil_cpu/env.sh' -czf "$OUT/code.tar.gz" anvil_cpu

say "specfem3d.tar.gz (trimmed source)"
SRC="$PARENT/../ite3_share/specfem3d"; [[ -d "$SRC" ]] || SRC="$PARENT/specfem3d"
run tar -C "$(dirname "$SRC")" --exclude='*/obj' --exclude='*/bin' --exclude='*/EXAMPLES' --exclude='*/doc' \
    --exclude='*/utils' --exclude='*/CUBIT_GEOCUBIT' --exclude='*/.git' --exclude='*/Makefile' --exclude='*/config.log' \
    --exclude='*/config.status' --exclude='*/OUTPUT_FILES/*' --exclude='*/DATABASES_MPI' \
    -czf "$OUT/specfem3d.tar.gz" --transform "s|^$(basename "$SRC")|specfem3d|" "$(basename "$SRC")"

say "tomo export of $(basename "$MODEL") (gll_to_tomo.py, parent NPROC=1 mesh; ~10-20 min, 5.4 GB -> gz)"
if [[ "$DRY" -eq 0 ]]; then
  XYZ="$OUT/tomography_model_$(basename "$MODEL").xyz"
  if [[ ! -s "$XYZ.gz" ]]; then
    "$PY" "$PARENT/tools/gll_to_tomo.py" --model "$MODEL" --out "$XYZ" || { echo "gll_to_tomo failed" >&2; exit 1; }
    gzip -f "$XYZ"
  fi
fi

say "events.tar.gz"
run tar -C "$PARENT" -czf "$OUT/events.tar.gz" events/s2_keep150 events/c1_ls40h events/allforward_all events/s2_keep150_members.txt

say "obs_data.tar (raw stage-0 CC + STATIONS + exclude list; 3.6 GB)"
run tar -C "$PARENT" -cf "$OUT/obs_data.tar" obs_data/STATIONS obs_data/exclude_zo_line_n81.txt obs_data/final_sac_new

if [[ "$DRY" -eq 0 ]]; then
  ( cd "$OUT" && sha256sum ./* > MANIFEST.sha256 && du -sh ./* )
  say "rsync:  rsync -avP $OUT/ <user>@anvil.rcac.purdue.edu:/anvil/scratch/<user>/wb_s3/bundle/"
fi
