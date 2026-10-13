#!/bin/bash
# anvil_cpu/adjoint/smooth_job.sh <db_dir> <condi_dir> <smooth_dir> -- INSIDE an NPROC-rank job.
# xsmooth_sem reads the mesh from ./DATABASES_MPI, so it runs in the model's DB dir.
# CPU path: no 6th (GPU) argument. Output names: <kernel>_smooth.bin per proc.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=smooth_job; source "$ROOT/lib/common.sh"
DB="${1:?}"; CONDI="${2:?}"; SMOOTH="${3:?}"
mkdir -p "$SMOOTH"
t0=$(date +%s)
if [[ "${SMOOTH_COMBINED:-1}" -eq 1 ]]; then
  klist=$(tr ' ' ',' <<<"$SMOOTH_KERNELS")
  ( cd "$DB" && mpi_run "$NPROC" "$ROOT/bin/xsmooth_sem" "$SMOOTH_SIGMA_H" "$SMOOTH_SIGMA_V" "$klist" "$CONDI/" "$SMOOTH/" ) \
    || die "xsmooth_sem failed on '$klist' (try SMOOTH_COMBINED=0 in config.sh)"
else
  for k in $SMOOTH_KERNELS; do
    ( cd "$DB" && mpi_run "$NPROC" "$ROOT/bin/xsmooth_sem" "$SMOOTH_SIGMA_H" "$SMOOTH_SIGMA_V" "$k" "$CONDI/" "$SMOOTH/" ) \
      || die "xsmooth_sem failed on $k"
  done
fi
log "smoothing finished in $(human_time $(( $(date +%s) - t0 )))"
