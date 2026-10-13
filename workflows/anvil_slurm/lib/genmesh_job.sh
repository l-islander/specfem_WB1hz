#!/bin/bash
# anvil_cpu/lib/genmesh_job.sh <db_dir> -- runs INSIDE a SLURM job: xgenerate_databases at NPROC.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=genmesh; source "$ROOT/lib/common.sh"; source "$ROOT/lib/solver.sh"
db="${1:?db dir}"
[[ -d "$db/DATABASES_MPI" ]] || die "no $db/DATABASES_MPI"
t0=$(date +%s)
( cd "$db" && mpi_run "$NPROC" "$ROOT/bin/xgenerate_databases" ) || {
  tail -30 "$db/OUTPUT_FILES/output_generate_databases.txt" 2>/dev/null
  die "xgenerate_databases FAILED in $db"
}
log "genmesh job: $(human_time $(( $(date +%s) - t0 )))"
grep -E 'Minimum period resolved|Maximum suggested time step|Max stability|Max element size|Min element size' \
  "$db/OUTPUT_FILES/output_generate_databases.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$LOGFILE"
