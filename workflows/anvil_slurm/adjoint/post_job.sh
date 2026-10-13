#!/bin/bash
# anvil_cpu/adjoint/post_job.sh <db_dir> [--force] -- INSIDE a 1-node job: clip_iso.py + sum_precond.py.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=post_job; source "$ROOT/lib/common.sh"
DB="${1:?db dir}"; FORCE=0; [[ "${2:-}" == "--force" ]] && FORCE=1
ADJ_WORK="$WORKBASE/adjoint"
NJOBS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
resolve_python
[[ "$FORCE" -eq 1 ]] && { clear_done post_clip; clear_done post_sum; }
cd "$ADJ_WORK" || die "cannot cd $ADJ_WORK"      # clip_iso.py's receiver-list glob is relative
mask_args=()
if [[ "$MASK" == "1" ]]; then
  mask_args=( --mask --topo-dir "$DB/DATABASES_MPI" --sigma-h "$SIGMA_H" --sigma-v "$SIGMA_V" )
fi
mkdir -p "$KERNELS"
for ev in "$ORI_KERNELS"/*; do [[ -e "$ev" ]] && mkdir -p "$KERNELS/$(basename "$ev")"; done
step post_clip "clip kernels to +-$CLIP_VALUE + receiver mask ($NJOBS workers)" -- \
  "$PYBIN" "$HERE/clip_iso.py" --clip "$CLIP_VALUE" --ori "$ORI_KERNELS" --out "$KERNELS" \
    --nproc "$NPROC" --jobs "$NJOBS" "${mask_args[@]}" || die "clip stage failed"
step post_sum "sum kernels, precondition by the approximate Hessian" -- \
  "$PYBIN" "$HERE/sum_precond.py" --input "$KERNELS" --hess-input "$ORI_KERNELS" \
    --out "$KERNEL_SUM" --nproc "$NPROC" --jobs "$NJOBS" || die "sum stage failed"
