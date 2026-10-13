#!/bin/bash
# anvil_cpu/cal_adjoint/build_env_anvil.sh -- the Python venv on Anvil. Run ONCE on a login node.
#
# Same as ite3_gpu's build_env_local.sh with the modules loaded first (env.sh)
# and the multi-rank check run under srun in a 2-rank debug allocation, because
# mpirun on a login node is not the launcher the jobs will use. The whole point
# is mpi4py compiled against the SAME OpenMPI the solver uses: a mismatched
# wheel makes every rank think it is rank 0 of 1 and the misfit comes out
# wrong and LOW with no error.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
cd "$HERE"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
UV="${UV_BIN:-uv}"
command -v "$UV" >/dev/null 2>&1 || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
MPICC_BIN="$(command -v "${MPICC:-mpicc}" || true)"
[[ -n "$MPICC_BIN" ]] || { echo "mpicc not found -- check the module loads in env.sh" >&2; exit 1; }
export MPICC="$MPICC_BIN"
echo "==> MPICC=$MPICC  ($(mpicc --version 2>/dev/null | head -1))"
# uv's cache and the venv must not live on $HOME's 25 GB if several are built; fine for one.
"$UV" python install 3.11 >/dev/null 2>&1 || true
"$UV" sync --python 3.11
"$UV" pip install --reinstall-package mpi4py --no-binary mpi4py mpi4py
"$UV" run python - <<'PY'
from mpi4py import MPI
print("  mpi4py :", MPI.Get_library_version().splitlines()[0])
import numpy, scipy, obspy, pandas
print("  numpy  :", numpy.__version__, " scipy:", scipy.__version__, " obspy:", obspy.__version__)
PY
echo "==> multi-rank check under srun (2 ranks, debug partition)"
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  srun --account="$SLURM_ACCOUNT" --partition="${PART_DEBUG:-debug}" -N1 -n2 -t 00:02:00 ${MPIRUN_EXTRA:-} \
    "$HERE/.venv/bin/python" -c "from mpi4py import MPI; import sys; c=MPI.COMM_WORLD; c.rank==0 and sys.stdout.write(f'  world size = {c.size}\n')" \
    || echo "  WARNING: srun test failed -- run 0_preflight.sh once the account/partition names are right" >&2
else
  echo "  (SLURM_ACCOUNT unset -- skipping; 0_preflight.sh repeats this test)"
fi
echo "venv ready: $HERE/.venv"
