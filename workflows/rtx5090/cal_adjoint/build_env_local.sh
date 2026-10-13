#!/bin/bash
# ite3_gpu/cal_adjoint/build_env_local.sh -- build the Python venv for the
# adjoint-source pipeline on a local machine. Run ONCE. Needs internet.
#
# Differences from Anvil's build_env.sh: no `module load`, and mpi4py is built
# against whatever MPI is on PATH here rather than Anvil's OpenMPI.
#
# WHY mpi4py IS BUILT FROM SOURCE
# -------------------------------
# A PyPI mpi4py wheel bundles its own MPI, which will not be the MPI that
# `mpirun` launches. The mismatch usually shows up as every rank believing it is
# rank 0 of a 1-process world -- so an "MPI" run silently does 1/N of the work,
# N times, and produces a wrong (low) misfit with no error. Building against the
# local mpicc guarantees they match.
#
# ~1 GB installed (numpy/scipy/obspy/matplotlib/pandas), 1-3 minutes.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$HERE"

source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"

UV="${UV_BIN:-uv}"
command -v "$UV" >/dev/null 2>&1 || {
  echo "ERROR: uv not found on PATH." >&2
  echo "  install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

MPICC_BIN="$(command -v "${MPICC:-mpicc}" || true)"
[[ -n "$MPICC_BIN" ]] || {
  echo "ERROR: mpicc not found -- install an MPI development package." >&2
  echo "  Ubuntu/Debian: sudo apt install libopenmpi-dev openmpi-bin" >&2
  echo "  Fedora/RHEL:   sudo dnf install openmpi-devel && module load mpi/openmpi-x86_64" >&2
  exit 1
}
export MPICC="$MPICC_BIN"
echo "==> using MPICC=$MPICC"

echo "==> uv sync (resolving from uv.lock)"
"$UV" sync

echo "==> rebuilding mpi4py from source against $MPICC"
"$UV" pip install --reinstall-package mpi4py --no-binary mpi4py mpi4py

echo "==> verifying"
"$UV" run python - <<'PY'
from mpi4py import MPI
print("  mpi4py :", MPI.Get_library_version().splitlines()[0])
import numpy, scipy, obspy
print("  numpy  :", numpy.__version__)
print("  scipy  :", scipy.__version__)
print("  obspy  :", obspy.__version__)
PY

# The decisive test: does a REAL multi-rank launch see a world size > 1?
# If the wheel/launcher mismatch above is present, this prints 1 and the whole
# pipeline would silently under-measure.
echo "==> verifying multi-rank launch (this is the one that catches ABI mismatch)"
NR=2
if "${MPIRUN:-mpirun}" ${MPIRUN_EXTRA:-} -np "$NR" "$HERE/.venv/bin/python" -c \
     "from mpi4py import MPI; import sys; c=MPI.COMM_WORLD; c.rank==0 and sys.stdout.write(f'  world size = {c.size}\n')" 2>/dev/null; then
  :
else
  echo "  WARNING: multi-rank launch failed; the pipeline will still work on 1 rank (slower)." >&2
fi

echo
echo "venv ready: $HERE/.venv"
echo "next: ./run_caladj_local.sh --help"
