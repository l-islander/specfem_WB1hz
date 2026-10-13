#!/bin/bash
# anvil_cpu/0_preflight.sh -- is this Anvil login session ready? Read-only.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=preflight; source "$ROOT/lib/common.sh"
ok() { printf '  [ok]   %s\n' "$*"; }; bad() { printf '  [FAIL] %s\n' "$*"; FAILS=$((FAILS+1)); }; wr() { printf '  [warn] %s\n' "$*"; }
FAILS=0
banner "preflight: anvil_cpu  plan=${MESH_TARGET_S}s NPROC=$NPROC ($NODES_PER_EVENT nodes/event) x $ARRAY_THROTTLE"
[[ -r "$ROOT/env.sh" ]] && ok "env.sh present" || bad "no env.sh -- cp env.sh.anvil env.sh and edit [EDIT] lines"
for c in sbatch squeue sacct; do command -v $c >/dev/null 2>&1 && ok "$c" || bad "$c not on PATH"; done
[[ -n "${SLURM_ACCOUNT:-}" ]] && ok "SLURM_ACCOUNT=$SLURM_ACCOUNT" || bad "SLURM_ACCOUNT empty"
if command -v sacctmgr >/dev/null 2>&1 && [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  sacctmgr -n show assoc user="$USER" format=account 2>/dev/null | grep -qw "$SLURM_ACCOUNT" && ok "account association exists" || wr "sacctmgr does not list $SLURM_ACCOUNT for $USER"
fi
command -v sinfo >/dev/null 2>&1 && for p in "${PART_COMPUTE:-}" "${PART_DEBUG:-}" "${PART_HIGHMEM:-}"; do
  [[ -z "$p" ]] && continue; sinfo -h -p "$p" >/dev/null 2>&1 && ok "partition $p" || wr "partition $p unknown to sinfo"
done
for c in "${MPIFC:-mpif90}" "${MPICC:-mpicc}" "${FC:-gfortran}"; do command -v "$c" >/dev/null 2>&1 && ok "$c: $(command -v "$c")" || bad "$c not found (module loads in env.sh)"; done
for b in xmeshfem3D xgenerate_databases xspecfem3D xsmooth_sem; do [[ -x "$ROOT/bin/$b" ]] && ok "bin/$b" || wr "bin/$b missing -- ./1_build_specfem_cpu.sh"; done
[[ -x "$ROOT/bin/xspecfem3D" ]] && { ldd "$ROOT/bin/xspecfem3D" 2>/dev/null | grep -q cudart && wr "xspecfem3D links CUDA -- this should be a CPU build" || ok "xspecfem3D is a CPU binary"; }
[[ -x "$ROOT/cal_adjoint/.venv/bin/python" ]] && ok "cal_adjoint venv" || wr "venv missing -- cal_adjoint/build_env_anvil.sh"
[[ -d "$ROOT/DATA/tomo_files" && -f "$ROOT/DATA/tomo_files/tomography_model.xyz" ]] && ok "tomo file" || wr "DATA/tomo_files/tomography_model.xyz missing (stage_to_anvil bundle)"
[[ -f "$ROOT/DATA/meshfem3D_files/Mesh_Par_file" ]] && ok "Mesh_Par_file (NEX $(par_get "$ROOT/DATA/meshfem3D_files/Mesh_Par_file" NEX_XI) x $(par_get "$ROOT/DATA/meshfem3D_files/Mesh_Par_file" NEX_ETA))" || wr "no Mesh_Par_file -- tools/plan_mesh.py <period> --apply"
[[ -f "$ROOT/mesh_plan.sh" ]] && ok "mesh_plan.sh: $(grep -E '^MESH_TARGET_S|^NPROC=' "$ROOT/mesh_plan.sh" | tr '\n' ' ')" || wr "mesh_plan.sh absent (config defaults in use)"
[[ $((NPROC % CORES_PER_NODE)) -eq 0 ]] && ok "NPROC=$NPROC is whole nodes" || wr "NPROC=$NPROC is not a multiple of $CORES_PER_NODE cores -- partial nodes are still charged whole"
[[ $((NEX_XI % NPROC_XI)) -eq 0 && $((NEX_ETA % NPROC_ETA)) -eq 0 ]] && ok "NEX divisible by decomposition" || bad "NEX $NEX_XI x $NEX_ETA not divisible by $NPROC_XI x $NPROC_ETA"
mkdir -p "$WORKBASE" 2>/dev/null && ok "WORKBASE=$WORKBASE ($(free_disk_gb "$WORKBASE") GB free)" || bad "cannot create WORKBASE=$WORKBASE"
case "$WORKBASE" in /anvil/scratch/*|/scratch/*|"$ROOT"/work*) ;; *) wr "WORKBASE is not on scratch" ;; esac
[[ "$WORKBASE" == "$HOME"* ]] && wr "WORKBASE under \$HOME (25 GB quota) -- point it at scratch in env.sh"
NEED=$(( ARRAY_THROTTLE * ${WAVEFIELD_GB_EST:-200} + 150 * (NEX_XI * NEX_ETA * NEX_Z * 125 * 4 * 8 / 1073741824 + 1) ))
[[ "$(free_disk_gb "$WORKBASE")" -ge "$NEED" ]] && ok "scratch headroom >= ${NEED} GB for one iteration" || wr "scratch headroom < ${NEED} GB (throttle x wavefield + 2 x kernels)"
command -v myquota >/dev/null 2>&1 && myquota 2>/dev/null | sed 's/^/     /' | head -8
for d in "$SRC_SET" "$LS_SUBSET" "$CALADJ_DATA"; do [[ -d "$d" ]] && ok "$(basename "$d") ($(find -L "$d" -mindepth 1 -maxdepth 1 | wc -l) entries)" || wr "missing $d"; done
[[ -d "$CALADJ_DATA/final_sac_new" ]] && ok "raw stage-0 data present ($(find "$CALADJ_DATA/final_sac_new" -name '*.sac' | wc -l) files)" || wr "no obs_data/final_sac_new -- stage 0 cannot make the new band"
log "band code for DT=$DT: $(band_code); CALADJ_BANDS=$CALADJ_BANDS"
hr
[[ "$FAILS" -eq 0 ]] && log "PREFLIGHT OK" || die "$FAILS hard failure(s)"
