#!/bin/bash
# ite3_gpu/0_preflight.sh -- validate this machine BEFORE spending a night on a build.
#
# Checks, in order of how expensive the failure is to discover later:
#   1. GPU present, driver version, compute capability, free VRAM
#   2. CUDA toolkit >= 12.8 (sm_120 needs it) and nvcc/host-compiler compatibility
#   3. MPI + Fortran/C compilers
#   4. Host RAM and free disk against this problem's real requirements
#   5. Input data staged (mesh params, tomo file, event set, observed data)
#   6. Python/uv for cal_adjoint
#
# Exit 0 = ready to build. Exit 1 = something blocking. Warnings do not fail.
# Nothing here writes outside the tree; it is safe to run repeatedly.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=preflight
source "$ROOT/lib/common.sh"

FAIL=0
PASS=0
WARNC=0
ok()   { printf '  \033[32m[ ok ]\033[0m %s\n' "$*"   | tee -a "$LOGFILE"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"   | tee -a "$LOGFILE"; FAIL=$((FAIL+1)); }
note() { printf '  \033[33m[warn]\033[0m %s\n' "$*"   | tee -a "$LOGFILE"; WARNC=$((WARNC+1)); }
info() { printf '         %s\n' "$*"                  | tee -a "$LOGFILE"; }

banner "ite3_gpu preflight -- $(hostname) -- $(date '+%F %T')"

# ---------------------------------------------------------------------------
# 1. GPU
# ---------------------------------------------------------------------------
log ""; log "== 1. GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "nvidia-smi not found -- no NVIDIA driver installed?"
else
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  GNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n "$((GPU_DEVICE+1))p")
  GMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sed -n "$((GPU_DEVICE+1))p")
  GFREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sed -n "$((GPU_DEVICE+1))p")
  GCC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n "$((GPU_DEVICE+1))p")
  NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

  if [[ -z "$GNAME" ]]; then
    bad "no GPU at index GPU_DEVICE=$GPU_DEVICE (found $NGPU GPU(s))"
  else
    ok "GPU $GPU_DEVICE: $GNAME  (driver $DRV, ${GMEM} MiB total, ${GFREE} MiB free)"
    [[ "$NGPU" -gt 1 ]] && info "$NGPU GPUs present; this workflow uses only device $GPU_DEVICE (NPROC=1)."

    # compute capability -> the -gencode this build must use
    if [[ -n "$GCC" ]]; then
      CCNUM=${GCC/./}
      if [[ "$CCNUM" == "$GPU_ARCH" ]]; then
        ok "compute capability $GCC matches config GPU_ARCH=$GPU_ARCH"
      else
        bad "compute capability is $GCC (=> GPU_ARCH=$CCNUM) but config.sh has GPU_ARCH=$GPU_ARCH"
        info "fix: set GPU_ARCH=$CCNUM in config.sh"
      fi
    else
      note "could not query compute capability; trusting config GPU_ARCH=$GPU_ARCH"
    fi

    if [[ -n "$GFREE" && "$GFREE" -lt "$GPU_VRAM_MIN_MB" ]]; then
      bad "only ${GFREE} MiB VRAM free, need >= ${GPU_VRAM_MIN_MB} MiB"
      info "close other GPU users (nvidia-smi shows them) before running the solver"
    elif [[ -n "$GFREE" ]]; then
      ok "free VRAM ${GFREE} MiB >= required ${GPU_VRAM_MIN_MB} MiB"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 2. CUDA toolkit
# ---------------------------------------------------------------------------
log ""; log "== 2. CUDA toolkit =="
if ! command -v nvcc >/dev/null 2>&1; then
  bad "nvcc not on PATH -- install the CUDA toolkit and set CUDA_HOME in env.sh"
  info "sm_${GPU_ARCH} requires CUDA >= ${CUDA_MAJOR_MIN}.${CUDA_MINOR_MIN} (config.sh / env.sh)"
else
  CUDA_VER=$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)
  CU_MAJ=${CUDA_VER%%.*}; CU_MIN=${CUDA_VER##*.}
  if [[ "$CU_MAJ" -gt "$CUDA_MAJOR_MIN" ]] || \
     { [[ "$CU_MAJ" -eq "$CUDA_MAJOR_MIN" ]] && [[ "$CU_MIN" -ge "$CUDA_MINOR_MIN" ]]; }; then
    ok "nvcc $CUDA_VER (>= ${CUDA_MAJOR_MIN}.${CUDA_MINOR_MIN}, supports sm_$GPU_ARCH)"
  else
    bad "nvcc $CUDA_VER is too old -- sm_$GPU_ARCH needs >= ${CUDA_MAJOR_MIN}.${CUDA_MINOR_MIN}"
  fi
  info "nvcc: $(command -v nvcc)"

  # Does this nvcc actually accept our target arch? Cheapest possible proof.
  TMPCU=$(mktemp --suffix=.cu)
  echo '__global__ void k(){}' > "$TMPCU"
  if nvcc -arch="sm_$GPU_ARCH" -cubin -o /dev/null "$TMPCU" >/dev/null 2>&1; then
    ok "nvcc accepts -arch=sm_$GPU_ARCH"
  else
    bad "nvcc REJECTS -arch=sm_$GPU_ARCH -- toolkit does not know this architecture"
    info "this is the check that matters most; upgrade CUDA before building"
  fi
  rm -f "$TMPCU"

  # host compiler compatibility -- nvcc hard-errors on a too-new gcc
  if command -v "${CC:-gcc}" >/dev/null 2>&1; then
    GCCV=$("${CC:-gcc}" -dumpfullversion -dumpversion 2>/dev/null | head -1)
    TMPCU=$(mktemp --suffix=.cu); echo 'int main(){return 0;}' > "$TMPCU"
    if nvcc -ccbin "${CC:-gcc}" -o /dev/null "$TMPCU" >/dev/null 2>&1; then
      ok "nvcc accepts host compiler ${CC:-gcc} $GCCV"
    else
      bad "nvcc rejects host compiler ${CC:-gcc} $GCCV (too new for CUDA $CUDA_VER)"
      info "fix: install an older gcc and set CC/CXX in env.sh, OR"
      info "     set NVCC_EXTRA=\"-allow-unsupported-compiler\" in env.sh"
    fi
    rm -f "$TMPCU"
  fi
fi

# ---------------------------------------------------------------------------
# 3. MPI + compilers
# ---------------------------------------------------------------------------
log ""; log "== 3. MPI and compilers =="
for c in "${FC:-gfortran}" "${CC:-gcc}" "${MPIFC:-mpif90}" "${MPICC:-mpicc}" "${MPIRUN:-mpirun}"; do
  if command -v "$c" >/dev/null 2>&1; then ok "$c -> $(command -v "$c")"
  else bad "$c not found"; fi
done
if command -v "${MPIRUN:-mpirun}" >/dev/null 2>&1; then
  MPIV=$("${MPIRUN:-mpirun}" --version 2>&1 | head -1)
  info "$MPIV"
  # A single-rank launch is the exact thing the workflow does; prove it works.
  if "${MPIRUN:-mpirun}" ${MPIRUN_EXTRA:-} -np 1 true >/dev/null 2>&1; then
    ok "mpirun -np 1 works"
  else
    bad "mpirun -np 1 failed -- fix MPI (or MPIRUN_EXTRA in env.sh) before building"
    info "try:  mpirun ${MPIRUN_EXTRA:-} -np 1 hostname"
  fi
fi

# ---------------------------------------------------------------------------
# 4. RAM and disk
# ---------------------------------------------------------------------------
log ""; log "== 4. memory and disk =="
RAMGB=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
NSPEC=$((NEX_XI * NEX_ETA * NEX_Z))
NGLOB_EST=$(( (NEX_XI*4+1) * (NEX_ETA*4+1) * (NEX_Z*4+1) ))
info "mesh: NEX ${NEX_XI}x${NEX_ETA}x${NEX_Z} = $NSPEC spectral elements, ~$((NGLOB_EST/1000000)) M global points"

# Host RAM: xgenerate_databases at NPROC=1 holds the whole mesh in one process,
# and lbfgs_update.py holds ~m+1 float64 copies of the model vector.
RAM_NEED=24
if [[ -n "$RAMGB" ]]; then
  if [[ "$RAMGB" -ge "$RAM_NEED" ]]; then ok "host RAM ${RAMGB} GB (>= ${RAM_NEED} GB)"
  else bad "host RAM ${RAMGB} GB -- need ~${RAM_NEED} GB (genmesh at NPROC=1 + L-BFGS vectors)"; fi
fi

# Disk. The high-water mark is the adjoint stage: one event's reconstruction
# wavefield (absorb_field, ~29 GB at this mesh/NSTEP) plus its kernels, plus the
# accumulated per-event kernels and one iteration of forward results.
NEV=$(list_events "$SRC_SET" 2>/dev/null | wc -l) || NEV=0
KERN_GB=$(( (NSPEC * 125 * 4 * 4) / 1073741824 + 1 ))     # 4 kernel arrays/event
FWD_GB=$(( NEV * 250 / 1024 + 1 ))                        # ~250 MB of .semd/event
ABS_GB=$(( (2*(NEX_XI*NEX_Z) + 2*(NEX_ETA*NEX_Z) + NEX_XI*NEX_ETA) * 25 * 3 * 4 * NSTEP / 1073741824 + 1 ))
NEED_GB=$(( ABS_GB + FWD_GB + NEV * KERN_GB + 40 ))
info "estimated: ${ABS_GB} GB transient wavefield/event, ~${KERN_GB} GB kernels/event,"
info "           ~${FWD_GB} GB forward output for $NEV events  =>  ~${NEED_GB} GB peak"
mkdir -p "$WORKBASE" 2>/dev/null
HAVE_GB=$(free_disk_gb "$WORKBASE")
if [[ -n "$HAVE_GB" ]]; then
  if [[ "$HAVE_GB" -ge "$NEED_GB" ]]; then ok "free disk on $WORKBASE: ${HAVE_GB} GB (>= ~${NEED_GB} GB)"
  else bad "free disk on $WORKBASE: ${HAVE_GB} GB -- need ~${NEED_GB} GB"
       info "point WORKBASE at a larger disk in env.sh, or lower the event count"; fi
  # NVMe check -- the adjoint stage is I/O bound on the absorb_field stream
  DEV=$(df --output=source "$WORKBASE" 2>/dev/null | tail -1)
  info "WORKBASE device: $DEV"
fi

# ---------------------------------------------------------------------------
# 5. staged inputs
# ---------------------------------------------------------------------------
log ""; log "== 5. staged inputs =="
chk_file() { if [[ -s "$1" ]]; then ok "$2"; else bad "$2 -- missing: $1"; fi; }
chk_dir()  { if [[ -d "$1" ]]; then ok "$2"; else bad "$2 -- missing: $1"; fi; }

chk_file "$GPU_ROOT/DATA/meshfem3D_files/Mesh_Par_file" "Mesh_Par_file"
chk_file "$GPU_ROOT/DATA/meshfem3D_files/interfaces.dat" "interfaces.dat"
chk_file "$GPU_ROOT/DATA/STATIONS" "STATIONS"
chk_file "$GPU_ROOT/DATA/Par_file" "Par_file template"

TOMO="$GPU_ROOT/DATA/tomo_files/tomography_model.xyz"
if [[ -s "$TOMO" ]]; then
  ok "tomography_model.xyz ($(du -h "$TOMO" | cut -f1))"
  # The ite2 disaster in one check: z must be METRES, not kilometres.
  zmin=$(awk 'NR>4{if(NR==5)m=$3; else if($3<m)m=$3} NR>200000{exit} END{printf "%.0f", m}' "$TOMO")
  if [[ -n "$zmin" && "$zmin" -lt -1000 ]]; then
    ok "tomo z units look like METRES (z_min=$zmin)"
  else
    bad "tomo z_min=$zmin looks like KILOMETRES -- this is the ite2 bug; rescale before use"
  fi
else
  bad "tomography_model.xyz missing: $TOMO  (data not distributed with this repository)"
fi

if [[ "$NEV" -gt 0 ]]; then ok "event set $SRC_SET: $NEV events"
else bad "no events found in SRC_SET=$SRC_SET (data not distributed with this repository)"; fi

# Mesh_Par_file must agree with config.sh on partitioning, or the mesher will
# silently produce a decomposition the solver cannot read.
MPF="$GPU_ROOT/DATA/meshfem3D_files/Mesh_Par_file"
if [[ -f "$MPF" ]]; then
  mx=$(par_get "$MPF" NPROC_XI); me=$(par_get "$MPF" NPROC_ETA)
  mex=$(par_get "$MPF" NEX_XI);  mee=$(par_get "$MPF" NEX_ETA)
  if [[ "$mex" != "$NEX_XI" || "$mee" != "$NEX_ETA" ]]; then
    note "Mesh_Par_file NEX = ${mex}x${mee} but config.sh has ${NEX_XI}x${NEX_ETA}"
    info "config.sh NEX_* is used only for the size/disk estimates above -- update it to match"
  fi
  if [[ "$mx" == "$NPROC_XI" && "$me" == "$NPROC_ETA" ]]; then
    ok "Mesh_Par_file NPROC_XI/ETA = $mx/$me matches config"
  else
    # Not blocking: a freshly staged Mesh_Par_file still carries Anvil's 25x50
    # decomposition, and 2_make_mesh.sh rewrites it (with a .bak) before meshing.
    note "Mesh_Par_file has NPROC_XI=$mx NPROC_ETA=$me; config wants $NPROC_XI/$NPROC_ETA"
    info "expected on a fresh stage -- 2_make_mesh.sh rewrites it (keeps a .bak)"
    # But the divisibility constraint IS worth failing on, since it is unfixable
    # without changing the mesh design.
    if [[ $((mex % NPROC_XI)) -ne 0 || $((mee % NPROC_ETA)) -ne 0 ]]; then
      bad "NEX ${mex}x${mee} is not divisible by NPROC_XI/ETA = $NPROC_XI/$NPROC_ETA"
    fi
  fi
fi

# observed data for cal_adjoint. The bands live one level down, under
# <CALADJ_DATA>/final_sac_intergrate_neg_snr3_keepdist_wb/{3_10,6_18}/ -- check
# for the actual band dirs the pipeline reads, not just the root.
CC="$CALADJ_DATA/final_sac_intergrate_neg_snr3_keepdist_wb"
if [[ -d "$CC" ]]; then
  nb=0; missing=""
  for band in 3_10 6_18; do
    if [[ -d "$CC/$band" ]]; then nb=$((nb + 1)); else missing="$missing $band"; fi
  done
  if [[ "$nb" -eq 2 ]]; then
    ok "observed data: both bands present ($(du -sh "$CC" 2>/dev/null | cut -f1))"
  else
    bad "observed data incomplete -- missing band(s):$missing under $CC"
  fi
  [[ -s "$CALADJ_DATA/STATIONS" ]] && ok "observed STATIONS" \
    || bad "missing $CALADJ_DATA/STATIONS"
  nw=$(find "$CALADJ_DATA/windows" -name '*.win' 2>/dev/null | wc -l)
  if [[ "$nw" -gt 0 ]]; then
    info "window cache: $nw .win files staged"
  else
    info "no window cache staged -- the first cal_adjoint run will PICK windows (expected)"
  fi
else
  note "observed data not staged at $CC -- needed for cal_adjoint (~11.6 GB)"
  info "observed data are not distributed with this repository"
fi

# ---------------------------------------------------------------------------
# 6. python / uv
# ---------------------------------------------------------------------------
log ""; log "== 6. python (cal_adjoint) =="
if command -v "${UV_BIN:-uv}" >/dev/null 2>&1; then
  ok "uv $(${UV_BIN:-uv} --version 2>&1 | awk '{print $2}')"
else
  note "uv not found -- needed to build the cal_adjoint venv"
  info "install: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
if [[ -x "$GPU_ROOT/cal_adjoint/.venv/bin/python" ]]; then
  ok "cal_adjoint venv built"
  if "$GPU_ROOT/cal_adjoint/.venv/bin/python" -c "from mpi4py import MPI" 2>/dev/null; then
    ok "mpi4py imports"
  else
    bad "cal_adjoint venv exists but mpi4py fails to import -- rebuild: cd cal_adjoint && bash build_env_local.sh"
  fi
else
  note "cal_adjoint venv not built yet -- run: cd cal_adjoint && bash build_env_local.sh"
fi

# ---------------------------------------------------------------------------
log ""
hr
if [[ "$FAIL" -eq 0 ]]; then
  log "PREFLIGHT PASSED  ($PASS ok, $WARNC warning(s))"
  log "next:  ./1_build_specfem_gpu.sh"
  hr
  exit 0
else
  log "PREFLIGHT FAILED  ($FAIL blocking, $PASS ok, $WARNC warning(s))"
  log "fix the [FAIL] items above, then re-run ./0_preflight.sh"
  hr
  exit 1
fi
