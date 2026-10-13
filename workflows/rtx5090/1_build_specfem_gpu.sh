#!/bin/bash
# ite3_gpu/1_build_specfem_gpu.sh -- build SPECFEM3D 4.1.1 with CUDA for this GPU.
#
# THE ONE THING THIS SCRIPT EXISTS FOR
# -----------------------------------
# SPECFEM3D 4.1.1 ships GPU architectures up to sm_90 (Hopper) and no further:
#
#     Makefile.in:214   GENCODE_90 = -gencode=arch=compute_90,code="sm_90,compute_90"
#     Makefile.in:218   @COND_CUDA12_TRUE@GENCODE = $(GENCODE_90) -DGPU_DEVICE_Hopper
#
# An RTX 5090 is sm_120. A stock `--with-cuda=cuda12` build therefore emits
# Hopper cubins that the 5090 cannot execute natively -- it would fall back to
# JIT-compiling the embedded compute_90 PTX at every single solver start, which
# is both slow to launch and not what you want to benchmark.
#
# `GENCODE` is a plain recursive assignment in the generated Makefile, so a make
# command-line override wins and propagates to BOTH the compile (`NVCC_FLAGS`)
# and the device-link (`NVCCLINK`) rules -- which is exactly what we need, since
# a mismatch between those two produces link errors that look like missing
# symbols. So: configure with cuda12, then override GENCODE on the make line.
#
# Dropping -DGPU_DEVICE_Hopper along with it is harmless: that macro's entire
# body in src/gpu/mesh_constants_cuda.h is `#undef USE_LAUNCH_BOUNDS`, and
# USE_LAUNCH_BOUNDS is commented out by default anyway.
#
# TWO THINGS NOT TO DO
# --------------------
#  * Do NOT run autoreconf/autogen.sh. This tree's m4/ is EMPTY -- the
#    CIT_CUDA_CONFIG macro exists only pre-expanded inside the shipped
#    ./configure. Regenerating would leave it undefined and break the build.
#  * Do NOT pass --with-cuda=cuda13 or any unrecognised value. Configure still
#    sets COND_CUDA true but leaves every COND_CUDAn false, silently falling back
#    to GENCODE_20 and an undefined NVCCLINK_BASE. Use cuda12.
#
# Usage:
#   ./1_build_specfem_gpu.sh              # configure + build (incremental)
#   ./1_build_specfem_gpu.sh --clean      # make clean first (after a flag change)
#   ./1_build_specfem_gpu.sh --check      # print the resolved flags and stop
#   ./1_build_specfem_gpu.sh --cpu        # CPU-only build, for A/B comparison
#
# OTHER GPUS / ARM HOSTS: nothing here is x86-specific. Set GPU_ARCH in env.sh
# (it overrides config.sh) -- e.g. GPU_ARCH=90 for a GH200/H100, where the
# override simply reproduces the stock Hopper line. See ../grace_hopper/env.sh.gh200.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=build
source "$ROOT/lib/common.sh"

DO_CLEAN=0; CHECK_ONLY=0; CPU_ONLY=0; JOBS=$(nproc)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) DO_CLEAN=1 ;;
    --check) CHECK_ONLY=1 ;;
    --cpu)   CPU_ONLY=1 ;;
    -j)      JOBS="$2"; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

banner "SPECFEM3D CUDA build -- target sm_${GPU_ARCH}"

[[ -d "$SPECFEM_DIR" ]] || die "SPECFEM source tree not found: $SPECFEM_DIR
     Put a SPECFEM3D 4.1.1 source tree there:
     git clone --branch v4.1.1 --depth 1 https://github.com/SPECFEM/specfem3d.git \"$SPECFEM_DIR\""
[[ -x "$SPECFEM_DIR/configure" ]] || die "$SPECFEM_DIR/configure missing or not executable -- is that really a SPECFEM tree?"

# ---------------------------------------------------------------------------
# resolve flags
# ---------------------------------------------------------------------------
# NB: the code-target LIST must use bracket form `code=[sm_X,compute_X]`, NOT the
# quoted form `code="sm_X,compute_X"` the stock Makefile ships. This value is
# passed as `make GENCODE=...`, make stores it, then the recipe line is re-parsed
# by /bin/sh -- which strips the double quotes, leaving nvcc with a bare
# `code=sm_X,compute_X`. nvcc then reads the trailing `compute_X` as a third,
# key-less gencode token and dies with "'compute_X' is not in 'keyword=value'
# format". Brackets carry no meaning to sh here (no single-char file matches the
# glob, so it stays literal) and nvcc parses the list natively. Verified against
# nvcc 12.8.
GENCODE_OVERRIDE="-gencode=arch=compute_${GPU_ARCH},code=[sm_${GPU_ARCH},compute_${GPU_ARCH}]"
# Note we keep BOTH sm_ (native cubin) and compute_ (PTX). The PTX costs a little
# binary size and buys forward compatibility with a future card, at no runtime
# cost -- the native cubin is always preferred when present.

: "${CUDA_HOME:=/usr/local/cuda}"
CUDA_INC_DIR="${CUDA_INC:-$CUDA_HOME/include}"
CUDA_LIB_DIR="${CUDA_LIB:-$CUDA_HOME/lib64}"

log "SPECFEM_DIR : $SPECFEM_DIR"
log "GPU_ARCH    : sm_${GPU_ARCH}"
log "GENCODE     : $GENCODE_OVERRIDE"
log "CUDA_HOME   : $CUDA_HOME"
log "  include   : $CUDA_INC_DIR   $( [[ -f "$CUDA_INC_DIR/cuda_runtime.h" ]] && echo '(cuda_runtime.h ok)' || echo '(!! cuda_runtime.h NOT FOUND)')"
log "  lib       : $CUDA_LIB_DIR   $( ls "$CUDA_LIB_DIR"/libcudart.so* >/dev/null 2>&1 && echo '(libcudart ok)' || echo '(!! libcudart NOT FOUND)')"
log "FC/CC       : ${FC:-gfortran} / ${CC:-gcc}"
log "MPIFC/MPICC : ${MPIFC:-mpif90} / ${MPICC:-mpicc}"
log "FLAGS_CHECK : ${FLAGS_CHECK:--O3 -march=native}"
log "jobs        : $JOBS"

[[ "$CHECK_ONLY" -eq 1 ]] && { log "--check: stopping before configure"; exit 0; }

if [[ "$CPU_ONLY" -eq 0 ]]; then
  need_cmd nvcc "install the CUDA toolkit, or use --cpu for a CPU-only build"
  [[ -f "$CUDA_INC_DIR/cuda_runtime.h" ]] || die "cuda_runtime.h not found under $CUDA_INC_DIR -- set CUDA_HOME (or CUDA_INC) in env.sh"
  ls "$CUDA_LIB_DIR"/libcudart.so* >/dev/null 2>&1 || die "libcudart not found under $CUDA_LIB_DIR -- set CUDA_HOME (or CUDA_LIB) in env.sh"
fi

# ---------------------------------------------------------------------------
# stage the Par_file so configure/make see a consistent DATA/
# (SPECFEM's build reads DATA/Par_file for a few setup/ headers)
# ---------------------------------------------------------------------------
mkdir -p "$SPECFEM_DIR/DATA"
if [[ -f "$ROOT/DATA/Par_file" ]]; then
  cp "$ROOT/DATA/Par_file" "$SPECFEM_DIR/DATA/Par_file"
  log "staged DATA/Par_file into the build tree"
fi

# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------
cd "$SPECFEM_DIR" || die "cannot cd $SPECFEM_DIR"

CONF_ARGS=(
  "FC=${FC:-gfortran}"
  "CC=${CC:-gcc}"
  "MPIFC=${MPIFC:-mpif90}"
  "MPICC=${MPICC:-mpicc}"
  "FLAGS_CHECK=${FLAGS_CHECK:--O3 -march=native -funroll-loops}"
  "CFLAGS=${CFLAGS_LOCAL:--O3 -march=native -funroll-loops}"
)
if [[ "$CPU_ONLY" -eq 0 ]]; then
  CONF_ARGS+=( --with-cuda=cuda12 "CUDA_INC=$CUDA_INC_DIR" "CUDA_LIB=$CUDA_LIB_DIR" )
  [[ -n "${NVCC_EXTRA:-}" ]] && CONF_ARGS+=( "CUDA_FLAGS=$NVCC_EXTRA" )
fi

log ""
log "configure: ./configure ${CONF_ARGS[*]}"
if ! ./configure "${CONF_ARGS[@]}" >"$LOGDIR/configure.log" 2>&1; then
  tail -40 "$LOGDIR/configure.log" | tee -a "$LOGFILE"
  die "configure FAILED -- full log: $LOGDIR/configure.log"
fi
log "configure OK"

# Prove configure actually enabled CUDA before spending minutes compiling.
if [[ "$CPU_ONLY" -eq 0 ]]; then
  if grep -qE '^[[:space:]]*NVCC[[:space:]]*=[[:space:]]*nvcc' Makefile; then
    log "verified: Makefile has NVCC = nvcc (CUDA path enabled)"
  else
    tail -30 "$LOGDIR/configure.log" | tee -a "$LOGFILE"
    die "configure did NOT enable CUDA (Makefile still has NVCC = \$(CC)).
     Check $LOGDIR/configure.log for 'CUDA runtime header not found' or
     'CUDA library linking with nvcc failed'."
  fi
fi

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
MAKE_ARGS=()
[[ "$CPU_ONLY" -eq 0 ]] && MAKE_ARGS+=( "GENCODE=$GENCODE_OVERRIDE" )

if [[ "$CPU_ONLY" -eq 0 ]]; then
  # Show the flags nvcc will actually receive. `print-%` is a real rule in
  # src/gpu/rules.mk, so this is the authoritative answer, not a guess.
  log ""
  log "resolved nvcc flags:"
  make "${MAKE_ARGS[@]}" print-GENCODE    2>/dev/null | tee -a "$LOGFILE" || true
  make "${MAKE_ARGS[@]}" print-NVCC_FLAGS 2>/dev/null | tee -a "$LOGFILE" || true
  # Guard against a silently-ignored override. On Hopper (GPU_ARCH=90, e.g. a
  # GH200) the override and the stock line are the same architecture, so the
  # "no compute_90" test would be a false alarm; check for our arch instead.
  if ! make "${MAKE_ARGS[@]}" print-NVCC_FLAGS 2>/dev/null | grep -q "compute_${GPU_ARCH}"; then
    die "GENCODE override did not take: nvcc flags do not mention compute_${GPU_ARCH}.
     Inspect '$SPECFEM_DIR/Makefile' around the 'GENCODE =' line."
  fi
  if [[ "$GPU_ARCH" != "90" ]] && make "${MAKE_ARGS[@]}" print-NVCC_FLAGS 2>/dev/null | grep -q "compute_90"; then
    die "GENCODE override did not take: nvcc flags still mention compute_90.
     Inspect '$SPECFEM_DIR/Makefile' around the 'GENCODE =' line."
  fi
fi

if [[ "$DO_CLEAN" -eq 1 ]]; then
  log "make clean"
  make clean >"$LOGDIR/make_clean.log" 2>&1 || warn "make clean returned nonzero (usually harmless on a fresh tree)"
fi

log ""
log "building (this takes a few minutes)..."
t0=$(date +%s)
if ! make "${MAKE_ARGS[@]}" -j"$JOBS" >"$LOGDIR/make.log" 2>&1; then
  tail -60 "$LOGDIR/make.log" | tee -a "$LOGFILE"
  die "make FAILED -- full log: $LOGDIR/make.log"
fi
# The FWI post-processing tools live in separate make targets.
for tgt in xsmooth_sem xclip_sem xsum_preconditioned_kernels xsum_kernels xadd_model_iso xcombine_vol_data; do
  make "${MAKE_ARGS[@]}" "$tgt" >>"$LOGDIR/make.log" 2>&1 \
    || warn "target $tgt failed to build (see $LOGDIR/make.log) -- continuing"
done
t1=$(date +%s)
log "build finished in $(human_time $((t1 - t0)))"

# ---------------------------------------------------------------------------
# stage binaries + verify
# ---------------------------------------------------------------------------
mkdir -p "$ROOT/bin"
cp "$SPECFEM_DIR"/bin/* "$ROOT/bin/" 2>/dev/null || die "no binaries in $SPECFEM_DIR/bin"
log "staged $(ls "$ROOT/bin" | wc -l) binaries -> $ROOT/bin"

for b in xmeshfem3D xgenerate_databases xspecfem3D xsmooth_sem; do
  [[ -x "$ROOT/bin/$b" ]] || die "missing required binary after build: $b"
done
log "required binaries present: xmeshfem3D xgenerate_databases xspecfem3D xsmooth_sem"

if [[ "$CPU_ONLY" -eq 0 ]]; then
  # The decisive check: does the solver binary actually contain sm_120 code?
  # cuobjdump lists the embedded ELF/PTX images. If this says sm_120, the build
  # is correct and the 5090 will run native cubins with no JIT.
  if command -v cuobjdump >/dev/null 2>&1; then
    ARCHES=$(cuobjdump "$ROOT/bin/xspecfem3D" 2>/dev/null \
             | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')
    if [[ -n "$ARCHES" ]]; then
      log "embedded GPU architectures in xspecfem3D: $ARCHES"
      if grep -q "sm_${GPU_ARCH}" <<<"$ARCHES"; then
        log "VERIFIED: xspecfem3D contains native sm_${GPU_ARCH} code."
      else
        warn "xspecfem3D does NOT contain sm_${GPU_ARCH} -- it will run via PTX JIT (slow start) or fail."
      fi
    else
      warn "cuobjdump found no GPU code in xspecfem3D -- did the CUDA objects link in?"
    fi
  else
    warn "cuobjdump not on PATH; skipping the embedded-architecture check"
  fi
fi

hr
log "BUILD COMPLETE"
log "next:  ./2_make_mesh.sh"
hr
