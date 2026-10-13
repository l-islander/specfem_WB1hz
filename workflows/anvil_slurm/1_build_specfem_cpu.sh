#!/bin/bash
# anvil_cpu/1_build_specfem_cpu.sh -- SPECFEM3D 4.1.1 CPU build for Anvil (znver3), or reuse.
#
# Two paths:
#   --reuse   copy the binaries of the project tree's existing CPU build
#             (SPECFEM_REUSE_BIN in env.sh, the ite3-era AOCC build). Quick, but
#             ONLY valid if that tree's MPI matches the modules env.sh loads --
#             `ldd bin/xspecfem3D | grep mpi` is checked against `which mpirun`.
#   (default) configure + make the tree at SPECFEM_DIR without CUDA, with the
#             compilers/flags from env.sh. ~10 min on a login node with -j16.
#
# Do NOT run autoreconf (m4/ is empty in this tree). The Par_file is staged into
# the build tree because configure reads DATA/Par_file for a few headers.
#
# Usage: ./1_build_specfem_cpu.sh [--reuse] [--clean] [-j N]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=build; source "$ROOT/lib/common.sh"
REUSE=0; DO_CLEAN=0; JOBS=16
while [[ $# -gt 0 ]]; do
  case "$1" in --reuse) REUSE=1 ;; --clean) DO_CLEAN=1 ;; -j) JOBS="$2"; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;; *) die "unknown option: $1" ;; esac; shift
done
REQ=(xmeshfem3D xgenerate_databases xspecfem3D xsmooth_sem)
mkdir -p "$ROOT/bin"

if [[ "$REUSE" -eq 1 ]]; then
  banner "REUSE existing CPU binaries from ${SPECFEM_REUSE_BIN:-?}"
  [[ -d "${SPECFEM_REUSE_BIN:-}" ]] || die "SPECFEM_REUSE_BIN=$SPECFEM_REUSE_BIN is not a directory (env.sh)"
  for b in "${REQ[@]}"; do [[ -x "$SPECFEM_REUSE_BIN/$b" ]] || die "missing $SPECFEM_REUSE_BIN/$b"; done
  want=$(dirname "$(dirname "$(command -v mpirun 2>/dev/null || command -v mpicc)")")
  have=$(ldd "$SPECFEM_REUSE_BIN/xspecfem3D" 2>/dev/null | grep -E 'libmpi' | awk '{print $3}' | head -1)
  log "loaded MPI prefix : $want"; log "binary links      : ${have:-static/unknown}"
  [[ -z "$have" || "$have" == "$want"* ]] || warn "the reused binary links a DIFFERENT MPI than env.sh loads -- prefer a fresh build"
  cp -f "$SPECFEM_REUSE_BIN"/x* "$ROOT/bin/"
  log "staged $(ls "$ROOT/bin" | wc -l) binaries -> $ROOT/bin"; exit 0
fi

banner "SPECFEM3D CPU build  FC=${FC:-gfortran} FLAGS='${FLAGS_CHECK:-}'"
[[ -x "$SPECFEM_DIR/configure" ]] || die "no $SPECFEM_DIR/configure -- stage_to_anvil.sh ships the trimmed source tree"
mkdir -p "$SPECFEM_DIR/DATA"; cp -f "$ROOT/DATA/Par_file" "$SPECFEM_DIR/DATA/Par_file"
cd "$SPECFEM_DIR" || die "cannot cd $SPECFEM_DIR"
CONF=( "FC=${FC:-gfortran}" "CC=${CC:-gcc}" "MPIFC=${MPIFC:-mpif90}" "MPICC=${MPICC:-mpicc}"
       "FLAGS_CHECK=${FLAGS_CHECK:--O3 -march=znver3 -funroll-loops}" "CFLAGS=${CFLAGS_LOCAL:--O3 -march=znver3}" --with-mpi )
log "configure ${CONF[*]}"
./configure "${CONF[@]}" >"$LOGDIR/configure.log" 2>&1 || { tail -30 "$LOGDIR/configure.log"; die "configure failed -- $LOGDIR/configure.log"; }
grep -qE '^[[:space:]]*NVCC[[:space:]]*=[[:space:]]*nvcc' Makefile && die "Makefile enables CUDA -- this must be a CPU build"
[[ "$DO_CLEAN" -eq 1 ]] && make clean >/dev/null 2>&1
t0=$(date +%s)
make -j"$JOBS" >"$LOGDIR/make.log" 2>&1 || { tail -40 "$LOGDIR/make.log"; die "make failed -- $LOGDIR/make.log"; }
for tgt in xsmooth_sem xclip_sem xsum_preconditioned_kernels xsum_kernels xadd_model_iso xcombine_vol_data; do
  make "$tgt" >>"$LOGDIR/make.log" 2>&1 || warn "target $tgt failed (non-fatal)"
done
log "build finished in $(human_time $(( $(date +%s) - t0 )))"
cp -f "$SPECFEM_DIR"/bin/x* "$ROOT/bin/"
for b in "${REQ[@]}"; do [[ -x "$ROOT/bin/$b" ]] || die "missing required binary after build: $b"; done
log "staged $(ls "$ROOT/bin" | wc -l) binaries -> $ROOT/bin  (znver3: $(grep -c znver3 "$SPECFEM_DIR/Makefile") flag hits in Makefile)"
hr; log "BUILD COMPLETE -- next: tools/plan_mesh.py <period> --apply, then ./2_make_mesh.sh"; hr
