#!/bin/bash
# ite3_gpu/4_bench.sh -- time ONE event and extrapolate the whole inversion.
#
# RUN THIS BEFORE run_loop.sh. Nobody knows what a 5090 does on a 202,500-element
# SPECFEM mesh until it is measured, and the answer decides whether this workflow
# is a pleasant overnight run or a two-week one. This script answers it in about
# the time of two forward simulations.
#
# It measures:
#   * genmesh wall time
#   * forward wall time (SAVE_FORWARD off) -- the per-iteration cost driver
#   * forward wall time (SAVE_FORWARD on)  -- the adjoint's first leg + its disk cost
#   * peak GPU memory, sampled during the runs
#   * the reconstruction wavefield size on disk
# and then extrapolates a full iteration and a full MAX_UPDATES loop.
#
# Usage:
#   ./4_bench.sh                 # full benchmark
#   ./4_bench.sh --quick         # forward only, skip the SAVE_FORWARD leg
#   ./4_bench.sh --nstep 500     # short run; scales the estimate back up
#   ./4_bench.sh --skip-genmesh  # reuse an existing external_mesh.bin instead of
#                                # the forced ~25-min rebuild. ONLY valid when a
#                                # database built from the CURRENT meshdata AND
#                                # the CURRENT $ORIMODEL exists (e.g. right after
#                                # ./3_make_M0.sh, whose genmesh_M0 workdir has
#                                # M0 baked in). The genmesh line in the report
#                                # then reads 0s and is NOT representative.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=bench
source "$ROOT/lib/common.sh"
source "$ROOT/lib/solver.sh"

QUICK=0
SKIP_GENMESH=0
BENCH_NSTEP="$NSTEP"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --skip-genmesh) SKIP_GENMESH=1 ;;
    --nstep) BENCH_NSTEP="$2"; shift ;;
    -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done

banner "BENCHMARK -- one event on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n "$((GPU_DEVICE+1))p")"
take_lock

[[ -f "$ORIMODEL/.done" ]] || die "no model at $ORIMODEL -- run ./3_make_M0.sh first"
mapfile -t EVLIST < <(list_events "$SRC_SET")
[[ "${#EVLIST[@]}" -gt 0 ]] || die "no events in $SRC_SET"
EV="${EVLIST[0]}"
NEV=${#EVLIST[@]}
log "event under test : $(basename "$EV")"
log "event set size   : $NEV"
log "NSTEP for bench  : $BENCH_NSTEP (config NSTEP=$NSTEP)"

WD="$WORKBASE/bench/solver"
rm -rf "$WORKBASE/bench"
prepare_workdir "$WD"
par_set "$WD/DATA/Par_file" NSTEP "$BENCH_NSTEP"

# ---------------------------------------------------------------------------
# GPU memory sampler -- runs alongside the solver
# ---------------------------------------------------------------------------
PEAKFILE="$WORKBASE/bench/peak_mem"
start_sampler() {
  : > "$PEAKFILE"
  ( while :; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_DEVICE" 2>/dev/null \
        | head -1 >> "$PEAKFILE"
      sleep 1
    done ) &
  SAMPLER_PID=$!
}
stop_sampler() {
  [[ -n "${SAMPLER_PID:-}" ]] && kill "$SAMPLER_PID" 2>/dev/null
  wait "$SAMPLER_PID" 2>/dev/null
  awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{print m}' "$PEAKFILE" 2>/dev/null || echo 0
}

# ---------------------------------------------------------------------------
# 1. genmesh
# ---------------------------------------------------------------------------
log ""; log "== genmesh =="
t0=$(date +%s)
if [[ "$SKIP_GENMESH" -eq 1 ]]; then
  # Reuse an existing baked database instead of the forced rebuild. The solver
  # reads proc*_external_mesh.bin alone, and the SAVE_FORWARD leg writes its
  # wavefield into THIS workdir's DATABASES_MPI -- so we symlink the mesh FILE
  # (never the directory: writes would land in the source workdir).
  # TRUST NOTE: there is no stamp tying an external_mesh.bin to the model that
  # was baked into it; this flag asserts the source matches $ORIMODEL. The
  # auto-discovery order matches where a correct one lives right after
  # 3_make_M0.sh (genmesh_M0) or after a loop stage ran on this model.
  SRC=""
  for c in "$WORKBASE/genmesh_M0/DATABASES_MPI" \
           "$WORKBASE/forward/solver/DATABASES_MPI" \
           "$WORKBASE/adjoint/solver/DATABASES_MPI"; do
    [[ -f "$c/proc000000_external_mesh.bin" ]] && { SRC="$c"; break; }
  done
  [[ -n "$SRC" ]] || die "--skip-genmesh: no proc*_external_mesh.bin found under $WORKBASE
     (run ./3_make_M0.sh first, or drop --skip-genmesh)"
  ln -sf "$SRC"/proc*_external_mesh.bin "$WD/DATABASES_MPI/"
  stage_model "$WD" "$ORIMODEL"
  n=$(count_procfiles "$WD/DATABASES_MPI" "_external_mesh.bin")
  [[ "$n" -eq "$NPROC" ]] || die "--skip-genmesh: expected $NPROC external_mesh.bin in $WD, found $n"
  # xspecfem3D also reads OUTPUT_FILES/values_from_mesher.h at startup -- a
  # genmesh side-product the normal flow leaves in the workdir (clean_output
  # spares *.h). Without it the solver dies right after its banner with
  # "Error opening file: ./OUTPUT_FILES//values_from_mesher.h".
  SRCWD="$(dirname "$SRC")"
  cp -f "$SRCWD/OUTPUT_FILES/values_from_mesher.h" "$WD/OUTPUT_FILES/" 2>/dev/null \
    || die "--skip-genmesh: $SRCWD/OUTPUT_FILES/values_from_mesher.h not found --
     the source workdir must hold a complete genmesh run (mesh bin + mesher headers)"
  log "genmesh: REUSED $SRC  (--skip-genmesh: asserts it was baked from $(basename "$ORIMODEL"))"
else
  ensure_genmesh "$WD" "$ORIMODEL" --force
fi
T_GENMESH=$(( $(date +%s) - t0 ))
log "genmesh: $(human_time $T_GENMESH)"

# ---------------------------------------------------------------------------
# 2. forward, SAVE_FORWARD off
# ---------------------------------------------------------------------------
log ""; log "== forward (SAVE_FORWARD off) =="
DEST="$WORKBASE/bench/out_fwd"
mkdir -p "$DEST"
start_sampler
t0=$(date +%s)
run_forward_event "$WD" "$EV" "$DEST" 0 || die "benchmark forward failed -- see $LOGFILE"
T_FWD=$(( $(date +%s) - t0 ))
PEAK_FWD=$(stop_sampler)
NSEM=$(find "$DEST" -maxdepth 1 -name '*.sem*' | wc -l)
SZ_FWD=$(du -sm "$DEST" 2>/dev/null | cut -f1)
log "forward     : $(human_time $T_FWD)"
log "peak VRAM   : ${PEAK_FWD} MiB"
log "seismograms : $NSEM files, ${SZ_FWD} MB"
log "band code   : $(band_code)  (from DT=$DT)"

# throughput, the number worth comparing against other machines
TPUT=$(awk -v n="$((NEX_XI * NEX_ETA * NEX_Z))" -v s="$BENCH_NSTEP" -v t="$T_FWD" \
      'BEGIN{ if(t>0) printf "%.2f", (n*s)/t/1e6; else print "n/a" }')
log "throughput  : ${TPUT} M element-timesteps/s"

# ---------------------------------------------------------------------------
# 3. forward with SAVE_FORWARD -- the adjoint's first leg + its disk footprint
# ---------------------------------------------------------------------------
T_FWDSAVE=0; SZ_WF=0; PEAK_SAVE=0
if [[ "$QUICK" -eq 0 ]]; then
  log ""; log "== forward (SAVE_FORWARD on) -- measures the adjoint's disk cost =="
  DEST2="$WORKBASE/bench/out_fwdsave"
  mkdir -p "$DEST2"
  drop_wavefield "$WD"
  start_sampler
  t0=$(date +%s)
  run_forward_event "$WD" "$EV" "$DEST2" 1 || die "benchmark SAVE_FORWARD run failed"
  T_FWDSAVE=$(( $(date +%s) - t0 ))
  PEAK_SAVE=$(stop_sampler)
  SZ_WF=$(du -sm "$WD/DATABASES_MPI" --exclude='*_external_mesh.bin' --exclude='*_Database' \
          --exclude='*_vp.bin' --exclude='*_vs.bin' --exclude='*_rho.bin' 2>/dev/null | cut -f1)
  log "forward+save: $(human_time $T_FWDSAVE)   (overhead vs plain: $(human_time $((T_FWDSAVE - T_FWD))))"
  log "peak VRAM   : ${PEAK_SAVE} MiB"
  log "wavefield   : ${SZ_WF} MB on disk  <-- this much is needed PER EVENT, transiently"
  drop_wavefield "$WD"
fi

# ---------------------------------------------------------------------------
# 4. extrapolate
# ---------------------------------------------------------------------------
# Scale back up if the benchmark used a shortened NSTEP.
SCALE=$(awk -v a="$NSTEP" -v b="$BENCH_NSTEP" 'BEGIN{printf "%.4f", a/b}')
F=$(awk -v t="$T_FWD" -v s="$SCALE" 'BEGIN{printf "%d", t*s}')
FS=$(awk -v t="$T_FWDSAVE" -v s="$SCALE" 'BEGIN{printf "%d", t*s}')
[[ "$FS" -eq 0 ]] && FS=$(( F * 12 / 10 ))    # assume +20% if not measured

# SIMULATION_TYPE=3 runs the adjoint field forward while reconstructing the
# forward field backward, and accumulates 4 kernels. Empirically ~2x a plain
# forward; we use 2.2x to stay honest about the kernel writes.
K=$(( F * 22 / 10 ))
ADJ_PER_EVENT=$(( FS + K ))

ITER_FWD=$(( F * NEV ))
ITER_ADJ=$(( ADJ_PER_EVENT * NEV ))
# line search: LS_STEPS trial models, each forwarded over the LS_SUBSET
NLS=$(wc -w <<<"$LS_STEPS")
NSUB=$(list_events "$LS_SUBSET" 2>/dev/null | wc -l); [[ "$NSUB" -eq 0 ]] && NSUB=40
ITER_LS=$(( F * NLS * NSUB ))
ITER_TOTAL=$(( ITER_FWD + ITER_ADJ + ITER_LS ))

hr
log "MEASURED"
log "  genmesh                       $(human_time $T_GENMESH)$( [[ "$SKIP_GENMESH" -eq 1 ]] && echo '   [REUSED via --skip-genmesh -- not representative]' )"
log "  forward (no save)             $(human_time $F)   [scaled to NSTEP=$NSTEP]"
[[ "$QUICK" -eq 0 ]] && \
log "  forward (save wavefield)      $(human_time $FS)"
log "  peak VRAM                     ${PEAK_FWD} MiB of $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_DEVICE" 2>/dev/null | head -1) MiB"
[[ "$SZ_WF" -gt 0 ]] && \
log "  transient wavefield/event     ${SZ_WF} MB"
log ""
log "EXTRAPOLATED (NEV=$NEV events, ${NLS} line-search steps x ${NSUB}-event subset)"
log "  full forward pass             $(human_time $ITER_FWD)"
log "  adjoint pass (fwd+kernel)     $(human_time $ITER_ADJ)   [kernel leg estimated at 2.2x forward]"
log "  line search                   $(human_time $ITER_LS)"
log "  ------------------------------"
log "  ONE ITERATION (GPU time)      $(human_time $ITER_TOTAL)"
log "  ${MAX_UPDATES} ITERATIONS               $(human_time $(( ITER_TOTAL * MAX_UPDATES )))"
log ""
log "  Excludes cal_adjoint (CPU, runs while the GPU idles) and post-processing."
log "  The kernel-leg factor is an estimate; the first real adjoint event will"
log "  print its true time and you can revise from there."
hr

# ---------------------------------------------------------------------------
# 5. verdict
# ---------------------------------------------------------------------------
TOTALVRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_DEVICE" 2>/dev/null | head -1)
PK=$PEAK_FWD; [[ "$PEAK_SAVE" -gt "$PK" ]] && PK=$PEAK_SAVE
if [[ -n "$TOTALVRAM" && "$PK" -gt 0 ]]; then
  HEAD=$(( TOTALVRAM - PK ))
  log "VRAM headroom: ${HEAD} MiB free at peak."
  if [[ "$HEAD" -lt 2000 ]]; then
    warn "Headroom is tight. The SIMULATION_TYPE=3 run needs MORE than the forward"
    warn "(it holds a second wavefield plus 4 kernel arrays) -- expect ~2x this peak."
  else
    log "The kernel run holds roughly 2x the forward's arrays; ${PK} MiB peak suggests"
    log "SIMULATION_TYPE=3 will fit comfortably."
  fi
fi
if [[ "$SZ_WF" -gt 0 ]]; then
  log "Disk: keep at least $(( SZ_WF / 1024 + 5 )) GB free on $WORKBASE for the adjoint stage."
fi
log ""
log "next:  ./run_loop.sh    (or run stages by hand -- see README.md)"
hr
