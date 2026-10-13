#!/bin/bash
# anvil_cpu/4_bench.sh -- time ONE event at NPROC ranks and price the whole loop in SUs.
#
# One SLURM job (NODES_PER_EVENT nodes): genmesh(M0) if needed, forward
# (SAVE_FORWARD off), forward (SAVE_FORWARD on) -> per-forward wall, the
# measured core rate (M element-timesteps/s/core), the transient wavefield size
# on scratch, and the extrapolated SU cost of an iteration. Feed the core rate
# back into tools/plan_mesh.py --core-rate to re-price the other mesh.
#
# Usage: ./4_bench.sh [--quick] [--nstep N]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"; [[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=bench; source "$ROOT/lib/common.sh"; source "$ROOT/lib/solver.sh"; source "$ROOT/lib/slurm.sh"
QUICK=0; BENCH_NSTEP="$NSTEP"; INJOB=0
while [[ $# -gt 0 ]]; do
  case "$1" in --quick) QUICK=1 ;; --nstep) BENCH_NSTEP="$2"; shift ;; --in-job) INJOB=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;; *) die "unknown option: $1" ;; esac; shift
done
[[ -f "$ORIMODEL/.done" ]] || die "no model at $ORIMODEL -- ./3_make_M0.sh first"
mapfile -t EVLIST < <(list_events "$SRC_SET"); EV="${EVLIST[0]}"; NEV=${#EVLIST[@]}
WD="$WORKBASE/bench/ev"; OUT="$WORKBASE/bench"

if [[ "$INJOB" -eq 0 ]]; then
  banner "BENCHMARK  event=$(basename "$EV")  NPROC=$NPROC ($NODES_PER_EVENT nodes)  NSTEP=$BENCH_NSTEP"
  ensure_genmesh_model "$ORIMODEL"
  rm -rf "$OUT"; mkdir -p "$OUT"
  sbatch_wait "bench" "$NODES_PER_EVENT" "$NPROC" "${EVENT_WALL}" "${PART_COMPUTE:?}" -- \
    bash "$0" --in-job $( [[ "$QUICK" -eq 1 ]] && echo --quick ) --nstep "$BENCH_NSTEP" || die "bench job failed"
  [[ -s "$OUT/times.txt" ]] || die "bench job left no $OUT/times.txt"
  source "$OUT/times.txt"
else
  prepare_event_workdir "$WD" "$(db_dir "$ORIMODEL")"
  par_set "$WD/DATA/Par_file" NSTEP "$BENCH_NSTEP"
  t0=$(date +%s); run_forward_event "$WD" "$EV" "$OUT/out_fwd" 0 || die "bench forward failed"; T_FWD=$(( $(date +%s) - t0 ))
  T_FWDSAVE=0; SZ_WF=0
  if [[ "$QUICK" -eq 0 ]]; then
    drop_wavefield "$WD"
    t0=$(date +%s); run_forward_event "$WD" "$EV" "$OUT/out_fwdsave" 1 || die "bench SAVE_FORWARD run failed"; T_FWDSAVE=$(( $(date +%s) - t0 ))
    SZ_WF=$(wavefield_gb "$WD"); drop_wavefield "$WD"
  fi
  SOLVER_S=$(grep -E 'Total elapsed time in seconds' "$OUT/out_fwd/output_solver.txt" | awk '{print $NF}' | head -1)
  printf 'T_FWD=%s\nT_FWDSAVE=%s\nSZ_WF=%s\nSOLVER_S=%s\nNODE=%s\n' "$T_FWD" "$T_FWDSAVE" "$SZ_WF" "${SOLVER_S:-0}" "$(hostname)" > "$OUT/times.txt"
  exit 0
fi

SCALE=$(awk -v a="$NSTEP" -v b="$BENCH_NSTEP" 'BEGIN{printf "%.4f", a/b}')
F=$(awk -v t="$T_FWD" -v s="$SCALE" 'BEGIN{printf "%d", t*s}')
FS=$(awk -v t="$T_FWDSAVE" -v s="$SCALE" 'BEGIN{printf "%d", t*s}'); [[ "$FS" -eq 0 ]] && FS=$(( F * 12 / 10 ))
K=$(( F * 34 / 10 ))   # kernel leg = 3.4 x forward, measured 2026-09-18 (CPU, 16 ranks)
NLS=$(wc -w <<<"$LS_STEPS"); NSUB=$(list_events "$LS_SUBSET" 2>/dev/null | wc -l); [[ "$NSUB" -eq 0 ]] && NSUB=40
NSPEC=$(( NEX_XI * NEX_ETA * NEX_Z ))
RATE=$(awk -v n="$NSPEC" -v s="$BENCH_NSTEP" -v t="${SOLVER_S:-$T_FWD}" -v p="$NPROC" 'BEGIN{ if (t>0) printf "%.2f", n*s/t/p/1e6; else print "n/a" }')
ITER_S=$(( F * NEV + (FS + K) * NEV + F * NLS * NSUB ))
NODEH=$(awk -v s="$ITER_S" -v n="$NODES_PER_EVENT" 'BEGIN{printf "%.1f", s*n/3600}')
hr
log "MEASURED on $NODE (NPROC=$NPROC, $NODES_PER_EVENT nodes, NSTEP=$BENCH_NSTEP)"
log "  forward wall            $(human_time "$T_FWD")   (solver time loop ${SOLVER_S:-?} s)"
[[ "$QUICK" -eq 0 ]] && log "  forward + save wall     $(human_time "$T_FWDSAVE")   transient wavefield ${SZ_WF} GB on scratch"
log "  core rate               ${RATE} M element-timesteps/s/core   <-- tools/plan_mesh.py --core-rate $RATE"
log ""
log "EXTRAPOLATED per iteration (NEV=$NEV, $NLS steps x $NSUB-event line search, kernel leg = 3.4 x forward)"
log "  forward pass  $(human_time $(( F * NEV )))   adjoint pass $(human_time $(( (FS + K) * NEV )))   line search $(human_time $(( F * NLS * NSUB )))"
log "  serial one-job time     $(human_time "$ITER_S")  =  ${NODEH} node-hours  =  $(awk -v h="$NODEH" -v c="$CORES_PER_NODE" 'BEGIN{printf "%.0f", h*c}') SU"
log "  wall with $ARRAY_THROTTLE in flight: ~$(awk -v s="$ITER_S" -v t="$ARRAY_THROTTLE" 'BEGIN{printf "%.1f", s/t/3600}') h  (+ cal_adjoint/post/update, ~1-2 h)"
log "  scratch at peak: $ARRAY_THROTTLE x ${SZ_WF} GB wavefield + kernels"
[[ "$SZ_WF" != 0 ]] && sed -i "s/^WAVEFIELD_GB_EST=.*/WAVEFIELD_GB_EST=$(awk -v w="$SZ_WF" 'BEGIN{printf "%d", w+1}')/" "$ROOT/mesh_plan.sh" 2>/dev/null
hr
