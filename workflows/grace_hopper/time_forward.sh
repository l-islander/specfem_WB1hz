#!/bin/bash
# workflows/grace_hopper/time_forward.sh -- timing harness: forward N sources of a model on this
# GPU and report the numbers compared against the RTX 5090.
#
# Wraps forward/run_forward.sh (so the run is the real workflow, sentinels and
# all) and adds: a GPU utilisation/memory sampler (duty cycle -- the 5090 sat at
# ~73% because of ASCII seismogram writes), genmesh wall time, per-event wall
# time from the runner log, and the solver's own timers from output_solver.txt.
#
# Usage (from workflows/rtx5090):
#   ../grace_hopper/time_forward.sh --n 1                       # one event
#   ../grace_hopper/time_forward.sh --n 40 --events events/lsauto_iter9 --tag ls40
#   ../grace_hopper/time_forward.sh --parse-only work/forward/result_M6   # stats of an existing run
#
# Options:
#   --model DIR    model to forward (default $MODELBASE/M6)
#   --events DIR   event set (default $SRC_SET = events/s2_keep150)
#   --n N          first N events of the set (default 1; 0 = all)
#   --out DIR      result dir (default $WORKBASE/forward/result_M6_gh200[_tag])
#   --tag NAME     label for the report (default: hostname)
#   --parse-only DIR   skip the run, just summarise an existing result dir
#
# Output: $OUT/timing_<tag>.md (markdown, paste into doc/GH200_RESULTS.md) and
#         $OUT/timing_<tag>.txt (key=value).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../rtx5090" && pwd)"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=gh200_timing
source "$ROOT/lib/common.sh"

MODEL_DIR="$MODELBASE/M6"; EVENTS="$SRC_SET"; N=1; OUT=""; TAG="$(hostname -s)"; PARSE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_DIR="$2"; shift ;;
    --events) EVENTS="$2"; shift ;;
    --n) N="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --tag) TAG="$2"; shift ;;
    --parse-only) PARSE="$2"; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done

NSPEC=$(( NEX_XI * NEX_ETA * NEX_Z ))
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n "$((GPU_DEVICE+1))p")

# ---------------------------------------------------------------------------
# solver-log statistics over every complete event in a result dir
# ---------------------------------------------------------------------------
solver_stats() {   # <resultdir> -> prints "n loop_mean total_mean total_min total_max step_ms nspec_log"
  local d="$1"
  for e in "$d"/*_Z; do
    f="$e/output_solver.txt"; [[ -f "$f" ]] || continue
    awk '
      /Mean elapsed time per time step/ {ms=$NF}
      /^ Elapsed time in seconds/       {loop=$NF}
      /Total elapsed time in seconds/   {tot=$NF}
      /^ NSPEC_global_min =/            {ns=$NF}
      END{ if (tot != "") printf "%s %s %s %s\n", loop, tot, ms, ns }' "$f"
  done | awk '{n++; l+=$1; t+=$2; s+=$3; ns=$4; if(n==1||$2<mn)mn=$2; if($2>mx)mx=$2}
              END{ if(n>0) printf "%d %.2f %.2f %.2f %.2f %.3f %s\n", n, l/n, t/n, mn, mx, 1000*s/n, ns; else print "0 0 0 0 0 0 0" }'
}

report() {   # <resultdir> <genmesh_s> <wall_per_event_s> <duty_pct> <peak_mib> <wall_total_s>
  local d="$1" tg="$2" wpe="$3" duty="$4" peak="$5" wtot="$6"
  read -r n loop tot tmin tmax step ns <<<"$(solver_stats "$d")"
  [[ "$n" -gt 0 ]] || die "no completed events with output_solver.txt under $d"
  [[ "$ns" != "0" && "$ns" != "" ]] && NSPEC="$ns"
  local tp_loop tp_tot tp_wall
  tp_loop=$(awk -v n="$NSPEC" -v s="$NSTEP" -v t="$loop" 'BEGIN{printf "%.1f", n*s/t/1e6}')
  tp_tot=$(awk  -v n="$NSPEC" -v s="$NSTEP" -v t="$tot"  'BEGIN{printf "%.1f", n*s/t/1e6}')
  tp_wall=$([[ "$wpe" != "na" ]] && awk -v n="$NSPEC" -v s="$NSTEP" -v t="$wpe" 'BEGIN{printf "%.1f", n*s/t/1e6}' || echo na)
  # iteration proxy (same arithmetic as 4_bench.sh): 150 forwards + 150 x
  # (forward+save + 2.2 x forward) + 3 LS steps x 40 forwards, on the wall basis
  local base="$wpe"; [[ "$base" == "na" ]] && base="$tot"
  local nev nls nsub
  nev=$(list_events "$SRC_SET" 2>/dev/null | wc -l); [[ "$nev" -eq 0 ]] && nev=150
  nls=$(wc -w <<<"$LS_STEPS"); nsub=$(list_events "$LS_SUBSET" 2>/dev/null | wc -l); [[ "$nsub" -eq 0 ]] && nsub=40
  local iter_h iter_scaled ref_base
  iter_h=$(awk -v f="$base" -v nev="$nev" -v nls="$nls" -v nsub="$nsub" \
           'BEGIN{ fs=f*1.2; k=f*2.2; printf "%.1f", (f*nev + (fs+k)*nev + f*nls*nsub)/3600 }')
  # ratio-scaled from the 5090's MEASURED 18.2 GPU-h/iteration on the same basis
  # (runner wall 75 s/event if we have it, else solver total 58.08 s)
  ref_base=75; [[ "$wpe" == "na" ]] && ref_base=58.08
  iter_scaled=$(awk -v f="$base" -v r="$ref_base" 'BEGIN{printf "%.1f", 18.2*f/r}')
  local md="$d/timing_${TAG}.md" txt="$d/timing_${TAG}.txt"
  {
    echo "### Forward timing: ${TAG} (${GPUNAME:-unknown GPU}, $(uname -m))"
    echo
    echo "- result dir: \`$d\`  events: $n  NSTEP: $NSTEP  DT: $DT  NSPEC: $NSPEC"
    echo
    echo "| metric | value | 5090 reference |"
    echo "|---|---|---|"
    echo "| genmesh (xgenerate_databases, serial CPU) | $( [[ "$tg" == na ]] && echo n/a || human_time "$tg") | ~22-25 min |"
    echo "| solver time loop, mean | ${loop} s | 45.1 s |"
    echo "| solver total elapsed, mean (min-max) | ${tot} s (${tmin}-${tmax}) | 58.1 s (57.4-58.7) |"
    echo "| mean per time step | ${step} ms | 6.54 ms |"
    echo "| runner wall per event (incl. mpirun start + staging) | $( [[ "$wpe" == na ]] && echo n/a || echo "${wpe} s") | ~75 s |"
    echo "| throughput, time-loop basis | ${tp_loop} M el-ts/s | 82.7 |"
    echo "| throughput, solver-total basis | ${tp_tot} M el-ts/s | 64.3 |"
    echo "| throughput, runner-wall basis | $( [[ "$tp_wall" == na ]] && echo n/a || echo "${tp_wall} M el-ts/s") | ~47-50 |"
    echo "| GPU duty cycle (nvidia-smi util, 1 s samples) | $( [[ "$duty" == na ]] && echo n/a || echo "${duty} %") | ~73 % |"
    echo "| peak GPU memory | $( [[ "$peak" == na ]] && echo n/a || echo "${peak} MiB") | (see 4_bench) |"
    echo "| total harness wall | $( [[ "$wtot" == na ]] && echo n/a || human_time "$wtot") | |"
    echo "| projected full iteration, ratio-scaled from 5090's measured 18.2 h | **${iter_scaled} GPU-h** | 18.2 GPU-h (measured) |"
    echo "| projected full iteration, 4_bench arithmetic (underestimates: 12.6 on the 5090) | ${iter_h} GPU-h | 12.6 GPU-h (arith.) |"
    echo
    echo "5090 reference = 150-event mean of work/forward/result_M6 solver logs (ascii seismograms,"
    echo "Aug 2026 campaign); the reference 47 M el-ts/s and 18.2 GPU-h are on the runner-wall basis."
    echo "This run used SEISMO_FORMAT=${SEISMO_FORMAT:-ascii}. 5090 bin-mode same-day pair (2026-09-11,"
    echo "3A.20050_Z): solver total 73.0 s ascii -> 49.5 s bin; seismogram write 17.2 s -> 0.24 s."
    echo "Seismogram write time this run: $(grep -h 'Writing the seismograms in parallel took' "$d"/*_Z/output_solver.txt 2>/dev/null | awk '{s+=$(NF-1); n++} END{if(n) printf "%.2f s mean over %d events", s/n, n; else print "n/a"}')"
  } | tee "$md"
  {
    echo "tag=$TAG gpu=$GPUNAME arch=$(uname -m) n=$n nstep=$NSTEP nspec=$NSPEC"
    echo "genmesh_s=$tg loop_s=$loop total_s=$tot total_min=$tmin total_max=$tmax step_ms=$step"
    echo "wall_per_event_s=$wpe duty_pct=$duty peak_mib=$peak harness_wall_s=$wtot"
    echo "tput_loop=$tp_loop tput_total=$tp_tot tput_wall=$tp_wall iter_scaled_h=$iter_scaled iter_arith_h=$iter_h"
  } > "$txt"
  log "wrote $md and $txt"
}

if [[ -n "$PARSE" ]]; then
  report "$PARSE" na na na na na; exit 0
fi

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
: "${OUT:=$WORKBASE/forward/result_M6_gh200${TAG:+_$TAG}}"
[[ -d "$MODEL_DIR" ]] || die "model dir missing: $MODEL_DIR"
[[ -d "$EVENTS" ]]    || die "event set missing: $EVENTS"
mkdir -p "$OUT"
banner "GH200 timing: $N event(s) of $(basename "$EVENTS") on $(basename "$MODEL_DIR") -> $OUT"

SAMPLE="$OUT/gpu_samples_${TAG}.txt"; : > "$SAMPLE"
( while :; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPU_DEVICE" 2>/dev/null | head -1 >> "$SAMPLE"
    sleep 1
  done ) & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

RUNLOG="$OUT/run_forward_${TAG}.log"
T0=$(date +%s)
ARGS=(--model "$MODEL_DIR" --events "$EVENTS" --out "$OUT")
[[ "$N" -gt 0 ]] && ARGS+=(--limit "$N")
"$ROOT/forward/run_forward.sh" "${ARGS[@]}" 2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
T1=$(date +%s)
kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; trap - EXIT
[[ "$RC" -eq 0 ]] || warn "run_forward.sh exited $RC -- reporting whatever completed"

# genmesh = time between its log line and the first "[1/N]" event line;
# per-event wall = (FORWARD DONE - first event line) / ok count
ts() { sed -n "s#^\[\([0-9-]* [0-9:]*\)\].*$1.*#\1#p" "$RUNLOG" | head -1; }
t_gm=$(ts "genmesh: building"); t_ev1=$(ts "\[1/"); t_done=$(ts "FORWARD DONE")
nok=$(sed -n 's/.*FORWARD DONE  ok=\([0-9]*\).*/\1/p' "$RUNLOG" | head -1)
GM=na; WPE=na
if [[ -n "$t_gm" && -n "$t_ev1" ]]; then GM=$(( $(date -d "$t_ev1" +%s) - $(date -d "$t_gm" +%s) )); fi
if [[ -n "$t_ev1" && -n "$t_done" && "${nok:-0}" -gt 0 ]]; then
  WPE=$(awk -v a="$(date -d "$t_done" +%s)" -v b="$(date -d "$t_ev1" +%s)" -v n="$nok" 'BEGIN{printf "%.1f", (a-b)/n}')
fi
# duty cycle over the event phase only (samples after genmesh); fall back to all
DUTY=na; PEAK=na
if [[ -s "$SAMPLE" ]]; then
  skip=0; [[ "$GM" != na ]] && skip=$GM
  DUTY=$(awk -F, -v s="$skip" 'NR>s {n++; u+=$1} END{if(n>0) printf "%.0f", u/n; else print "na"}' "$SAMPLE")
  PEAK=$(awk -F, 'BEGIN{m=0} {if($2+0>m)m=$2+0} END{print m}' "$SAMPLE")
fi
report "$OUT" "$GM" "$WPE" "$DUTY" "$PEAK" $((T1 - T0))
