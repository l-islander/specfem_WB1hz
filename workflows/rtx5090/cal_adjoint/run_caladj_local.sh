#!/bin/bash
# ite3_gpu/cal_adjoint/run_caladj_local.sh -- adjoint sources + misfit, locally.
#
# Replaces run_caladj_anvil.slurm / run_caladj_misfit_anvil.slurm /
# run_caladj_repick_anvil.slurm -- one script, three modes.
#
#   --full     (default) stages 1 -> 2(reuse) -> 3 -> 4, writes adjoint sources
#              AND the misfit. This is what the loop runs each iteration.
#   --misfit   stages 1 -> 2(reuse) -> 3 only. No adjoint sources. Used by the
#              line search to score a trial model.
#   --repick   stages 1 -> 2(NO reuse, pyflex re-picks) -> 3. For re-tuning the
#              window selection; mirrors the fresh picks into WINDIR.
#
# This stage is pure CPU (numpy/obspy/pyflex/pyadjoint over MPI), so it is the
# one place in the workflow that should use the whole machine while the GPU idles.
#
# TWO GUARDS THE ANVIL VERSION LACKS
# ----------------------------------
# 1. EMPTY WINDOW CACHE. `2_measure_windows --reuse` stages windows from WINDIR.
#    If WINDIR is empty it logs "No cached windows" per source, continues, and
#    produces zero measurements -- and because the misfit is computed as a shell
#    sum over .adjinfo files, the job then reports "NA 0" and EXITS ZERO. An
#    automated loop reads that as success. Here an empty cache is detected up
#    front and the run switches to picking mode instead.
# 2. ZERO MEASUREMENTS. Even with a populated cache, N==0 is treated as a hard
#    failure rather than a valid misfit.
#
# Usage:
#   ./run_caladj_local.sh                          # full, on $WORKBASE/forward/result
#   ./run_caladj_local.sh --misfit --datadir DIR --out FILE
#   ./run_caladj_local.sh --repick
#   ./run_caladj_local.sh --ranks 8                # cap CPU ranks

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=caladj
source "$ROOT/lib/common.sh"

MODE=full
DATADIR_OVERRIDE=""
MISFIT_OUT=""
RANKS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)    MODE=full ;;
    --misfit)  MODE=misfit ;;
    --repick)  MODE=repick ;;
    --datadir) DATADIR_OVERRIDE="$2"; shift ;;
    --out)     MISFIT_OUT="$2"; shift ;;
    --ranks)   RANKS="$2"; shift ;;
    -h|--help) sed -n '2,34p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done

cd "$HERE" || die "cannot cd $HERE"

PY="$HERE/.venv/bin/python"
[[ -x "$PY" ]] || die "cal_adjoint venv not built -- run: bash build_env_local.sh"

# ---------------------------------------------------------------------------
# export the paths config.py reads (config.sh stays the single source of truth)
# ---------------------------------------------------------------------------
export CALADJ_DATA="$CALADJ_DATA"
export CALADJ_ADJDIR="$ADJSRC"
export CALADJ_SEMD_BAND="$(band_code)"
[[ -n "$DATADIR_OVERRIDE" ]] && export CALADJ_DATADIR="$DATADIR_OVERRIDE" \
                             || export CALADJ_DATADIR="${CALADJ_DATADIR:-$WORKBASE/forward/result}"

# ranks: cal_adjoint parallelises over SOURCES, so more ranks than sources is waste
# -L follows symlinks: the line search's step-0 fallback builds $CALADJ_DATADIR
# as a farm of symlinks into the real result dir, and plain `-type d` does NOT
# match a symlink-to-directory -- it would report 0 sources and die.
NSRC=$(find -L "$CALADJ_DATADIR" -maxdepth 1 -type d -name '*_Z' 2>/dev/null | wc -l)
if [[ -z "$RANKS" ]]; then
  if [[ "${CALADJ_NPROC:-0}" -gt 0 ]]; then RANKS="$CALADJ_NPROC"
  else RANKS=$(nproc); fi
fi
[[ "$NSRC" -gt 0 && "$RANKS" -gt "$NSRC" ]] && RANKS="$NSRC"
[[ "$RANKS" -lt 1 ]] && RANKS=1

banner "cal_adjoint [$MODE]  ranks=$RANKS"
log "DATADIR : $CALADJ_DATADIR   ($NSRC source dirs)"
log "obs data: $CALADJ_DATA"
log "ADJDIR  : $ADJSRC"
log "band    : $CALADJ_SEMD_BAND (from DT=$DT)"

[[ "$NSRC" -gt 0 ]] || die "no <source>_Z dirs under $CALADJ_DATADIR -- has the forward stage finished?"
[[ -d "$CALADJ_DATA" ]] || die "observed data not found: $CALADJ_DATA (observed data are not distributed with this repository)"

# ---------------------------------------------------------------------------
# GUARD 1: window cache
# ---------------------------------------------------------------------------
WINDIR=$("$PY" -c 'import config; print(config.WINDIR)')
NWIN=$(find "$WINDIR" -name '*.win' 2>/dev/null | wc -l)
REUSE="--reuse"
if [[ "$MODE" == "repick" ]]; then
  REUSE=""
  log "window mode: RE-PICK (pyflex reselects; fresh picks mirrored into $WINDIR)"
elif [[ "$NWIN" -eq 0 ]]; then
  REUSE=""
  warn "window cache $WINDIR is EMPTY -- switching to PICK mode for this run."
  warn "  (--reuse on an empty cache yields 0 measurements and a misfit of 'NA 0',"
  warn "   which an automated loop would read as success. Picking instead.)"
else
  log "window cache: $NWIN .win files in $WINDIR (reusing)"
fi

mkdir -p log

run_stage() {
  local name="$1"; shift
  log "----- $name -----"
  local t0; t0=$(date +%s)
  if [[ "$RANKS" -gt 1 ]]; then
    cpu_run "$RANKS" "$PY" "$@" >>"$LOGFILE" 2>&1
  else
    "$PY" "$@" >>"$LOGFILE" 2>&1
  fi
  local rc=$?
  local t1; t1=$(date +%s)
  [[ $rc -eq 0 ]] || die "$name FAILED (rc=$rc) -- see $LOGFILE and $HERE/log/"
  log "$name done in $(human_time $((t1 - t0)))"
}

T0=$(date +%s)
run_stage "1_process_data"    1_process_data.py
if [[ -n "$REUSE" ]]; then
  run_stage "2_measure_windows" 2_measure_windows.py $REUSE
else
  run_stage "2_measure_windows" 2_measure_windows.py
fi
run_stage "3_calculateadj"    3_calculateadj.py ${CALADJ_EXCLUDE:+--exclude-same-group "$CALADJ_EXCLUDE"}

if [[ "$MODE" == "full" ]]; then
  run_stage "4_prep_adj"      4_prep_adj.py
fi

# ---------------------------------------------------------------------------
# misfit -- sum field 2 over every .adjinfo
# ---------------------------------------------------------------------------
: "${MISFIT_OUT:=$HERE/misfit_total.txt}"
# Sum the per-pair misfit (field 2) over every .adjinfo. Two traps make the
# obvious `xargs awk '{s+=$2}'` wrong here:
#   1. xargs may split the file list across SEVERAL awk processes, each running
#      END and printf-ing its partial sum with no separator -- the totals then
#      concatenate into one unparseable string (e.g. 684.31..633.62..181.99).
#   2. each .adjinfo has NO trailing newline, so `cat`-ing them merges the last
#      line of one file into the first line of the next -- corrupting $2.
# Robust fix: a SINGLE awk that reads the filename list on stdin and getline's
# each file's (single) line itself -- no batching, no ARG_MAX, no line-merging.
TOTAL=$(find "$CALADJ_DATADIR"/*_Z/OPT_* -name '*.adjinfo' 2>/dev/null | awk '
  { fn=$0; if ((getline rec < fn) > 0) { split(rec, f, /[ \t]+/); s += f[2]+0; c++ } close(fn) }
  END { if (c>0) printf "%.10g", s }')
N=$(find "$CALADJ_DATADIR"/*_Z/OPT_* -name '*.adjinfo' 2>/dev/null | wc -l)

# GUARD 2: zero measurements is a failure, not a misfit of zero.
if [[ -z "$TOTAL" || "$N" -eq 0 ]]; then
  die "cal_adjoint produced ZERO measurements (no .adjinfo files).
     This is the silent-failure mode: the stages all exited 0 but measured nothing.
     Likely causes, most common first:
       * synthetics not found -- check the band code. SPECFEM wrote *.$(band_code)Z.semd;
         confirm $CALADJ_DATADIR/<src>_Z/SYN/ actually contains them.
       * observed data missing for these sources under $CALADJ_DATA
       * window cache empty or non-overlapping with this source set
     See $HERE/log/ for the per-rank detail."
fi

printf '%s %s %(%Y-%m-%d %H:%M:%S)T\n' "$TOTAL" "$N" -1 > "$MISFIT_OUT"
T1=$(date +%s)

hr
log "MISFIT = $TOTAL  over $N measurements"
log "written to $MISFIT_OUT"
if [[ "$MODE" == "full" ]]; then
  NADJ=$(find "$ADJSRC" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  log "adjoint sources: $NADJ event dir(s) -> $ADJSRC"
  [[ "$NADJ" -gt 0 ]] || die "stage 4 wrote no adjoint source dirs to $ADJSRC"
fi
log "total wall: $(human_time $((T1 - T0)))"
hr
