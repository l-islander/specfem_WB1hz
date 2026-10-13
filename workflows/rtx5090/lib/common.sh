# ite3_gpu/lib/common.sh -- shared helpers for the single-GPU FWI workflow.
#
# Sourced by every orchestrator, right after config.sh. Provides the four things
# that SLURM used to provide and that a bare workstation does not:
#
#   1. LOGGING      log/warn/die with timestamps, tee'd to a per-stage log file
#   2. SENTINELS    mark_done/is_done -- the resumability substrate. On Anvil a
#                   step was "done" if its SLURM job had completed; here every
#                   step writes an explicit sentinel, so re-running any script
#                   skips finished work and picks up exactly where it stopped.
#   3. LAUNCHING    gpu_run -- the one place `mpirun` is spelled out, with the
#                   GPU pinned and the run timed.
#   4. LOCKING      one GPU means one solver at a time; take_lock makes that a
#                   hard guarantee instead of a convention.
#
# Nothing here is Anvil- or SLURM-aware. shellcheck-clean under `bash -n`.

# ---------------------------------------------------------------------------
# guard against double-sourcing (run_loop sources config+common, then calls
# sub-scripts that do the same)
[[ -n "${_ITE3GPU_COMMON_LOADED:-}" ]] && return 0
_ITE3GPU_COMMON_LOADED=1

: "${GPU_ROOT:?common.sh must be sourced AFTER config.sh}"
: "${LOGDIR:=$GPU_ROOT/log}"
mkdir -p "$LOGDIR"

# STATEDIR holds every sentinel. Keyed by iteration so a new iteration starts
# clean without disturbing the previous one's record.
STATEDIR="${STATEDIR:-$GPU_ROOT/.state/ite${ITER:-0}}"
mkdir -p "$STATEDIR"

# LOGFILE is per-stage; each entry script sets STAGE before sourcing us.
STAGE="${STAGE:-common}"
LOGFILE="${LOGFILE:-$LOGDIR/${STAGE}.log}"

# ---------------------------------------------------------------------------
# 1. logging
# ---------------------------------------------------------------------------
log()  { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOGFILE"; }
warn() { printf '[%(%Y-%m-%d %H:%M:%S)T] WARN: %s\n' -1 "$*" | tee -a "$LOGFILE" >&2; }
die()  { printf '[%(%Y-%m-%d %H:%M:%S)T] FATAL: %s\n' -1 "$*" | tee -a "$LOGFILE" >&2; exit 1; }
hr()   { printf '%s\n' "------------------------------------------------------------" | tee -a "$LOGFILE"; }

# banner "text" -- a visually distinct stage header in the log
banner() {
  hr; log "$*"; hr
}

# need_cmd <cmd> [hint] -- hard requirement on an external tool
need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1${2:+ -- $2}"
}

# ---------------------------------------------------------------------------
# 2. sentinels (the resumability substrate)
# ---------------------------------------------------------------------------
# A sentinel is an empty file under $STATEDIR. Presence == that unit of work
# finished successfully. Everything that costs more than a few seconds gets one.
mark_done()  { mkdir -p "$STATEDIR"; : > "$STATEDIR/$1.done"; }
is_done()    { [[ -f "$STATEDIR/$1.done" ]]; }
clear_done() { rm -f "$STATEDIR/$1.done"; }

# step <name> <description> -- run the remaining args as a command, guarded by a
# sentinel and timed. Usage:
#     step mesh "build the mesh" -- mpirun -np 1 ./bin/xmeshfem3D
# Re-running the script skips any step whose sentinel exists.
step() {
  local name="$1" desc="$2"; shift 2
  [[ "${1:-}" == "--" ]] && shift
  if is_done "$name"; then
    log "SKIP  $name  ($desc) -- already done"
    return 0
  fi
  log "START $name  ($desc)"
  local t0 t1
  t0=$(date +%s)
  if "$@" >>"$LOGFILE" 2>&1; then
    t1=$(date +%s)
    log "OK    $name  ($(human_time $((t1 - t0))))"
    mark_done "$name"
    return 0
  else
    local rc=$?
    t1=$(date +%s)
    warn "FAIL  $name  (rc=$rc after $(human_time $((t1 - t0))))  -- see $LOGFILE"
    return $rc
  fi
}

# human_time <seconds> -> "1h 23m 45s"
human_time() {
  local s=${1:-0} h m
  h=$((s / 3600)); m=$(((s % 3600) / 60)); s=$((s % 60))
  if   [[ $h -gt 0 ]]; then printf '%dh %02dm %02ds' "$h" "$m" "$s"
  elif [[ $m -gt 0 ]]; then printf '%dm %02ds' "$m" "$s"
  else                      printf '%ds' "$s"; fi
}

# ---------------------------------------------------------------------------
# 3. launching
# ---------------------------------------------------------------------------
# MPIRUN is resolved once. env.sh may override it (e.g. to srun, or to an
# absolute path when several MPIs are installed).
: "${MPIRUN:=mpirun}"
: "${MPIRUN_EXTRA:=}"     # e.g. "--allow-run-as-root" inside a container

# gpu_run <nproc> <binary> [args...]
# The single place an MPI launch is spelled out. Pins the chosen GPU, keeps the
# solver off the CPU-thread oversubscription path, and echoes the exact command
# so the log is reproducible by hand.
gpu_run() {
  local np="$1"; shift
  local bin="$1"; shift
  [[ -x "$bin" ]] || die "not executable: $bin  (did the build finish? run ./1_build_specfem_gpu.sh)"

  # One rank per GPU. OpenMP is not used by the CUDA solver path; pinning it to 1
  # avoids the runtime spawning $(nproc) idle threads per rank.
  local -a env_pfx=(
    env
    "CUDA_VISIBLE_DEVICES=${GPU_DEVICE:-0}"
    "OMP_NUM_THREADS=1"
  )

  log "RUN   ${MPIRUN} -np ${np} $(basename "$bin") $*"
  if [[ "${NICE_LEVEL:-0}" -ne 0 ]]; then
    nice -n "${NICE_LEVEL}" "${env_pfx[@]}" "$MPIRUN" $MPIRUN_EXTRA -np "$np" "$bin" "$@"
  else
    "${env_pfx[@]}" "$MPIRUN" $MPIRUN_EXTRA -np "$np" "$bin" "$@"
  fi
}

# cpu_run <nproc> <cmd> [args...] -- for the CPU-side MPI python stages
# (cal_adjoint). No GPU pinning, no nice.
cpu_run() {
  local np="$1"; shift
  log "RUN   ${MPIRUN} -np ${np} $*"
  "$MPIRUN" $MPIRUN_EXTRA -np "$np" "$@"
}

# ---------------------------------------------------------------------------
# 4. locking -- one GPU, one solver
# ---------------------------------------------------------------------------
# Without a scheduler nothing stops two shells from launching xspecfem3D at the
# same time; the second would either OOM the card or silently corrupt the shared
# DATABASES_MPI. flock makes it a clean refusal instead.
LOCKFILE="${LOCKFILE:-$GPU_ROOT/.gpu.lock}"
take_lock() {
  # RE-ENTRANT ACROSS PROCESSES. The stages nest: linesearch.sh takes the lock,
  # then spawns run_forward.sh, which would take it again. flock(2) locks are per
  # open-file-description, and the child's `exec 9>` creates a NEW description --
  # so a naive implementation has the child deadlock against its own parent.
  # (Not theoretical: this made every line search fail.)
  #
  # The exported marker is inherited by children, so a nested take_lock is a
  # no-op while an ancestor holds the lock; an unrelated shell has no marker and
  # is correctly refused.
  if [[ -n "${_ITE3GPU_LOCK_HELD:-}" ]]; then
    return 0
  fi
  exec 9>"$LOCKFILE" || die "cannot open lock file $LOCKFILE"
  if ! flock -n 9; then
    die "another ite3_gpu run holds the GPU lock ($LOCKFILE).
     Check with:  ps -ef | grep -E 'xspecfem3D|run_loop'
     If you are sure nothing is running, remove $LOCKFILE and retry."
  fi
  export _ITE3GPU_LOCK_HELD=$$
  # release happens implicitly when the holding shell exits
}

# ---------------------------------------------------------------------------
# Par_file helpers
# ---------------------------------------------------------------------------
# SPECFEM reads DATA/Par_file at runtime, so every simulation-type switch is a
# text edit. ite3 uses change_simulation_type.pl for this; these helpers do the
# same job with sed, plus a read accessor, and are anchored to the start of the
# line so a value inside a comment can never be hit.

# par_get <parfile> <KEY> -> value (whitespace stripped)
par_get() {
  grep -E "^[[:space:]]*$2[[:space:]]*=" "$1" | head -1 | cut -d= -f2- \
    | sed -E 's/[[:space:]]*(#.*)?$//' | tr -d '[:space:]'
}

# par_set <parfile> <KEY> <VALUE>  -- in place; fails loudly if the key is absent
par_set() {
  local f="$1" k="$2" v="$3"
  grep -qE "^[[:space:]]*$k[[:space:]]*=" "$f" \
    || die "Par_file key not found: $k  (in $f)"
  # Preserve SPECFEM's column alignment (key padded to 32, then "= value"), so a
  # rewritten Par_file stays diffable against the stock one.
  sed -i -E "s|^([[:space:]]*$k[[:space:]]*)=.*|$(printf '%-32s' "$k")= $v|" "$f"
}

# sim_forward <parfile>        SIMULATION_TYPE=1, SAVE_FORWARD=.false.
#   the cheap forward: seismograms only, no 29 GB wavefield dump.
# seismo_format_par <parfile> <on|off> -- USE_BINARY_FOR_SEISMOGRAMS follows
# config.sh SEISMO_FORMAT for forward runs; it MUST be off for kernel runs
# (SPECFEM: "Adjoint simulations not supported with .bin format", found 2026-09-11).
seismo_format_par() {
  if [[ "$2" == "on" && "${SEISMO_FORMAT:-ascii}" == "bin" ]]; then
    par_set "$1" USE_BINARY_FOR_SEISMOGRAMS .true.
  else
    par_set "$1" USE_BINARY_FOR_SEISMOGRAMS .false.
  fi
}
sim_forward() {
  par_set "$1" SIMULATION_TYPE 1
  par_set "$1" SAVE_FORWARD .false.
  seismo_format_par "$1" on
}
# sim_forward_save <parfile>   SIMULATION_TYPE=1, SAVE_FORWARD=.true.
#   the adjoint's first leg: writes proc*_save_forward_arrays.bin + absorb_field.
sim_forward_save() {
  par_set "$1" SIMULATION_TYPE 1
  par_set "$1" SAVE_FORWARD .true.
  seismo_format_par "$1" on
}
# sim_kernel <parfile>         SIMULATION_TYPE=3, SAVE_FORWARD=.false.
#   the adjoint's second leg: reconstructs the forward field and writes kernels.
sim_kernel() {
  par_set "$1" SIMULATION_TYPE 3
  par_set "$1" SAVE_FORWARD .false.
  seismo_format_par "$1" off
}

# assert_gpu_parfile <parfile> -- refuse to launch with a Par_file that SPECFEM's
# GPU path will reject at runtime (each of these is a hard `stop` in the solver:
# initialize_simulation.F90 / read_parameter_file.F90). Catching them here turns
# a 3-minute mystery crash into a one-line message.
assert_gpu_parfile() {
  local f="$1" v
  [[ "$(par_get "$f" GPU_MODE)" == ".true." ]] || die "GPU_MODE must be .true. in $f"
  for k in SAVE_MOHO_MESH USE_LDDRK LTS_MODE PML_CONDITIONS; do
    v=$(par_get "$f" "$k")
    [[ "$v" == ".false." ]] || die "$k must be .false. for a GPU run (is '$v' in $f)"
  done
  v=$(par_get "$f" NPROC)
  [[ "$v" == "${NPROC}" ]] || die "Par_file NPROC=$v but config.sh NPROC=$NPROC ($f)"
}

# ---------------------------------------------------------------------------
# event-set helpers
# ---------------------------------------------------------------------------
# The Anvil source sets come in two shapes: flat (<src>_Z/ at top level) and
# pre-grouped (group_N/RUN/<src>_Z/). With one GPU there is nothing to group, so
# we always flatten. Prints one absolute event-dir path per line, sorted.
list_events() {
  local src="$1"
  [[ -d "$src" ]] || die "event set not found: $src"
  if compgen -G "$src/group_*/RUN/*/" >/dev/null; then
    find "$src"/group_*/RUN -mindepth 1 -maxdepth 1 -type d | sort
  elif compgen -G "$src/group_*/*/" >/dev/null; then
    find "$src"/group_* -mindepth 1 -maxdepth 1 -type d | sort
  else
    find "$src" -mindepth 1 -maxdepth 1 -type d | sort
  fi
}

# event_complete <dir> -- the explicit, filesystem-independent completion test.
# See the EVENT_MIN_SEMD note in config.sh for why we do NOT use `du`.
event_complete() {
  local d="$1" n
  [[ -f "$d/.event_done" ]] || return 1
  n=$(find "$d" -maxdepth 1 -name '*.sem*' 2>/dev/null | wc -l)
  [[ "$n" -ge "${EVENT_MIN_SEMD:-700}" ]]
}

# semd_glob -- SPECFEM derives the channel BAND CODE from DT
# (write_seismograms.f90:band_instrument_code):
#     DT >  0.1     -> MX        0.0125 < DT <= 0.1    -> BX
#     0.004 < DT <= 0.0125 -> HX        DT <= 0.004    -> CX/FX
# ite3 runs DT=0.02, so files are named <net>.<sta>.BXZ.semd. ite2 ran DT=0.005
# and produced HXZ -- which is why cal_adjoint's 1_process_data.py hardcodes an
# HXZ glob and finds NOTHING on ite3 output. This port patches that glob to
# accept both (see cal_adjoint/run_caladj_local.sh), and exports the expected
# code so the runners can assert it.
band_code() {
  awk -v dt="${DT:-0.02}" 'BEGIN{
    if      (dt >  1.0)    print "LX";
    else if (dt >  0.1)    print "MX";
    else if (dt >  0.0125) print "BX";
    else if (dt >  0.004)  print "HX";
    else if (dt >  0.001)  print "CX";
    else                   print "FX";
  }'
}

# ---------------------------------------------------------------------------
# python interpreter resolution
# ---------------------------------------------------------------------------
# The post-processing and model-update stages are numpy programs. A bare
# `python3` is NOT safe to assume: it may be too old (Anvil's is 3.6.8) or have
# no numpy. The cal_adjoint venv is built to a known spec (>=3.10, numpy, scipy)
# and is the natural interpreter for all of them, so prefer it and verify.
#
# Sets PYBIN. Dies with an actionable message rather than letting a stage fail
# halfway through with "ModuleNotFoundError: numpy".
resolve_python() {
  [[ -n "${PYBIN:-}" ]] && return 0
  local cands=("$GPU_ROOT/cal_adjoint/.venv/bin/python" "${PYTHON:-}" python3 python)
  local c
  for c in "${cands[@]}"; do
    [[ -n "$c" ]] || continue
    command -v "$c" >/dev/null 2>&1 || [[ -x "$c" ]] || continue
    if "$c" -c 'import sys,numpy; sys.exit(0 if sys.version_info>=(3,8) else 1)' >/dev/null 2>&1; then
      PYBIN="$c"; export PYBIN
      log "python: $PYBIN ($("$c" --version 2>&1), numpy $("$c" -c 'import numpy;print(numpy.__version__)' 2>/dev/null))"
      return 0
    fi
  done
  die "no usable Python found (need >= 3.8 with numpy).
     The cal_adjoint venv is the intended interpreter for these stages:
         cd $GPU_ROOT/cal_adjoint && bash build_env_local.sh
     Or set PYTHON=/path/to/python in env.sh."
}

# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
# free_disk_gb <path> -- integer GB available on the filesystem holding <path>
free_disk_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# require_disk <path> <gb> <what>
require_disk() {
  local have; have=$(free_disk_gb "$1")
  [[ -n "$have" && "$have" -ge "$2" ]] \
    || die "need at least $2 GB free on $1 for $3 (have ${have:-?} GB)"
}

# count_procfiles <dir> <suffix> -- how many proc*<suffix> live in <dir>
count_procfiles() { find "$1" -maxdepth 1 -name "proc*$2" 2>/dev/null | wc -l; }
