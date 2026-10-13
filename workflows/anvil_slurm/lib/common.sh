# anvil_cpu/lib/common.sh -- shared helpers for the SLURM/CPU port.
#
# Same contract as ite3_gpu/lib/common.sh (logging, sentinels, Par_file
# helpers, event-set helpers) with three substitutions:
#   gpu_run  -> mpi_run    the one place an MPI launch is spelled out (srun/mpirun)
#   take_lock              a DRIVER lock, not a GPU lock: two run_loops must not
#                          submit against the same WORKBASE; the solver jobs
#                          themselves are serialised by SLURM
#   assert_gpu_parfile -> assert_cpu_parfile
# SLURM submission helpers live in lib/slurm.sh.

[[ -n "${_ITE3GPU_COMMON_LOADED:-}" ]] && return 0
_ITE3GPU_COMMON_LOADED=1

: "${GPU_ROOT:?common.sh must be sourced AFTER config.sh}"
: "${LOGDIR:=$GPU_ROOT/log}"
mkdir -p "$LOGDIR"

# STATEDIR holds every sentinel, keyed by iteration (ite3_gpu gotcha: this is
# $GPU_ROOT/.state/ite$ITER, NOT work/.state).
STATEDIR="${STATEDIR:-$GPU_ROOT/.state/ite${ITER:-0}}"
mkdir -p "$STATEDIR"

STAGE="${STAGE:-common}"
LOGFILE="${LOGFILE:-$LOGDIR/${STAGE}.log}"

# ---------------------------------------------------------------------------
# 1. logging
# ---------------------------------------------------------------------------
log()  { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOGFILE"; }
warn() { printf '[%(%Y-%m-%d %H:%M:%S)T] WARN: %s\n' -1 "$*" | tee -a "$LOGFILE" >&2; }
die()  { printf '[%(%Y-%m-%d %H:%M:%S)T] FATAL: %s\n' -1 "$*" | tee -a "$LOGFILE" >&2; exit 1; }
hr()   { printf '%s\n' "------------------------------------------------------------" | tee -a "$LOGFILE"; }
banner() { hr; log "$*"; hr; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1${2:+ -- $2}"; }

# ---------------------------------------------------------------------------
# 2. sentinels
# ---------------------------------------------------------------------------
mark_done()  { mkdir -p "$STATEDIR"; : > "$STATEDIR/$1.done"; }
is_done()    { [[ -f "$STATEDIR/$1.done" ]]; }
clear_done() { rm -f "$STATEDIR/$1.done"; }

step() {
  local name="$1" desc="$2"; shift 2
  [[ "${1:-}" == "--" ]] && shift
  if is_done "$name"; then log "SKIP  $name  ($desc) -- already done"; return 0; fi
  log "START $name  ($desc)"
  local t0 t1; t0=$(date +%s)
  if "$@" >>"$LOGFILE" 2>&1; then
    t1=$(date +%s); log "OK    $name  ($(human_time $((t1 - t0))))"; mark_done "$name"; return 0
  else
    local rc=$?; t1=$(date +%s)
    warn "FAIL  $name  (rc=$rc after $(human_time $((t1 - t0))))  -- see $LOGFILE"; return $rc
  fi
}

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
: "${MPIRUN:=srun}"
: "${MPIRUN_EXTRA:=}"

# mpi_run <nproc> <binary> [args...]  -- MUST be called inside a SLURM job.
# srun takes -n; mpirun takes -np. OMP is pinned to 1: the CPU solver is pure
# MPI and idle OpenMP threads only contend for the 128 cores.
mpi_run() {
  local np="$1"; shift
  local bin="$1"; shift
  [[ -x "$bin" ]] || die "not executable: $bin  (run ./1_build_specfem_cpu.sh)"
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "mpi_run must run inside a SLURM job (no SLURM_JOB_ID); use lib/slurm.sh sbatch_wait"
  local npflag="-np"; [[ "$(basename "$MPIRUN")" == "srun" ]] && npflag="-n"
  log "RUN   ${MPIRUN} ${MPIRUN_EXTRA} ${npflag} ${np} $(basename "$bin") $*"
  env OMP_NUM_THREADS=1 "$MPIRUN" $MPIRUN_EXTRA "$npflag" "$np" "$bin" "$@"
}
gpu_run() { mpi_run "$@"; }               # name kept so verbatim-ported call sites work

# cpu_run <nproc> <cmd> [args...] -- the mpi4py stages
cpu_run() {
  local np="$1"; shift
  local npflag="-np"; [[ "$(basename "$MPIRUN")" == "srun" ]] && npflag="-n"
  log "RUN   ${MPIRUN} ${MPIRUN_EXTRA} ${npflag} ${np} $*"
  "$MPIRUN" $MPIRUN_EXTRA "$npflag" "$np" "$@"
}

# ---------------------------------------------------------------------------
# 4. driver lock
# ---------------------------------------------------------------------------
LOCKFILE="${LOCKFILE:-$GPU_ROOT/.driver.lock}"
take_lock() {
  [[ -n "${_ITE3GPU_LOCK_HELD:-}" ]] && return 0
  exec 9>"$LOCKFILE" || die "cannot open lock file $LOCKFILE"
  flock -n 9 || die "another anvil_cpu driver holds $LOCKFILE (ps -ef | grep run_loop). Remove it only if nothing is running."
  export _ITE3GPU_LOCK_HELD=$$
}

# ---------------------------------------------------------------------------
# Par_file helpers (verbatim)
# ---------------------------------------------------------------------------
par_get() {
  grep -E "^[[:space:]]*$2[[:space:]]*=" "$1" | head -1 | cut -d= -f2- \
    | sed -E 's/[[:space:]]*(#.*)?$//' | tr -d '[:space:]'
}
par_set() {
  local f="$1" k="$2" v="$3"
  grep -qE "^[[:space:]]*$k[[:space:]]*=" "$f" || die "Par_file key not found: $k  (in $f)"
  sed -i -E "s|^([[:space:]]*$k[[:space:]]*)=.*|$(printf '%-32s' "$k")= $v|" "$f"
}
seismo_format_par() {
  if [[ "$2" == "on" && "${SEISMO_FORMAT:-ascii}" == "bin" ]]; then
    par_set "$1" USE_BINARY_FOR_SEISMOGRAMS .true.
  else
    par_set "$1" USE_BINARY_FOR_SEISMOGRAMS .false.
  fi
}
sim_forward()      { par_set "$1" SIMULATION_TYPE 1; par_set "$1" SAVE_FORWARD .false.; seismo_format_par "$1" on; }
sim_forward_save() { par_set "$1" SIMULATION_TYPE 1; par_set "$1" SAVE_FORWARD .true.;  seismo_format_par "$1" on; }
sim_kernel()       { par_set "$1" SIMULATION_TYPE 3; par_set "$1" SAVE_FORWARD .false.; seismo_format_par "$1" off; }

# assert_cpu_parfile <parfile> -- the CPU counterpart of assert_gpu_parfile.
assert_cpu_parfile() {
  local f="$1" v
  [[ "$(par_get "$f" GPU_MODE)" == ".false." ]] || die "GPU_MODE must be .false. in $f (CPU build)"
  v=$(par_get "$f" NPROC)
  [[ "$v" == "${NPROC}" ]] || die "Par_file NPROC=$v but config.sh NPROC=$NPROC ($f)"
  [[ "$(par_get "$f" MODEL)" == "gll" ]] || die "MODEL must be gll for a solver run ($f)"
}
assert_gpu_parfile() { assert_cpu_parfile "$@"; }

# ---------------------------------------------------------------------------
# event-set helpers (verbatim)
# ---------------------------------------------------------------------------
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
event_complete() {
  local d="$1" n
  [[ -f "$d/.event_done" ]] || return 1
  n=$(find "$d" -maxdepth 1 -name '*.sem*' 2>/dev/null | wc -l)
  [[ "$n" -ge "${EVENT_MIN_SEMD:-700}" ]]
}
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
# python
# ---------------------------------------------------------------------------
resolve_python() {
  [[ -n "${PYBIN:-}" ]] && return 0
  local cands=("$GPU_ROOT/cal_adjoint/.venv/bin/python" "${PYTHON:-}" python3 python) c
  for c in "${cands[@]}"; do
    [[ -n "$c" ]] || continue
    command -v "$c" >/dev/null 2>&1 || [[ -x "$c" ]] || continue
    if "$c" -c 'import sys,numpy; sys.exit(0 if sys.version_info>=(3,8) else 1)' >/dev/null 2>&1; then
      PYBIN="$c"; export PYBIN
      log "python: $PYBIN ($("$c" --version 2>&1))"
      return 0
    fi
  done
  die "no usable Python (>=3.8 + numpy). Build the venv: cd $GPU_ROOT/cal_adjoint && bash build_env_anvil.sh"
}

# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
free_disk_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
require_disk() {
  local have; have=$(free_disk_gb "$1")
  [[ -n "$have" && "$have" -ge "$2" ]] || die "need at least $2 GB free on $1 for $3 (have ${have:-?} GB)"
}
count_procfiles() { find "$1" -maxdepth 1 -name "proc*$2" 2>/dev/null | wc -l; }
