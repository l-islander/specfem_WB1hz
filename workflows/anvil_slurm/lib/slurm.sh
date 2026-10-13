# anvil_cpu/lib/slurm.sh -- the four things SLURM does for this workflow.
#
#   sbatch_wait   submit ONE job that runs a command in this tree, block until
#                 it ends, return its exit status. Used for genmesh, mesh, M0,
#                 cal_adjoint, post, smooth, update, bench.
#   sbatch_array  submit a JOB ARRAY over a list file (one event per task),
#                 block until every task ends. Used by forward + adjoint.
#   job_state     sacct-based exit-code query (for the summary lines).
#   in_slurm      are we inside an allocation already?
#
# Scripts are submitted with --wrap "bash /abs/path/script args" rather than
# `sbatch script`: sbatch copies a script into its spool dir, which breaks
# every $(dirname "${BASH_SOURCE[0]}") in this tree. --wrap keeps the path.
#
# Every job inherits the caller's environment (--export=ALL), so a driver that
# exported CALADJ_WINDIR / LS_SUBSET (run_loop's REWINDOW logic) propagates
# them into the jobs unchanged -- the same mechanism as the local port, one
# process boundary further away.

[[ -n "${_ITE3GPU_SLURM_LOADED:-}" ]] && return 0
_ITE3GPU_SLURM_LOADED=1
: "${_ITE3GPU_COMMON_LOADED:?slurm.sh must be sourced after lib/common.sh}"

in_slurm() { [[ -n "${SLURM_JOB_ID:-}" ]]; }

_sb_common() {
  # account/partition/extra are from env.sh; a missing account is fatal early
  [[ -n "${SLURM_ACCOUNT:-}" ]] || die "SLURM_ACCOUNT is empty -- set it in env.sh"
  printf '%s' "--account=$SLURM_ACCOUNT --export=ALL --parsable ${SBATCH_EXTRA:-}"
}

# sbatch_wait <name> <nodes> <ntasks> <wall> <partition> -- <command...>
#   Returns the job's exit status (0 = ok). Logs the job id and wall time.
sbatch_wait() {
  local name="$1" nodes="$2" ntasks="$3" wall="$4" part="$5"; shift 5
  [[ "${1:-}" == "--" ]] && shift
  local cmd; cmd=$(printf '%q ' "$@")
  mkdir -p "$LOGDIR/slurm"
  local jid t0 t1 rc
  t0=$(date +%s)
  log "SBATCH $name: nodes=$nodes ntasks=$ntasks wall=$wall partition=$part"
  log "       $cmd"
  jid=$(sbatch $(_sb_common) --job-name="$name" --nodes="$nodes" --ntasks="$ntasks" \
          --time="$wall" --partition="$part" --output="$LOGDIR/slurm/${name}_%j.out" \
          --wait --wrap "cd $(printf '%q' "$GPU_ROOT") && $cmd" 2>>"$LOGFILE")
  rc=$?
  t1=$(date +%s)
  jid="${jid%%;*}"
  if [[ $rc -eq 0 ]]; then
    log "JOB    $name ($jid) finished OK in $(human_time $((t1 - t0)))"
  else
    warn "JOB    $name (${jid:-?}) FAILED rc=$rc after $(human_time $((t1 - t0)))  -- $LOGDIR/slurm/${name}_${jid:-JOBID}.out"
  fi
  return $rc
}

# sbatch_array <name> <listfile> <nodes_per_task> <ntasks_per_task> <wall> <partition> <throttle> -- <worker script> [args]
#   The worker receives the list file and reads line $SLURM_ARRAY_TASK_ID.
#   Blocks until the whole array ends. Return status is sbatch --wait's: nonzero
#   if ANY task failed -- callers verify sentinels afterwards regardless.
sbatch_array() {
  local name="$1" list="$2" nodes="$3" ntasks="$4" wall="$5" part="$6" throttle="$7"; shift 7
  [[ "${1:-}" == "--" ]] && shift
  local n; n=$(wc -l < "$list")
  [[ "$n" -gt 0 ]] || { log "ARRAY  $name: empty list, nothing to submit"; return 0; }
  local cmd; cmd=$(printf '%q ' "$@")
  mkdir -p "$LOGDIR/slurm"
  local jid t0 t1 rc
  t0=$(date +%s)
  log "SBATCH ARRAY $name: $n tasks, %$throttle in flight, each nodes=$nodes ntasks=$ntasks wall=$wall partition=$part"
  jid=$(sbatch $(_sb_common) --job-name="$name" --array="1-${n}%${throttle}" \
          --nodes="$nodes" --ntasks="$ntasks" --time="$wall" --partition="$part" \
          --output="$LOGDIR/slurm/${name}_%A_%a.out" \
          --wait --wrap "cd $(printf '%q' "$GPU_ROOT") && $cmd $(printf '%q' "$list")" 2>>"$LOGFILE")
  rc=$?
  t1=$(date +%s)
  jid="${jid%%;*}"
  log "ARRAY  $name (${jid:-?}) ended rc=$rc after $(human_time $((t1 - t0)))"
  [[ -n "$jid" ]] && job_summary "$jid"
  return $rc
}

# job_summary <jobid> -- per-task exit codes from sacct (best effort)
job_summary() {
  command -v sacct >/dev/null 2>&1 || return 0
  sacct -j "$1" --format=JobID,State,ExitCode,Elapsed,MaxRSS -n -P 2>/dev/null \
    | grep -vE '\.(batch|extern)\|' | awk -F'|' '
      { st[$2]++; if ($2 != "COMPLETED") bad = bad " " $1 "(" $2 "," $3 ")" }
      END { for (s in st) printf "  %s=%d", s, st[s]; print ""; if (bad != "") print "  non-completed:" bad }' \
    | tee -a "$LOGFILE"
}

# array_line <listfile> -- the event for THIS array task
array_line() {
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "not inside a job array (SLURM_ARRAY_TASK_ID unset)"
  sed -n "${SLURM_ARRAY_TASK_ID}p" "$1"
}
