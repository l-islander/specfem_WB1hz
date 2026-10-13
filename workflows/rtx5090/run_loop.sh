#!/bin/bash
# ite3_gpu/run_loop.sh -- autonomous FWI loop on one GPU.
#
# The local port of ite3/run_loop.sh. Same science, same order, same stopping
# rules -- but every `sbatch --wait` + `squeue` poll becomes a direct call, so
# the whole thing is one sequential process you can watch.
#
# ONE ITERATION (model M_x):
#
#   0. forward(M_x) complete?              -> else run it              [GPU]
#   1. cal_adjoint(M_x) full               -> misfit + adjoint sources [CPU]
#      MISFIT GATE: misfit(M_x) > misfit(M_{x-1}) -> STOP
#   2. adjoint(M_x) -> kernels             -> clip/sum/smooth          [GPU]
#      archive the gradient as m{x}to{x+1}
#   3. line search                         -> alpha*                   [GPU]
#      alpha* <= 0 -> STOP without committing (the update would be a no-op)
#   4. commit M_{x+1} at alpha* (--no-clamp, real L-BFGS state)        [CPU]
#   5. forward(M_{x+1})                    -> next iteration's misfit  [GPU]
#   6. stop if alpha* < LS_CONV (converged), ITER == TARGET, or ./STOP
#      else bump ITER and loop
#
# RESUMABLE BY CONSTRUCTION. There is no step counter: every step is skipped when
# its on-disk product exists. Kill it at any point and re-run; it continues.
# To force a step to redo, delete its product (each step logs what that is).
#
# Launch:
#   ./run_loop.sh                    # foreground, watch it
#   nohup ./run_loop.sh &            # background; tail log/run_loop.log
#   tmux new -s fwi './run_loop.sh'  # best: survives a dropped ssh session
#
# Stop gracefully: touch ./STOP  (halts at the next iteration boundary)
# Stop now:        Ctrl-C        (safe; re-run to resume)

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="$ROOT/config.sh"
source "$CONF"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=run_loop
source "$ROOT/lib/common.sh"

HIST="$ROOT/misfit_history.txt"
STATEF="$ROOT/.loop_state"
MARKS="$ROOT/.loop_marks"
STOPF="$ROOT/STOP"
mkdir -p "$MARKS"

# ITER lives in config.sh and is re-read each pass -- config.sh is the single
# source of truth for "which model are we on", exactly as on Anvil.
cur_iter() { grep -E '^ITER=' "$CONF" | head -1 | sed -E 's/^ITER=([0-9]+).*/\1/'; }
bump_iter() { sed -i -E "s/^(ITER=)[0-9]+/\1$1/" "$CONF"; }

# retry_stage <label> <attempts> -- <command...>
#
# The event stages are strict: they refuse to write .stage_done unless EVERY
# event in the set completed, because a missing event changes the misfit basis
# and would silently corrupt the cross-iteration misfit gate. That is the right
# science, but it means one flaky event would end an unattended overnight run.
#
# Since every stage is resumable and skips finished events, simply re-invoking it
# retries only what failed. So retry the stage a couple of times before giving up
# -- transient failures (a GPU hiccup, a momentary OOM from something else on the
# box) self-heal, while a genuine, reproducible failure still stops the loop.
retry_stage() {
  local label="$1" attempts="$2"; shift 2
  [[ "${1:-}" == "--" ]] && shift
  local i
  for (( i = 1; i <= attempts; i++ )); do
    if "$@"; then return 0; fi
    if [[ "$i" -lt "$attempts" ]]; then
      warn "$label failed (attempt $i/$attempts) -- retrying; completed work is skipped"
      sleep 10
    fi
  done
  warn "$label failed after $attempts attempts"
  return 1
}
STAGE_RETRIES="${STAGE_RETRIES:-3}"

# ---------------------------------------------------------------------------
# preflight: fail fast, before an eight-hour stage
# ---------------------------------------------------------------------------
banner "ite3_gpu autonomous FWI loop"
for b in xspecfem3D xgenerate_databases xsmooth_sem; do
  [[ -x "$ROOT/bin/$b" ]] || die "missing $ROOT/bin/$b -- run ./1_build_specfem_gpu.sh"
done
[[ -f "$MESHDATA/.mesh_stamp" ]] || die "no mesh -- run ./2_make_mesh.sh"
[[ -x "$ROOT/cal_adjoint/.venv/bin/python" ]] || die "cal_adjoint venv missing -- run: cd cal_adjoint && bash build_env_local.sh"
command -v nvidia-smi >/dev/null 2>&1 || warn "nvidia-smi not found -- cannot monitor the GPU"

if [[ -f "$STATEF" ]]; then
  source "$STATEF"
  log "RESUME: TARGET_ITER=$TARGET_ITER (config ITER=$(cur_iter))"
else
  TARGET_ITER=$(( $(cur_iter) + MAX_UPDATES ))
  printf 'TARGET_ITER=%d\n' "$TARGET_ITER" > "$STATEF"
  log "START: ITER=$(cur_iter)  MAX_UPDATES=$MAX_UPDATES -> TARGET_ITER=$TARGET_ITER  LS_CONV=$LS_CONV"
fi

LOOP_T0=$(date +%s)

# ---------------------------------------------------------------------------
while :; do
  ITER=$(cur_iter)
  cur="M$ITER"; nxt="M$((ITER + 1))"
  CURMODEL="$MODELBASE/$cur"
  OUT="$MODELBASE/$nxt"
  GRADDIR="$WORKBASE/adjoint/m${ITER}to$((ITER + 1))"
  GRADCHK="$GRADDIR/proc000000_alpha_kernel_clip_smooth.bin"
  RESULT_CUR="$WORKBASE/forward/result_$cur"
  LSALPHA="$WORKBASE/linesearch/ls_alpha.txt"
  # per-iteration sentinel namespace (lib/common.sh keys STATEDIR off ITER)
  STATEDIR="$ROOT/.state/ite${ITER}"; mkdir -p "$STATEDIR"

  if [[ -f "$STOPF" ]]; then
    log "STOP sentinel -> graceful stop at ITER=$ITER"; rm -f "$STOPF"; break
  fi
  if [[ "$ITER" -ge "$TARGET_ITER" ]]; then
    log "reached TARGET_ITER=$TARGET_ITER ($MAX_UPDATES update(s)) -> done"; break
  fi

  hr
  log "=============== ITERATION  model $cur  (target M$TARGET_ITER) ==============="
  hr
  [[ -f "$CURMODEL/.done" ]] || die "model $CURMODEL has no .done sentinel.
     If this is the first iteration, run ./3_make_M0.sh."

  # --- step 0: forward(M_x) -------------------------------------------------
  if [[ -f "$RESULT_CUR/.stage_done" ]]; then
    log "step 0: forward($cur) already complete -> $RESULT_CUR"
  else
    log "step 0: forward($cur)"
    retry_stage "forward($cur)" "$STAGE_RETRIES" -- \
      bash "$ROOT/forward/run_forward.sh" --model "$CURMODEL" --events "$SRC_SET" --out "$RESULT_CUR" \
      || die "forward($cur) failed after $STAGE_RETRIES attempts.
     Some events are reproducibly failing -- see log/forward.log for which.
     Re-run the loop once the cause is fixed; finished events are skipped."
  fi
  ln -sfn "$RESULT_CUR" "$WORKBASE/forward/result"

  # --- REWINDOW=1: fresh windows + representative LS group EVERY iteration --
  # Windows age with the model (one strong update per pick, measured across
  # three campaigns), so re-pick on the CURRENT synthetics each iteration:
  # point CALADJ_WINDIR at an empty per-iteration cache -- run_caladj_local's
  # empty-cache guard then PICKS instead of reusing. The export propagates to
  # every cal_adjoint call this iteration, including the line search's trial
  # misfits, so one basis rules the whole iteration. The misfit gate below
  # must then compare M_x and M_{x-1} measured under the SAME fresh basis
  # (history lines are cross-basis and no longer comparable).
  if [[ "${REWINDOW:-0}" -eq 1 ]]; then
    export CALADJ_WINDIR="$GPU_ROOT/obs_data/windows/${REWINDOW_TAG:-c1}_iter${ITER}"
    mkdir -p "$CALADJ_WINDIR"
    log "REWINDOW: window cache for this iteration -> $CALADJ_WINDIR"
  fi

  # --- step 1+2: gradient ---------------------------------------------------
  if [[ -e "$GRADCHK" ]]; then
    log "step 1-2: gradient $GRADDIR already archived -- skipping cal_adjoint + adjoint"
    log "          (misfit gate is skipped this iteration; delete $GRADDIR to force a redo)"
  else
    # --- 1a cal_adjoint: misfit + adjoint sources ---------------------------
    if [[ ! -f "$MARKS/caladj.$cur" ]]; then
      log "step 1: cal_adjoint($cur)"
      bash "$ROOT/cal_adjoint/run_caladj_local.sh" --full --datadir "$RESULT_CUR" \
        || die "cal_adjoint($cur) failed -- see log/caladj.log"
      mnow=$(awk '{print $1; exit}' "$ROOT/cal_adjoint/misfit_total.txt" 2>/dev/null)
      nmeas=$(awk '{print $2; exit}' "$ROOT/cal_adjoint/misfit_total.txt" 2>/dev/null)
      [[ -n "$mnow" && "$mnow" != "NA" ]] || die "cal_adjoint($cur) produced no usable misfit"
      printf '%s %s %s %(%Y-%m-%d %H:%M:%S)T\n' "$cur" "$mnow" "$nmeas" -1 >> "$HIST"
      touch "$MARKS/caladj.$cur"
      log "step 1: misfit($cur) = $mnow over $nmeas measurements -> $HIST"
    else
      log "step 1: cal_adjoint($cur) already done (marker $MARKS/caladj.$cur)"
    fi

    # --- 1b MISFIT GATE -----------------------------------------------------
    # A one-step monotonicity check against the previous model. Only valid within
    # a FIXED source + window set -- growing coverage or re-picking windows
    # changes the basis and the comparison is meaningless across that change.
    # Under REWINDOW=1 the basis is fresh EVERY iteration, so the gate
    # re-measures the PREVIOUS model's (existing) synthetics under the current
    # windows and compares the same-basis pair instead of trusting history.
    mnow=$(grep -E "^M$ITER "          "$HIST" 2>/dev/null | tail -1 | awk '{print $2}')
    RESULT_PREV="$WORKBASE/forward/result_M$((ITER - 1))"
    if [[ "${REWINDOW:-0}" -eq 1 && "$ITER" -gt 0 && -d "$RESULT_PREV" ]]; then
      PREVMF="$MARKS/prevmisfit.$cur"
      if [[ ! -s "$PREVMF" ]]; then
        log "step 1b: REWINDOW gate -- re-measuring M$((ITER-1)) under this iteration's windows"
        # stale-byproduct hygiene (gotcha #2): the previous result still holds
        # OLD-basis OPT_*/WIN_* artifacts; the misfit sum reads every .adjinfo
        # it finds, so they MUST go before re-measuring.
        for d in "$RESULT_PREV"/*_Z; do
          rm -rf "$d"/DATAP_* "$d"/SYNP_* "$d"/OPT_* "$d"/WIN_*
        done
        bash "$ROOT/cal_adjoint/run_caladj_local.sh" --misfit \
            --datadir "$RESULT_PREV" --out "$PREVMF" \
          || die "REWINDOW gate: re-measurement of M$((ITER-1)) failed"
      fi
      mprev=$(awk '{print $1; exit}' "$PREVMF")
      nprev=$(awk '{print $2; exit}' "$PREVMF")
      log "step 1b: same-basis pair -- $cur=$mnow  M$((ITER-1))=$mprev  (over $nprev pairs, fresh windows)"
    else
      mprev=$(grep -E "^M$((ITER - 1)) " "$HIST" 2>/dev/null | tail -1 | awk '{print $2}')
    fi
    if [[ -n "${mprev:-}" && -n "${mnow:-}" && "$mnow" != NA && "$mprev" != NA ]]; then
      if awk -v a="$mnow" -v b="$mprev" 'BEGIN{exit !(a+0 > b+0)}'; then
        log "MISFIT ROSE: $cur ($mnow) > M$((ITER-1)) ($mprev) -> STOP. $cur is kept as the result."
        break
      fi
      log "step 1b: misfit gate OK -- $cur ($mnow) <= M$((ITER-1)) ($mprev)"
    else
      log "step 1b: misfit gate skipped (no comparable baseline for M$((ITER-1)))"
    fi

    # --- 1c BACKWARD CHECK (diagnostic only) ---------
    # The forward gate scores both models on windows picked on the CURRENT
    # model's synthetics -- selection slightly favors it. The conservative
    # validator of the last commit is the reverse: M$ITER measured under the
    # PREVIOUS iteration's windows (the objective that step optimized), paired
    # with M$((ITER-1))'s step-1 misfit from history (same basis). Never gates.
    # Farm of SYN symlinks so the live result's byproducts (the LS-manifest
    # basis) stay untouched.
    PREVWIN="$GPU_ROOT/obs_data/windows/${REWINDOW_TAG:-c1}_iter$((ITER-1))"
    BACKMF="$MARKS/backcheck.$cur"
    if [[ "${REWINDOW:-0}" -eq 1 && -d "$PREVWIN" && ! -s "$BACKMF" ]]; then
      FARM="$WORKBASE/rewindow_backcheck"
      rm -rf "$FARM"; mkdir -p "$FARM"
      for d in "$RESULT_CUR"/*_Z; do
        s=$(basename "$d"); mkdir -p "$FARM/$s"
        if [[ -d "$d/SYN" ]]; then ln -s "$d/SYN" "$FARM/$s/SYN"; else ln -s "$d" "$FARM/$s/SYN"; fi
      done
      if CALADJ_WINDIR="$PREVWIN" bash "$ROOT/cal_adjoint/run_caladj_local.sh" --misfit \
           --datadir "$FARM" --out "$BACKMF" >>"$LOGFILE" 2>&1; then
        mback=$(awk '{print $1; exit}' "$BACKMF"); nback=$(awk '{print $2; exit}' "$BACKMF")
        mhistprev=$(grep -E "^M$((ITER - 1)) " "$HIST" 2>/dev/null | tail -1 | awk '{print $2}')
        log "step 1c: backward check -- $cur=$mback vs M$((ITER-1))=${mhistprev:-NA} on iter$((ITER-1)) windows ($nback pairs, diagnostic)"
      else
        log "step 1c: backward check FAILED -- non-fatal, continuing"
      fi
      rm -rf "$FARM"
    fi

    # --- 2 adjoint -> kernels -> gradient -----------------------------------
    log "step 2: adjoint($cur) -> kernels"
    retry_stage "adjoint($cur)" "$STAGE_RETRIES" -- \
      bash "$ROOT/adjoint/run_adjoint.sh" --model "$CURMODEL" \
      || die "adjoint($cur) failed after $STAGE_RETRIES attempts -- see log/adjoint.log."
    log "step 2: post-processing (clip -> sum -> smooth)"
    bash "$ROOT/adjoint/run_post.sh" \
      || die "post-processing($cur) failed -- see log/post.log"
    [[ -e "$GRADCHK" ]] || die "post-processing finished but no gradient at $GRADDIR
     (is KERNEL_ARCHIVE=1 in config.sh?)"
    log "step 2: gradient archived -> $GRADDIR"
  fi

  # --- REWINDOW: re-select the line-search group for THIS basis ------------
  if [[ "${REWINDOW:-0}" -eq 1 ]]; then
    LSAUTO="$GPU_ROOT/events/lsauto_iter${ITER}"
    if [[ ! -d "$LSAUTO" ]]; then
      "$ROOT/cal_adjoint/.venv/bin/python" "$ROOT/tools/select_ls_group.py" \
          --result "$RESULT_CUR" --n 40 --out "$LSAUTO" >>"$LOGFILE" 2>&1 \
        || die "REWINDOW: LS-group selection failed"
    fi
    export LS_SUBSET="$LSAUTO"
    log "REWINDOW: LS_SUBSET -> $LSAUTO (coverage+pairs selection on this basis)"
  fi

  # --- step 3: line search --------------------------------------------------
  if [[ -e "$OUT/.done" ]]; then
    log "step 3: $nxt already committed -- skipping line search"
    alpha=$(cat "$LSALPHA" 2>/dev/null || echo "")
  else
    if [[ ! -s "$LSALPHA" ]]; then
      log "step 3: line search($cur)"
      bash "$ROOT/adjoint/forward_update/linesearch.sh" \
          --model "$CURMODEL" --grad "$GRADDIR" --result-mx "$RESULT_CUR" --out "$LSALPHA" \
        || die "line search($cur) failed -- see log/linesearch.log"
    fi
    alpha=$(cat "$LSALPHA" 2>/dev/null || echo "")

    # No descending step: every trial increased the misfit. Committing would
    # produce M_{x+1} == M_x, which is worse than useless -- it would burn an
    # iteration and pollute the L-BFGS history with a zero-curvature pair.
    if [[ -z "$alpha" ]] || awk -v a="$alpha" 'BEGIN{exit !(a+0 <= 0)}'; then
      log "alpha* = ${alpha:-NA}: the line search found NO misfit-reducing step."
      log "  NOT committing $nxt. STOP."
      log "  Likely: the gradient is not a descent direction on this subset, or the"
      log "  current band/windows have been exhausted. Re-tune windows (cal_adjoint"
      log "  --repick), delete $GRADDIR, and relaunch."
      rm -f "$LSALPHA"
      break
    fi
  fi

  # --- step 4: commit -------------------------------------------------------
  if [[ -e "$OUT/.done" ]]; then
    log "step 4: commit($cur -> $nxt) already done"
  else
    log "step 4: commit $cur -> $nxt at alpha* = $alpha (--no-clamp, real L-BFGS state)"
    bash "$ROOT/adjoint/run_update.sh" \
        --model "$CURMODEL" --grad "$GRADDIR" --out "$OUT" \
        --state "$LBFGS_STATE" --step "$alpha" --no-clamp --tag "$cur" \
      || die "commit($cur -> $nxt) failed -- see log/update.log"
    [[ -e "$OUT/.done" ]] || die "commit finished but $OUT/.done is absent"
  fi

  # --- step 5: forward the new model ---------------------------------------
  RESULT_NXT="$WORKBASE/forward/result_$nxt"
  if [[ -f "$RESULT_NXT/.stage_done" ]]; then
    log "step 5: forward($nxt) already complete"
  else
    log "step 5: forward($nxt)"
    retry_stage "forward($nxt)" "$STAGE_RETRIES" -- \
      bash "$ROOT/forward/run_forward.sh" --model "$OUT" --events "$SRC_SET" --out "$RESULT_NXT" \
      || die "forward($nxt) failed after $STAGE_RETRIES attempts -- see log/forward.log."
  fi

  # --- step 6: advance or stop ---------------------------------------------
  # `-n "$alpha"` matters: if $OUT/.done exists but ls_alpha.txt is gone, alpha is
  # empty and awk would evaluate 0 < LS_CONV as TRUE, halting with a bogus
  # "CONVERGED" message that hides the real state.
  if [[ -n "$alpha" ]] && awk -v a="$alpha" -v c="$LS_CONV" 'BEGIN{exit !(a+0 < c+0)}'; then
    # Same stale-alpha hazard as the normal path below: without this, a relaunch
    # after a CONVERGED stop finds a non-empty ls_alpha.txt, skips the line
    # search entirely, and commits this iteration's step length against the next
    # gradient (bit 2026-08-13: M2->M3 reused M1->M2's alpha exactly).
    rm -f "$LSALPHA"
    bump_iter $((ITER + 1))
    log "CONVERGED: alpha* = $alpha < LS_CONV = $LS_CONV."
    log "  $nxt was committed and forwarded. Stopping -- re-tune windows/band to continue."
    break
  fi
  # Delete ls_alpha.txt BEFORE bumping ITER, not after. Killed between the two,
  # the next start would be at ITER+1 with a stale non-empty ls_alpha.txt -- and
  # step 3's guard is `[[ ! -s "$LSALPHA" ]]`, so it would SKIP the line search
  # and commit the previous iteration's step length against a new gradient.
  rm -f "$LSALPHA"
  bump_iter $((ITER + 1))
  log "iteration complete; ITER $ITER -> $((ITER + 1))"
done

# ---------------------------------------------------------------------------
LOOP_T1=$(date +%s)
hr
log "LOOP FINISHED after $(human_time $((LOOP_T1 - LOOP_T0)))"
rm -f "$STATEF"; rm -rf "$MARKS"
if [[ -f "$HIST" ]]; then
  log "misfit history ($HIST):"
  sed 's/^/    /' "$HIST" | tee -a "$LOGFILE"
fi
log "final model: M$(cur_iter) at $MODELBASE/M$(cur_iter)"
hr
exit 0
