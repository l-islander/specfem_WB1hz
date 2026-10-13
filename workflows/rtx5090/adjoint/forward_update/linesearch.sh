#!/bin/bash
# ite3_gpu/adjoint/forward_update/linesearch.sh -- parabolic step-length search.
#
# Replaces ite3's linesearch.sh + run_scan_anvil.slurm + run_forward_ls_anvil.slurm.
#
# WHY A LINE SEARCH AT ALL
# ------------------------
# ite2 overshot: a fixed 3% cap on |ln-model change| was applied every iteration
# regardless of whether 3% was too far, and the misfit rose. The fix is to
# measure. For each trial step a in LS_STEPS, build the candidate model, forward
# it over a 40-event subset, measure the misfit, then fit a parabola through
# (0, misfit_0) and the trial points and take its vertex.
#
# WHAT --no-clamp IS FOR (the subtle bit)
# ---------------------------------------
# The default L-BFGS step is a = min(1, max_pert/max|d|). Once the L-BFGS
# direction is well scaled, max_pert/max|d| exceeds 1 for every trial value, so
# every candidate collapses to a = 1 and all three "different" models come out
# IDENTICAL -- the parabola then gets fitted to measurement noise. With
# --no-clamp, a = max_pert/max|d| exactly, so the trial value IS the step length
# and the candidates genuinely differ.
#
# EACH TRIAL USES A THROWAWAY COPY OF THE L-BFGS STATE
# ----------------------------------------------------
# lbfgs_update.py appends a snapshot to the state every time it runs. If the
# trials wrote to the real state, three spurious curvature pairs would be
# recorded before the real step was ever committed. So each candidate gets a
# copy, and only run_loop's final commit touches the persistent state.
#
# COST ON ONE GPU: len(LS_STEPS) x len(LS_SUBSET) forwards, run sequentially.
# With 3 steps and 40 events that is 120 forwards -- typically the single most
# expensive part of an iteration. Trim LS_STEPS or LS_SUBSET if it dominates.
#
# Usage:
#   ./linesearch.sh --model DIR --grad DIR --result-mx DIR [--out FILE]
# Writes alpha* (a bare number) to --out (default ls_alpha.txt).

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$ROOT/config.sh"
[[ -r "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
STAGE=linesearch
source "$ROOT/lib/common.sh"
source "$ROOT/lib/solver.sh"

# Default OUT lives with the rest of the line-search scratch, NOT next to this
# script -- run_loop.sh and status.sh both look in $WORKBASE/linesearch, so a
# manual run without --out would otherwise write somewhere status.sh never reads.
MODEL=""; GRAD=""; RESULT_MX=""; OUT="$WORKBASE/linesearch/ls_alpha.txt"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)     MODEL="$2"; shift ;;
    --grad)      GRAD="$2"; shift ;;
    --result-mx) RESULT_MX="$2"; shift ;;
    --out)       OUT="$2"; shift ;;
    -h|--help)   sed -n '2,36p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac; shift
done
: "${MODEL:?--model is required}"
: "${GRAD:?--grad is required}"
: "${RESULT_MX:?--result-mx is required}"

LSWORK="$WORKBASE/linesearch"
SCAN="$LSWORK/MODEL_SCAN"
PTS="$LSWORK/ls_points.txt"

banner "LINE SEARCH  model=$(basename "$MODEL")  steps='$LS_STEPS'"

[[ -f "$MODEL/.done" ]] || die "model $MODEL has no .done sentinel"
[[ -e "$GRAD/proc000000_alpha_kernel_clip_smooth.bin" ]] || die "no smoothed gradient at $GRAD"

mapfile -t SUBSET < <(list_events "$LS_SUBSET")
[[ "${#SUBSET[@]}" -gt 0 ]] || die "no events in LS_SUBSET=$LS_SUBSET"
NSTEPS=$(wc -w <<<"$LS_STEPS")
log "subset: ${#SUBSET[@]} events; $NSTEPS trial step(s) => $(( NSTEPS * ${#SUBSET[@]} )) forwards"

# ---------------------------------------------------------------------------
# stale-scratch guard
# ---------------------------------------------------------------------------
# Without this, iteration x+1 silently reuses iteration x's candidate models
# (they live at the same paths) and line-searches the WRONG gradient. ite3 hit
# exactly this bug going M3 -> M4.
mkdir -p "$LSWORK"
STAMP="$LSWORK/.ls_model"
if [[ -f "$STAMP" && "$(cat "$STAMP")" != "$MODEL" ]]; then
  log "scratch belongs to $(basename "$(cat "$STAMP")"), now searching $(basename "$MODEL") -- wiping"
  rm -rf "$SCAN" "$LSWORK"/Result_* "$LSWORK/step0_misfit.txt" "$PTS" "$OUT" "$LSWORK/.state_tmp" \
         "$LSWORK/kernel_pairs.manifest"   # REWINDOW: basis changes every iteration -- a stale manifest silently shrinks the matched-pair set
fi
printf '%s\n' "$MODEL" > "$STAMP"

take_lock

# ---------------------------------------------------------------------------
# step 1: build the candidate models
# ---------------------------------------------------------------------------
log ""
log "== 1. building $NSTEPS candidate model(s) =="
for a in $LS_STEPS; do
  cand="$SCAN/pert_$a"
  if [[ -f "$cand/.done" ]]; then log "  pert_$a already built"; continue; fi
  tmpstate="$LSWORK/.state_tmp/pert_$a"
  rm -rf "$tmpstate"; mkdir -p "$tmpstate"
  # throwaway copy of the real state -- see the header note
  if compgen -G "$LBFGS_STATE/*" >/dev/null; then cp -r "$LBFGS_STATE"/. "$tmpstate"/; fi
  log "  pert_$a  (--no-clamp, throwaway state)"
  bash "$ROOT/adjoint/run_update.sh" \
      --model "$MODEL" --grad "$GRAD" --out "$cand" \
      --state "$tmpstate" --step "$a" --no-clamp --tag "ls_$a" \
    || die "candidate model build failed at step $a"
done
rm -rf "$LSWORK/.state_tmp"

# ---------------------------------------------------------------------------
# step 2: forward each candidate over the subset
# ---------------------------------------------------------------------------
log ""
log "== 2. forward each candidate over the ${#SUBSET[@]}-event subset =="
for a in $LS_STEPS; do
  cand="$SCAN/pert_$a"
  res="$LSWORK/Result_$a"
  if [[ -f "$res/.stage_done" ]]; then log "  Result_$a already complete"; continue; fi
  log "  forwarding pert_$a"
  bash "$ROOT/forward/run_forward.sh" --model "$cand" --events "$LS_SUBSET" --out "$res" \
    || die "trial forward failed at step $a"
done

# ---------------------------------------------------------------------------
# step 3: misfit for each candidate, plus step 0
# ---------------------------------------------------------------------------
log ""
log "== 3. misfit =="

# Step 0 is the CURRENT model's misfit ON THE SAME SUBSET. It is already
# measured -- cal_adjoint left .adjinfo files in the full result -- so reuse them
# rather than re-forwarding 40 events. Restricting to the subset sources is what
# makes it comparable with the trial misfits.
# ---------------------------------------------------------------------------
# KERNEL-PAIR MANIFEST. Every line-search point -- step 0 AND every trial --
# is summed over the SAME pair list: the post-decimation pairs that actually
# built the gradient, restricted to the subset sources. Without this, the
# trial cal_adjoint runs re-run decimate_pairs_by_cell on the 40-source pool
# and keep pairs the 200-source run never measured (cell winners change when
# the pool shrinks): trials then carry a constant positive misfit offset the
# anchor lacks, the parabola inverts, and alpha* comes out 0 against a
# genuinely descending direction. (Observed 2026-07-30: anchor n=3466,
# trials n=3483, +4.7 fake rise at step 0.01.)
# ---------------------------------------------------------------------------
MANIFEST="$LSWORK/kernel_pairs.manifest"
if [[ ! -s "$MANIFEST" ]]; then
  for s in "${SUBSET[@]}"; do
    n=$(basename "$s")
    find "$RESULT_MX/$n"/OPT_* -name '*.adjinfo' 2>/dev/null | sed "s|^$RESULT_MX/||"
  done | sort > "$MANIFEST"
fi
NMAN=$(wc -l < "$MANIFEST")
[[ "$NMAN" -gt 0 ]] || die "empty kernel-pair manifest -- has cal_adjoint measured $RESULT_MX?"
log "  kernel-pair manifest: $NMAN pairs (the gradient's own pairs, subset sources)"

# sum_misfit_manifest <resultdir> -> "sum count" over manifest pairs present
# there. Single awk, getline per file: robust to xargs batching and to
# .adjinfo files without trailing newlines.
sum_misfit_manifest() {
  awk -v root="$1" '
    { fn=root "/" $0
      if ((getline rec < fn) > 0) { split(rec, f, /[ \t]+/); s += f[2]+0; c++ }
      close(fn) }
    END { printf "%.10g %d", s, c }' "$MANIFEST"
}

M0FILE="$LSWORK/step0_misfit.txt"
if [[ -s "$M0FILE" ]]; then
  read -r m0 n0 < "$M0FILE"
  log "  step 0 (reused): $m0 over $n0 manifest pairs"
else
  read -r m0 n0 < <(sum_misfit_manifest "$RESULT_MX")
  [[ -n "$m0" && "$n0" -gt 0 ]] || die "could not determine the step-0 misfit"
  printf '%s %s\n' "$m0" "$n0" > "$M0FILE"
  log "  step 0: $m0 over $n0 manifest pairs"
fi

: > "$PTS"
printf '0 %s\n' "$m0" >> "$PTS"

for a in $LS_STEPS; do
  res="$LSWORK/Result_$a"
  mf="$res/misfit_total.txt"
  if [[ ! -s "$mf" ]]; then
    log "  measuring step $a"
    bash "$ROOT/cal_adjoint/run_caladj_local.sh" --misfit --datadir "$res" --out "$mf" \
      || die "misfit measurement failed at step $a"
  fi
  # The trial run measures its own (re-decimated) pair set; for the parabola we
  # sum ONLY the kernel-pair manifest so all points share one basis. mf is kept
  # as the raw record.
  read -r m n < <(sum_misfit_manifest "$res")
  [[ -n "$m" && "$n" -gt 0 ]] || die "no manifest-pair misfit for step $a"
  if [[ "$n" -ne "$NMAN" ]]; then
    warn "  step $a: only $n of $NMAN manifest pairs measured -- parabola may be biased"
  fi
  raw=$(awk '{print $1; exit}' "$mf")
  log "  step $a: misfit=$m over $n manifest pairs   (raw run total: $raw)"
  printf '%s %s\n' "$a" "$m" >> "$PTS"
done

# ---------------------------------------------------------------------------
# step 4: fit
# ---------------------------------------------------------------------------
log ""
log "== 4. parabolic fit =="
log "  points (step misfit):"
sed 's/^/    /' "$PTS" | tee -a "$LOGFILE"

PYBIN="$ROOT/cal_adjoint/.venv/bin/python"
[[ -x "$PYBIN" ]] || PYBIN=python3
alpha=$("$PYBIN" "$HERE/polyfit.py" "$PTS") || die "polyfit failed"
[[ -n "$alpha" ]] || die "polyfit produced no alpha"
printf '%s\n' "$alpha" > "$OUT"

hr
log "alpha* = $alpha   -> $OUT"
# Interpretation is run_loop's job, but say what the number means here too.
if awk -v a="$alpha" 'BEGIN{exit !(a+0 <= 0)}'; then
  log "  alpha* <= 0: NO trial step reduced the misfit on the subset."
  log "  Committing would give M_{x+1} == M_x. Re-tune windows/band or check the gradient."
elif awk -v a="$alpha" -v c="$LS_CONV" 'BEGIN{exit !(a+0 < c+0)}'; then
  log "  alpha* < LS_CONV=$LS_CONV: converged for this band/window set."
else
  log "  healthy descending step."
fi
hr
