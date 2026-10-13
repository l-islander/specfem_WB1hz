#!/bin/bash
# Measure an existing forward result against an arbitrary window basis, without
# touching the live result dir.
#
# The result dirs' OPT_*/WIN_* ARE the basis the line search and
# select_ls_group read, so we never measure in place (docs/GOTCHAS.md, item 2).
# Instead we build a symlink farm holding only <src>_Z/SYN -- the sole per-
# workdir input the --misfit path (stages 1->2->3) reads -- and let the run
# write its DATAP_*/SYNP_*/OPT_*/WIN_* into the farm.
#
# Usage: tools/measure_on_basis.sh <model-tag> <window-dir> [ranks]
#   e.g. tools/measure_on_basis.sh M6 obs_data/windows/c1_iter3 8
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:?model tag, e.g. M6}"
WINDIR_REL="${2:?window dir, e.g. obs_data/windows/c1_iter3}"
RANKS="${3:-8}"

RESULT="$ROOT/work/forward/result_$TAG"
WINDIR="$(cd "$ROOT/$WINDIR_REL" && pwd)"
BASIS="$(basename "$WINDIR")"
FARMROOT="$ROOT/work/misfit_$BASIS"
FARM="$FARMROOT/farm_$TAG"
OUT="$FARMROOT/misfit_${TAG}_${BASIS}.txt"

[[ -d "$RESULT" ]] || { echo "no such result dir: $RESULT" >&2; exit 1; }
[[ -d "$WINDIR" ]] || { echo "no such window dir: $WINDIR" >&2; exit 1; }

echo "== $TAG on $BASIS =="
echo "   result : $RESULT"
echo "   windows: $WINDIR"
echo "   farm   : $FARM"

# Fresh farm every time: a stale OPT_* would be summed into the total.
rm -rf "$FARM"
mkdir -p "$FARM"
# A result dir that cal_adjoint has already processed keeps its synthetics in
# <src>_Z/SYN/; one that has only been forward-modelled (e.g. the final model,
# where the loop stopped before measuring) still has the raw .semd at the top
# level. Point the farm's SYN at whichever holds them -- 1_process_data only
# ever READS SYN (it moves .semd solely when SYN is absent, which it is not
# here), and writes its DATAP_*/SYNP_*/OPT_*/WIN_* into the farm workdir.
n=0
for src in "$RESULT"/*_Z; do
  if [[ -d "$src/SYN" ]]; then
    synsrc="$src/SYN"
  elif compgen -G "$src/*.semd" > /dev/null; then
    synsrc="$src"
  else
    continue
  fi
  d="$FARM/$(basename "$src")"
  mkdir -p "$d"
  ln -s "$synsrc" "$d/SYN"
  n=$((n + 1))
done
echo "   sources linked: $n"
[[ $n -gt 0 ]] || { echo "no sources with SYN/ found" >&2; exit 1; }

export CALADJ_WINDIR="$WINDIR"
start=$SECONDS
bash "$ROOT/cal_adjoint/run_caladj_local.sh" --misfit \
     --datadir "$FARM" --out "$OUT" --ranks "$RANKS"
echo "   elapsed: $((SECONDS - start)) s"
echo -n "   misfit : "; cat "$OUT"
