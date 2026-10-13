# multi_gpu/lib_multi_gpu.sh -- shared helpers for running many virtual sources on many GPUs.
#
# Builds on the production single-GPU library (workflows/rtx5090/lib/*.sh): the per-source
# solver call (run_forward_event) is the same one that ran 15 validated model updates on the
# RTX 5090 and the forward/kernel timing on the GH200. What this file adds is the multi-GPU
# plumbing around it:
#
#   * ONE mesh database per model, built once (xgenerate_databases bakes the GLL model into
#     proc*_external_mesh.bin), shared READ-ONLY by every GPU through symlinks;
#   * ONE private work directory per source, so any number of sources can run at once, each
#     writing only its own seismograms;
#   * completion markers (.event_done) as the only coordination: a source is done when its
#     marker exists, so any scheduler -- a shell pool, a Slurm array, a Kubernetes/Run:ai
#     indexed job -- can dispatch, retry or backfill sources in any order.
#
# Source this after workflows/rtx5090/{config.sh,env.sh,lib/common.sh,lib/solver.sh}.

[[ -n "${_MGPU_LOADED:-}" ]] && return 0
_MGPU_LOADED=1
: "${_ITE3GPU_SOLVER_LOADED:?source lib/solver.sh before lib_multi_gpu.sh}"

# build_model_database <db_wd> <model_dir>
#   Run xgenerate_databases ONCE for this model (cached by ensure_genmesh's model stamp).
build_model_database() {
  local db="$1" model="$2"
  prepare_workdir "$db"
  ensure_genmesh "$db" "$model"
}

# prepare_source_workdir <wd> <db_wd> <model_dir>
#   A per-source solver directory that reuses the shared database: the baked mesh file is
#   symlinked (read-only), the 300 MB GLL model is copied (see lib/solver.sh for why), and the
#   mesher header the solver reads at start-up is copied.
prepare_source_workdir() {
  local wd="$1" db="$2" model="$3" f
  prepare_workdir "$wd"
  for f in "$db"/DATABASES_MPI/proc*_external_mesh.bin; do
    [[ -e "$f" ]] || die "no baked mesh in $db -- run build_model_database first"
    ln -sfn "$f" "$wd/DATABASES_MPI/$(basename "$f")"
  done
  stage_model "$wd" "$model"
  cp -f "$db/OUTPUT_FILES/values_from_mesher.h" "$wd/OUTPUT_FILES/" \
    || die "missing $db/OUTPUT_FILES/values_from_mesher.h (genmesh side product)"
}

# run_one_source <source_dir> <db_wd> <model_dir> <out_root> [save_forward 0|1]
#   Run one virtual source on the GPU named by GPU_DEVICE (default 0). Under Slurm
#   (--gres=gpu:1) or Kubernetes (nvidia.com/gpu: 1) each task sees exactly one GPU, which is
#   device 0 inside the task. Returns 0 if the source is complete afterwards.
run_one_source() {
  local src="$1" db="$2" model="$3" out="$4" save="${5:-0}"
  local name; name=$(basename "$src")
  local dest="$out/$name" wd="$WORKBASE/mgpu_work/$(basename "$out")/$name"
  event_complete "$dest" && { log "$name: already complete"; return 0; }
  prepare_source_workdir "$wd" "$db" "$model"
  local ok=1 try
  for try in $(seq 0 "${MAX_EVENT_RETRIES:-1}"); do
    if run_forward_event "$wd" "$src" "$dest" "$save"; then ok=0; break; fi
    warn "$name: attempt $try failed on GPU ${GPU_DEVICE:-0}"
  done
  [[ $ok -eq 0 && "$save" -ne 1 ]] && rm -rf "$wd"   # keep failed workdirs, and keep the wavefield when save=1
  return $ok
}
