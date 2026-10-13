#!/bin/bash
# mock end-to-end: mesh -> M0 -> smoke (2 events) on this box via tools/mock_slurm
cd "$(dirname "$0")/../mock_run/tree" || exit 1   # a copy of this tree with env.sh.mock5090 as env.sh
echo "=== chain start $(date)"
while pgrep -f band_snr_survey >/dev/null; do sleep 20; done
echo "=== 2_make_mesh (done) $(date)"
echo "=== 3_make_M0 $(date)";    ./3_make_M0.sh     || { echo "M0 FAILED"; exit 1; }
echo "=== smoke_test $(date)";   ./smoke_test.sh --keep || { echo "SMOKE FAILED"; exit 1; }
echo "=== CHAIN DONE $(date)"
