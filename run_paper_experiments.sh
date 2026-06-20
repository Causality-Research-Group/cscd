#!/bin/bash
# Full synthetic experiment suite for the UAI revision.
#
# Produces:
#   results/synthetic_main          main table: PC/GES/FCI + baselines + decision task
#   results/knowledge_<reg>_<mode>  domain-knowledge regimes (correct/noisy x hard/soft)
#   results/nsens_N<N>              sensitivity to number of calibration chunks
#                                   (chunk-size discussion)
#
# The no-knowledge reference for the regime table is the main run restricted
# to repeats 0-9 (same seeds, paired comparison).
set -euo pipefail

PY=.venv/bin/python
COMMON="--dataset synthetic --distance_mode skeleton --n 1000 --p 20 \
  --n_subsets 25 --n_candidate_graphs 100 \
  --n_calibration_resamples 20 --n_test_chunks 20 \
  --stability_n_bootstraps 50 --n_workers ${N_WORKERS:-8}"

echo "=== main run (PC/GES/FCI, 30 repeats) ==="
$PY experiments_cscd.py $COMMON \
  --methods pc ges fci --n_repeats 30 \
  --outdir results/synthetic_main

for regime in correct noisy; do
  for mode in hard soft; do
    echo "=== knowledge run: $regime / $mode ==="
    $PY experiments_cscd.py $COMMON \
      --methods pc fci --n_repeats 10 \
      --knowledge_regime $regime --knowledge_mode $mode --lambda_penalty 1.0 \
      --outdir results/knowledge_${regime}_${mode}
  done
done

for N in 10 50; do
  echo "=== N-sensitivity run: N=$N ==="
  $PY experiments_cscd.py --dataset synthetic --distance_mode skeleton \
    --n 1000 --p 20 --n_subsets 25 --n_candidate_graphs 100 \
    --n_calibration_resamples $N --n_test_chunks 20 \
    --stability_n_bootstraps 50 --n_workers ${N_WORKERS:-8} \
    --methods pc --n_repeats 10 \
    --outdir results/nsens_N${N}
done

echo "ALL RUNS DONE"
