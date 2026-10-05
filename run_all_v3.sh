#!/bin/bash
# Second-revision experiments (resumable). Logs to results_v3/run.log.
cd "$(dirname "$0")"
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
W=${N_WORKERS:-9}
mkdir -p results_v3
$PY experiments_v3.py --workers $W \
  main_v3 valid_N9 hubs_v3 lowdim_n1000_v3 quadratic_v3 \
  know_none know_hard_rev0 know_hard_rev10 know_hard_rev20 know_hard_rev30 know_hard_rev40 know_hard_rev50 \
  know_soft_correct know_soft_noisy alpha_N4 alpha_N19 sachs_obs_v3 highdim_v3 \
  valid_N19 valid_N4 scale_p100_mb scale_p200_mb semi_ecoli70 \
  2>&1 | grep --line-buffered -v -i -E "warning|StructureScore|instead\.|get_example_model|^\s*$" >> results_v3/run.log
echo "ALL DONE" >> results_v3/run.log
