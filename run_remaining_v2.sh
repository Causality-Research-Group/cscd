#!/bin/bash
# Experiments still pending after the stop at ~8 h (resumable).
cd "$(dirname "$0")"
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
W=${N_WORKERS:-9}
mkdir -p results_v2
$PY experiments_v2.py --workers $W \
  latent_n1600 latent_n6400 n12800_N9 scale_p50 scale_p100 kci_quadratic latent_n25600 \
  bnlearn_light_child bnlearn_light_insurance bnlearn_light_alarm \
  2>&1 | grep --line-buffered -v -i -E "warning|StructureScore" >> results_v2/run.log
echo "ALL DONE" >> results_v2/run.log
