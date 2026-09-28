#!/bin/bash
# Run every revision experiment (resumable). Logs to results_v2/run.log.
cd "$(dirname "$0")"
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
W=${N_WORKERS:-9}
mkdir -p results_v2
$PY experiments_v2.py --workers $W \
  main_v2 design_N20_c40 design_N19 design_N9 design_sym_N9 \
  sys_control sys_latent_hubs sys_highdim sys_wrongK sys_quadratic \
  n200_N9 n400_N9 n800_N9 n1600_N9 n6400_N9 \
  distances sub_k0.3_m10 sub_k0.3_m25 sub_k0.3_m50 sub_k0.5_m10 sub_k0.5_m25 sub_k0.5_m50 \
  sub_k0.7_m10 sub_k0.7_m25 sub_k0.7_m50 sub_k0.5_m171 sub_k0.7_m33 sub_mb_k0.5 sub_mb_k0.3 sub_k0.3_m10_nocover \
  knowledge_none knowledge_correct_hard knowledge_correct_soft knowledge_noisy_hard knowledge_noisy_soft \
  latent_random large_p50 nongauss_laplace sachs_obs sachs_pooled \
  bnlearn_asia bnlearn_child bnlearn_insurance bnlearn_alarm \
  latent_n1600 latent_n6400 n12800_N9 scale_p50 scale_p100 latent_n25600 kci_quadratic \
  2>&1 | grep --line-buffered -v -i -E "warning|StructureScore" >> results_v2/run.log
echo "ALL DONE" >> results_v2/run.log
