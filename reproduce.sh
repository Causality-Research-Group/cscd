#!/bin/bash
# Reproduce every experiment of the paper, then its tables and figures.
#
# Usage:
#   ./reproduce.sh                  # everything (about 15 h on 9 cores)
#   N_WORKERS=4 ./reproduce.sh      # change the number of parallel workers
#   ./reproduce.sh tables           # only rebuild tables and figures from existing records
#
# Two drivers are used.  experiments_v2.py holds the design sweeps (chunk
# design, sample size, subsets, distances, hidden-variable sweep, robustness,
# bnlearn); experiments_v3.py holds the validity study, the main experiment
# with the candidate-family arms, multi-chain order MCMC and the external
# detectors, the detection scenarios, domain knowledge, the alpha sweep,
# scalability, the semi-synthetic benchmark and Sachs.  Both are resumable:
# finished records are skipped.  `--list` on either driver prints its presets.
#
# Set up the environment first:
#   python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=.venv/bin/python
W=${N_WORKERS:-9}
PAPER=${PAPER_ROOT:-../aistats2027}

if [[ "${1:-}" != "tables" ]]; then
  $PY experiments_v2.py --workers "$W" \
    main_v2 design_N20_c40 design_N19 design_N9 design_sym_N9 \
    n200_N9 n400_N9 n800_N9 n1600_N9 n6400_N9 n12800_N9 distances \
    sub_k0.3_m10 sub_k0.3_m25 sub_k0.3_m50 sub_k0.5_m10 sub_k0.5_m25 sub_k0.5_m50 \
    sub_k0.7_m10 sub_k0.7_m25 sub_k0.7_m50 sub_k0.5_m171 sub_k0.7_m33 sub_mb_k0.5 sub_mb_k0.3 sub_k0.3_m10_nocover \
    latent_random latent_n1600 latent_n6400 latent_n25600 sys_latent_hubs large_p50 nongauss_laplace kci_quadratic \
    bnlearn_asia bnlearn_light_child bnlearn_light_insurance bnlearn_light_alarm
  ./run_all_v3.sh
fi

$PY analyze_v2.py
$PY analyze_v3.py
$PY make_paper_tables_v2.py --out "$PAPER/tables/v2"
$PY make_tables_v3.py --paper_root "$PAPER"
$PY make_figures_v2.py --out "$PAPER/figures/results_v2"
$PY make_figures_v3.py --paper_root "$PAPER"
