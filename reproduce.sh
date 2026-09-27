#!/bin/bash
# Reproduce every experiment in the paper, then build its tables and figures.
#
# Usage:
#   ./reproduce.sh                 # run everything
#   ./reproduce.sh main latent     # run only the named experiments
#   N_WORKERS=4 ./reproduce.sh     # change the number of parallel workers
#
# Experiments (results go to results/<name>):
#   main        PC, GES, FCI on synthetic linear-Gaussian data (p=20), with all
#               baselines, the localization ablations, and the decision task
#   knowledge   correct / noisy domain knowledge, as hard or soft constraints
#   chunksize   calibration chunks of 40, 80, 120 samples at a fixed training size
#   samplesize  n in {800, 1600, 3200, 6400}
#   subsets     subset size, number of subsets, and pair coverage
#   distance    marginal-PAG distance instead of marginal-skeleton distance
#   large       p = 50
#   nongauss    Laplace noise, PC and DirectLiNGAM
#   latent      20% of the variables hidden, PC and FCI
#   sachs       Sachs protein-signalling data (continuous)
#   bnlearn     Asia, Child, Insurance, and Alarm networks (discrete)
#
# Set up the environment first:
#   python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
WORKERS=${N_WORKERS:-8}

# Shared protocol.  Calibration uses N=20 chunks and exceedance is measured
# on 20 more held-out chunks.  n=3200 with a 50/50 split gives 40 chunks of
# 40 samples each.
PROTOCOL="--n_calibration_resamples 20 --n_test_chunks 20 \
  --n_candidate_graphs 100 --stability_n_bootstraps 50 \
  --distance_mode skeleton --n_workers $WORKERS"
SYNTHETIC="--dataset synthetic --p 20 --n 3200 --n_subsets 25 --cal_chunk_size 40"

run() {
  local name=$1; shift
  echo "=== $name: $*"
  $PY experiments_cscd.py "$@" --outdir "results/$name"
}

exp_main() {
  run main $SYNTHETIC $PROTOCOL --methods pc ges fci --n_repeats 30
}

exp_knowledge() {
  # Same seeds as the "none" run, so the regimes are paired.
  run knowledge_none $SYNTHETIC $PROTOCOL --methods pc fci --n_repeats 10 \
    --no_bayesian --no_ablations
  for regime in correct noisy; do
    for mode in hard soft; do
      run "knowledge_${regime}_${mode}" $SYNTHETIC $PROTOCOL --methods pc fci --n_repeats 10 \
        --knowledge_regime $regime --knowledge_mode $mode --lambda_penalty 1.0 \
        --no_bayesian --no_ablations
    done
  done
}

exp_chunksize() {
  # n_train fixed at 1600; D_cal has 4800 samples, enough for 40 chunks of 120.
  for c in 40 80 120; do
    run "chunksize_c$c" --dataset synthetic --p 20 --n 6400 --train_size 1600 --n_subsets 25 \
      --cal_chunk_size $c $PROTOCOL --methods pc ges fci --n_repeats 10 --no_bayesian
  done
}

exp_samplesize() {
  # The 40 calibration and test chunks split D_cal evenly, so chunks hold
  # n/80 samples: 10, 20, 40, and 80.
  for n in 800 1600 3200 6400; do
    run "samplesize_n$n" --dataset synthetic --p 20 --n $n --n_subsets 25 $PROTOCOL \
      --methods pc ges fci --n_repeats 10
  done
}

exp_subsets() {
  for frac in 0.3 0.5 0.7; do
    for m in 10 25 50; do
      run "subsets_k${frac}_m${m}" --dataset synthetic --p 20 --n 3200 --cal_chunk_size 40 \
        --subset_frac $frac --n_subsets $m $PROTOCOL --methods pc fci --n_repeats 10 \
        --no_bayesian
    done
  done
  run subsets_k0.3_m10_nocover --dataset synthetic --p 20 --n 3200 --cal_chunk_size 40 \
    --subset_frac 0.3 --n_subsets 10 --no_pair_coverage $PROTOCOL --methods pc fci \
    --n_repeats 10 --no_bayesian
}

exp_distance() {
  run distance_pag $SYNTHETIC $PROTOCOL --distance_mode pag --methods pc fci --n_repeats 10 \
    --no_bayesian
}

exp_large() {
  run large_p50 --dataset synthetic --p 50 --n 3200 --n_subsets 50 --cal_chunk_size 40 \
    $PROTOCOL --methods pc fci --n_repeats 10
}

exp_nongauss() {
  run nongauss_laplace $SYNTHETIC $PROTOCOL --noise laplace --methods pc lingam --n_repeats 10
}

exp_latent() {
  # 25 variables, 5 hidden: 20 observed, as in the main experiment.
  run latent_20pct --dataset synthetic --p 25 --latent_fraction 0.2 --n 3200 --n_subsets 25 \
    --cal_chunk_size 40 $PROTOCOL --methods pc fci --n_repeats 10
}

exp_sachs() {
  run sachs --dataset sachs_real --n_subsets 15 $PROTOCOL --methods pc ges fci --n_repeats 10
}

exp_bnlearn() {
  # Discrete data: G-square tests; the Gaussian Bayesian baseline does not apply.
  for net in asia child insurance alarm; do
    run "bnlearn_$net" --dataset $net --n 4000 --n_subsets 40 --independence_test gsq \
      $PROTOCOL --methods pc fci --n_repeats 10 --no_bayesian
  done
}

ALL="main chunksize knowledge samplesize subsets distance large nongauss latent sachs bnlearn"
for exp in ${@:-$ALL}; do
  "exp_$exp"
done

$PY make_paper_artifacts.py --results_root results --paper_root ../..
echo "ALL DONE"
