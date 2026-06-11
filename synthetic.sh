#!/bin/bash

python experiments_cscd.py \
  --dataset synthetic \
  --methods pc ges fci \
  --distance_mode skeleton \
  --n 1000 \
  --p 20 \
  --n_repeats 30 \
  --n_subsets 25 \
  --n_candidate_graphs 100 \
  --n_calibration_resamples 100 \
  --outdir results/synthetic