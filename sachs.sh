#!/bin/bash

python experiments_cscd.py \
  --dataset sachs_real \
  --methods pc ges fci \
  --n_repeats 30 \
  --n_subsets 25 \
  --n_calibration_resamples 100 \
  --n_candidate_graphs 100 \
  --outdir results/sachs
