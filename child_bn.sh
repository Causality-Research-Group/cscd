#!/bin/bash

python experiments_cscd.py \
  --dataset child \
  --methods pc ges fci \
  --n 5000 \
  --n_repeats 20 \
  --distance_mode skeleton \
  --outdir results/child