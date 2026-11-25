#!/bin/bash
# Queue of the wave-3 rebuttal experiments.
# Records: results_v3_wave3/<exp>/rec_*.pkl (resumable: existing records are skipped).
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=".venv/bin/python -W ignore -m rebuttal.wave3_experiments"
W=${WORKERS:-9}
# Priority order.  The "all" and "none" knowledge arms are results_v3/know_hard_rev0,
# know_hard_rev30 and know_none (same seeds, same options), so only the new arms run.
$PY kscope_candidates_rev0 kscope_candidates_rev30 --repeats 20 --workers $W
$PY trainfrac_30 trainfrac_40 trainfrac_50 trainfrac_60 trainfrac_70 trainfrac_80 --workers $W
$PY $(for n in 400 1600 6400 25600; do echo nscale_lin_n$n nscale_hubs_n$n; done) --workers $W
$PY hubdose_h0 hubdose_h2 hubdose_h6 --workers $W
$PY hubdose2_h0 hubdose2_h2 hubdose2_h6 --workers $W
$PY nscand_hubs_n1600 nscand_hubs_n25600 --workers $W
$PY sinks_v3 --workers $W
$PY nscalea_lin_n6400 nscalea_hubs_n6400 nscalea_lin_n25600 nscalea_hubs_n25600 nscalea_lin_n102400 nscalea_hubs_n102400 --workers $W
$PY $(for n in 800 3200 12800; do echo nscale_lin_n$n nscale_hubs_n$n; done) hubdose_h1 hubdose_h4 hubdose2_h4 nscand_lin_n1600 nscand_lin_n25600 --workers $W
echo ALL_DONE
