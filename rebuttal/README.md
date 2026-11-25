# Rebuttal experiments

Code and results added during the AISTATS 2027 discussion phase. Review items are named as in the reviews (e.g. `Kmr8-Q1`, `AI-3.1`).

| Wave | Scripts | Outputs | Records |
|---|---|---|---|
| 1 | `wave1_summaries.py`, `bound_eval.py`, `decision_eval.py`, `dedup_eval.py` | `out/wave1_summaries.md`, `out/bound_main_v3.csv`, `out/decision_corrected.csv` | `results_v3/` |
| 2 | `experiments_v3.py` (`split_f*`, `latent_n*`, `know_hard_rev*_nolocal`), `wave2_summaries.py`, `wave2_tables.py`, `nosub_baseline.py`, `dedup_exact.py` | `out/wave2_summaries.md`, `out/*.tex` | `results_v3_rebuttal/` |
| 3 | `wave3_experiments.py`, `wave3_nosub.py`, `wave3_analyze.py`, `run_wave3.sh` | `out/wave3/` (one `.csv` and `.md` per section, plus `*_per_run.csv`) | `results_v3_wave3/` |

Raw records (`rec_*.pkl`) are not versioned; every record is regenerated from its seed (`random_seed + repeat`, as in `experiments_v3.py`).

## Wave 3: where each reported number comes from

| Result | Review item | Experiment | Output (row → column) |
|---|---|---|---|
| Hidden hubs vs. the same graphs without confounders: AUROC 0.87 (PC), 0.86 (GES) | Rfqt-W3, xFQE-W3, AC | `sinks_v3` (same DAGs and data as `results_v3/hubs_v3`, 6 sinks hidden instead of 6 hubs) | `out/wave3/sinks.csv` → row "AUROC hubs vs sinks", column `cscd_flag` |
| Flagged control runs have about 3× the skeleton error of kept ones (PC 6.2 vs 2.3; GES 11.8 vs 3.3) | Rfqt-W3, xFQE-W3 | `sinks_v3` | `out/wave3/sinks_gating.csv` → setting "hidden sinks" |
| Without subsets, the check flags none of the hidden-hub runs | AI-3.1, xFQE-W3 | `wave3_nosub.py` on `results_v3/{main,hubs,highdim,lowdim_n1000,quadratic}_v3` | `out/wave3/nosub.csv` → detector "No subsets, full-data run" |
| Shrinking test level: subsets with an error 61% (fixed level, n = 6,400) → 25% (αₙ = 0.05·1600/n, n = 102,400) for FCI; A(G*) = 0 in 20% of runs | d2sj-W4 | `nscale_lin_n*`, `nscalea_lin_n*` | `out/wave3/nscale.csv` → scenario `lin`, method `fci`, columns `eps_train`, `a_gstar_zero` |
| Under hidden hubs the event of Thm. 2 never occurs, for FCI or PC | d2sj-W4 | `nscale_hubs_n*`, `nscalea_hubs_n*` | `out/wave3/nscale.csv` → scenario `hubs`, columns `tau_zero`, `a_gstar_zero` |
| Training fraction: G* kept in 87% (PC) / 90% (FCI) of runs at 0.3, in every run from 0.5 on (30 runs each) | Kmr8-Q1 | `trainfrac_*` | `out/wave3/trainfrac.csv` → column `gstar_wilson` |
| Hard knowledge in the candidates only vs. everywhere: same passing set and claims | Rfqt-Q1 | `kscope_candidates_rev*` vs `results_v3/know_hard_rev*` | `out/wave3/kscope.csv` |
| Soft penalty: in calibration it raises τ and leaves claims nearly unchanged; on candidates only, it shrinks the set | Rfqt-Q1 | offline, `results_v3/know_soft_*` | `out/wave3/lam.csv` |

Further wave-3 outputs: hidden-hub dose-response (`hubdose*.csv`), false and missing adjacencies of admitted graphs as n grows (`nscand.csv`), and claim error as a function of γ (`gamma.csv`).

## Reproduce wave 3

```bash
WORKERS=8 rebuttal/run_wave3.sh                              # all experiments (resumable)
.venv/bin/python -W ignore -m rebuttal.wave3_nosub --workers 8
.venv/bin/python -W ignore -m rebuttal.wave3_analyze          # writes rebuttal/out/wave3/
.venv/bin/python -m rebuttal.wave3_experiments --list         # every experiment and its size
```

Setting `OMP_NUM_THREADS=1` avoids oversubscription with many workers (the queue script sets it).
