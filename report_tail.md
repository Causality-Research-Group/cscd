
## 3. Code added on this branch

| File | Purpose |
|---|---|
| `cscd/pipeline.py` | One repeat that stores everything the analyses need:<br>• chunk design (`n_cal_chunks`, `cal_chunk_size`, `train_size`)<br>• fresh held-out chunks<br>• several distances from the same local graphs<br>• local error indicators ε_j (train and chunk)<br>• separation cover<br>• bootstrap graphs on D_train and on D<br>• order-MCMC posteriors on D_train and D<br>• ablations, timings<br>• the decision task |
| `cscd/scenarios.py` | Linear SEM (random or hub latents) and quadratic shared-bias SEM, each with a fresh held-out sample |
| `cscd/decision_v2.py` | Local IDA; set, bootstrap and BMA averaging; coverage of the true effect |
| `cscd/analysis.py` | Directed and adjacency claim metrics, PR summaries, stability selection, Meinshausen–Bühlmann bound, ablations |
| `experiments_v2.py`, `run_all_v2.sh` | 51 named presets (E1–E14); resumable, one pickle per repeat and method |
| `analyze_v2.py` | Per-repeat metrics → `results_v2/<exp>/metrics.csv` and `results_v2/summary/<exp>.csv` |
| `make_report_tables.py`, `make_paper_tables_v2.py`, `make_figures_v2.py` | Report tables, LaTeX tables and PDF figures for the paper |
| `backfill_truth.py` | Adds the true MAG skeleton to records written before it was stored. It regenerates every dataset from its seed and asserts that the regenerated truth matches the record, which also checks reproducibility. |
| `data_files/sachs_observational.txt` | 853-cell observational Sachs data (bnlearn), with loader `load_sachs_observational` |
| `tests/test_pipeline.py`, `tests/test_decision_v2.py` | 14 new tests; the suite now has 50, all passing |

Raw records (`results_v2/*/rec_*.pkl`, about 300 MB in total) are **not** committed, since `results/` is gitignored. The per-repeat `metrics.csv` and the summaries are committed, and the records can be regenerated with `./run_all_v2.sh`.

### Bugs found and fixed during the session
1. **BLAS oversubscription** (load average 34 on 10 cores). Fixed by pinning single-threaded BLAS in `run_all_v2.sh`.
2. **Orphaned worker processes** after an interrupted pool. They were killed. Writes are atomic (`os.replace`), so no record could be corrupted.
3. **Real-data held-out chunks were always empty.** Found by a review subagent: the leftover of D_cal is always smaller than one chunk, so exceedance on Sachs and bnlearn would have been NaN. Held-out chunks are now carved next to the calibration chunks. No real-data record had been produced before the fix.
4. **The quadratic scenario standardised the held-out sample separately.** That was a small exchangeability violation. Train and held-out data now come from one draw that is split afterwards. The fix went in before any quadratic record was written.
5. **Full-graph skeleton errors were counted against the directed claims only**, which ignores bidirected MAG edges. They are now counted against the true MAG skeleton, and records were backfilled.
6. **Failures in the no-subset ablation were dropped silently.** They are now counted in `rec["failures"]["nosub_cal"]`.

## 4. Deviations from the plan, and caveats

- **Held-out exceedance** uses 10 chunks of a fresh sample from the same model (synthetic data) instead of spare chunks of D_cal. With N = 9 the calibration budget leaves no spare chunks, and a fresh sample is a cleaner test of Prop. 1. Real data use 3 extra chunks carved from D_cal.
- **CPSS** was replaced by the Meinshausen–Bühlmann error-bound threshold applied to the same 50 bootstraps (E[V] ≤ 1). Complementary-pairs subsampling was not run.
- **E7:** the adjustment-identification distance (gadjid) was not implemented. It is defined for DAGs/CPDAGs over all variables, not for marginal PAGs on subsets. The asymmetric skeleton weightings were evaluated instead.
- **E4(b) "test misspecification that differs between full and subset runs"** was realised as the high-dimensional scenario (p = 50, n = 1000). Heteroscedastic noise was not run.
- **E11:** 8 of 30 repeats have no confounded treatment–outcome pair in the true DAG and contribute no decision numbers.
- **Repeats:** 30 for main and E4 (20 for p = 50), 20 elsewhere (30 for n ≤ 800); 10 for KCI, 5 for the scaling runs.
- **bnlearn Child, Insurance and Alarm used a reduced protocol:** 10 repeats, 30 candidates, 10 bootstraps, 20 subsets, grid {0.01, 0.1}, no ablations. With the full protocol, G-square PC/FCI took 2–3 hours per repeat; this was spotted mid-run, and the queue was stopped and reordered. Five full-protocol Child records exist but are not used. Asia ran with the full protocol.
- **Two paper-writing subagents** drafted the theory, related work and results text. I checked every number they wrote against the generated tables and corrected five statements: repeat counts, one rate, the distance summary, the adjacency-precision comparison for GES, and the description of the held-out sample. I also removed a sentence that named a reviewer.
- **Literature:** 9 references were verified online and added. Faller et al. (2024) use the ADMG latent projection, give binary notions of compatibility, and provide no calibration. Six items could not be fully verified (Strieder & Drton, Debeire et al., Henckel et al., R²-sortability, Triantafillou & Tsamardinos, and two 2026 arXiv papers); they are **not** cited. Colombo et al.'s exact FCI rate is not quoted.

## 5. Updated abstract (as committed in the paper branch)

The active abstract is in `sections/00-abstract.tex` on the paper's local branch. It follows the plan's §8, with these numbers filled in:
- admission 59/42/78%;
- adjacency precision 0.98–0.995;
- detection 87–90% (PC, GES under hidden hubs) against essentially 0% for bootstrap compatibility.

It is candid that claim precision does not beat voting over the unfiltered pool. The E4 fallback sentence was not needed.

## 6. Paper repository: not pushed

`6ab804def2ecfc179daef8f7` has origin `git.overleaf.com`. Overleaf's git bridge supports a single branch, so a push would overwrite the live Overleaf project. All paper changes are therefore **local commits on branch `experiments/chris`** of that clone (`git log main..experiments/chris`). The paper compiles without errors or undefined references (25 pages with the appendix).

**Page limit.** The main text through the conclusion ends on page 8. The *AI use statement* starts at the bottom of page 8 and runs six lines onto page 9, before the references. Please check whether AISTATS 2027 counts that statement toward the 8 pages. If it does, about 6 lines must still be cut. To publish, after review:

```bash
cd "AISTATS 27/6ab804def2ecfc179daef8f7" && git checkout main && git merge experiments/chris && git push origin main
```

Remaining `\todo` markers in the paper: none (checked with `grep -rn todo sections appendices`).

## 7. Reproducing

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # or: mamba create -p .venv python=3.11
./run_all_v2.sh                                   # all presets, resumable (about 3–4 h on 9 cores)
PYTHONPATH=. .venv/bin/python backfill_truth.py   # only for records written by older code
PYTHONPATH=. .venv/bin/python analyze_v2.py
./make_report.sh                                  # rebuilds report.md, results_v2/tables, paper tables and figures
```
