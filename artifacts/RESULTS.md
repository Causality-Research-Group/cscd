# Empirical results for the UAI revision — mapping to reviewer items

All numbers from `results/` (regenerate with `./run_paper_experiments.sh`, then
`python make_paper_artifacts.py`). Protocol: synthetic linear-Gaussian SCMs,
p=20, n=1000, expected degree 2, 30 repeats (10 for knowledge/N-sensitivity
runs), m=25 subsets, N=20 calibration chunks + 20 held-out test chunks,
100 candidate graphs, skeleton distance.

## HIGH: missing baselines (Table 1 rows) — CLOSED
`table_baselines.tex`. Accepted-claim precision exceeds single-run precision
for every base method at α=0.10:

| Method | CSCD accepted | Single run | Stability sel. | Uncalibrated SC |
|--------|---------------|------------|----------------|-----------------|
| PC     | 0.781 ± 0.024 | 0.648 ± 0.021 | 0.751 ± 0.026 | 0.637 ± 0.052 |
| GES    | 0.961 ± 0.014 | 0.607 ± 0.025 | 0.735 ± 0.027 | 0.534 ± 0.071 |
| FCI    | 0.596 ± 0.012 | 0.517 ± 0.013 | 0.721 ± 0.016 | 0.540 ± 0.036 |

The abstract's claim ("accepted claims achieve higher precision than
single-run discovery") is now substantiated; the recall cost is reported in
the same table. GES results exist (closes the "GES listed but absent" item).

## HIGH: domain-knowledge regimes (research question iii) — CLOSED
`table_knowledge.tex`, `figure_knowledge.pdf` (10 repeats, λ=1.0,
knowledge_frac=0.3, knowledge_noise=0.3). At α=0.10:
- Correct hard knowledge raises accepted precision (PC 0.802→0.850,
  FCI 0.598→0.680) and lowers the unresolved rate (PC 0.072→0.060).
- Noisy hard constraints cost recall (PC 0.577→0.526, FCI 0.875→0.803).
- The soft penalty under noisy knowledge preserves recall (PC 0.577,
  FCI 0.875) at no-knowledge precision — supporting "soft penalty preferable
  to hard constraints when expert information is unreliable."

## HIGH: soft-penalty miscalibration — CLOSED (code-level)
With knowledge_mode=soft, λR_K now enters BOTH the graph-level score and the
per-chunk calibration scores (penalty of calibration local graphs against
restricted knowledge, averaged over subsets), so the threshold is calibrated
against the same score family used for admission. For λ=0 the previous
behavior is unchanged. The paper can state the guarantee for the penalized
score family rather than scoping to λ=0.

## MEDIUM: identical PC/FCI numbers in Table 1 — EXPLAINED + FIXED
Two separate causes:
1. **Exceedance was computed on the calibration scores that define τ_α**,
   making it ⌊α(N+1)⌋/N for every method — a genuine bug, now fixed by
   evaluating on held-out test chunks. New exceedance (30 repeats):
   PC 0.050/0.088/0.200, GES 0.060/0.098/0.175, FCI 0.058/0.107/0.205 at
   α=0.05/0.10/0.20 — method-dependent, within 2 SE of nominal everywhere.
2. **Self-compatibility 0.092 vs 0.092** is a rounded-mean coincidence, not
   copy-paste: per repeat the PC and FCI scores are never exactly equal
   (mean |Δ| = 0.017) but are highly correlated because FCI's adjacency
   phase coincides with PC's under the skeleton distance.

## MEDIUM: chunk-size / sample-size mismatch — EMPIRICAL ANSWER
`table_nsens.tex`: increasing N from 10→50 shrinks chunks and raises τ
(0.136→0.174) while exceedance approaches nominal; accepted precision/recall
and unresolved rate are stable (prec 0.81/0.80/0.83). So the threshold scale
reflects chunk-size instability, but the induced claim partition is
insensitive to N — a one-paragraph discussion plus this table answers the
critique.

## Workshop fit: decision task — CLOSED
`figure_decision.pdf`, columns in `summary_digest.csv`. Treatment–outcome
pairs restricted to confounded pairs (a parent of T remains an ancestor of Y
after removing T), absolute bias of OLS total-effect estimates, α=0.10:

| Method | Oracle | CSCD set-avg | CSCD not-rejected | CSCD accepted | Single run | Naive |
|--------|--------|--------------|-------------------|---------------|-----------|-------|
| PC     | 0.057  | **0.309**    | 0.328             | 0.396         | 0.357     | 0.500 |
| GES    | 0.059  | **0.290**    | 0.337             | 0.446         | 0.291     | 0.493 |
| FCI    | 0.056  | **0.387**    | 0.438             | 0.362         | 0.392     | 0.482 |

Headline: averaging the effect estimate over the calibrated set C_α matches
or beats the single discovered graph for all methods and clearly beats no
adjustment. Honest caveats for the paper: the GES margin over single-run is
within noise (0.290 vs 0.291), and strict accepted-only adjustment
under-adjusts for PC/GES because conservative claim sets omit unresolved
confounders — that asymmetry (omission bias vs inclusion harmlessness) is
itself a useful observation for the decision-making discussion.

## Calibration figure
`figure_calibration.pdf` replaces the old version: exceedance vs α per
method with ±1 SE error bars, print-legible at single-column width.

## Repro
- `run_paper_experiments.sh` — all runs (≈15 min on 8 workers).
- `make_paper_artifacts.py` — tables/figures/CHECKS from `results/`.
- Raw per-run rows: `results/*/summary.csv`; aggregated: `artifacts/summary_digest.csv`.
