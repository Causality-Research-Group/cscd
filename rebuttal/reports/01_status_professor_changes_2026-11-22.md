# Rebuttal status as of 22 Nov 2026: changes made since 1 Nov

Scope: every commit dated November 2026 in `aistats2027` (paper) and `cscd` (code), on all branches.
Reviews: AC wG9M (meta-review, leans reject), AI review (Program Chairs), d2sj (3, borderline accept), Kmr8 (4, accept), xFQE (1, reject), Rfqt (2, borderline reject).

## 1. Where the work is

| Repo | Branch | Commits (Nov) | Content |
|---|---|---|---|
| aistats2027 | `main` | a13b8c2, e768a1d, 0437b53, 42fe8b3 (20 Nov) | Reviews added to the repo; Overleaf sync; code folder ignored |
| aistats2027 | `rebuttal_v1` | f6cafe9 → d025d4e (20–22 Nov, 7 commits) | Paper revised with brown revision markup (`\rev`, `revblock`); draft responses in `reviews/aistats/drafts/response-*.md` |
| cscd | `rebuttal_exp` | 9b4539a (21 Nov) | Bug fixes, exact PAG dedup key, rebuttal analysis scripts and their outputs in `rebuttal/out/` |

`aistats2027/chris/revisions` and `cscd/experiments/chris` (October) are already merged into `main` on both repos and were not touched in November. `rebuttal_exp` branches from `cscd/main`, so it contains all October work.

## 2. Code changes (`cscd`, `rebuttal_exp`, 9b4539a)

| Change | File | Review point |
|---|---|---|
| Decision task: CSCD set and "pool" both drawn from the standard 101-graph family, so the CSCD set is a subset of the pool; empty CSCD sets kept as abstentions, not dropped | `cscd/pipeline_v3.py` | Rfqt-Q3 (real error in Table 27) |
| Deterministic outcome draw in the decision task (it depended on Python's string hash seed) | `cscd/decision.py` | Rfqt-Q3 (reproducibility) |
| `mec_key_pag`: exact Markov-equivalence key from the PAG, and a test | `cscd/pipeline_v3.py`, `tests/test_mec_key.py`, `rebuttal/dedup_eval.py` | d2sj-W3 |
| Results root set by environment variable (`CSCD_RESULTS_ROOT`) for reruns in `results_v3_rebuttal/` | `experiments_v3.py` | infrastructure |
| Cluster-bootstrap intervals for exceedance, Wilson intervals over runs, per-run FDP of claims, subsets × calibration decomposition, run gating | `rebuttal/wave1_summaries.py` → `rebuttal/out/wave1_summaries.md` | d2sj-W1, d2sj-W2, Rfqt-Q2, xFQE-W2, Kmr8-W2 |
| False-adjacency bound at a positive threshold, with a numerical check | `rebuttal/bound.md`, `rebuttal/bound_eval.py` → `out/bound_main_v3.csv` | Kmr8-Q2, d2sj-W1 |
| Corrected decision-task table (old vs fixed) | `rebuttal/decision_eval.py`, `decision_table.py` → `out/decision_corrected.csv` | Rfqt-Q3 |
| Report figures | `rebuttal/report_figures.py` | — |

The raw records of the reruns (`results_v3_rebuttal/`) are not on this machine; only their summaries are committed.

## 3. Paper changes (`aistats2027`, `rebuttal_v1`)

| Change | Where | Review point |
|---|---|---|
| New abstract (problem → scale → calibration → exact inclusion condition → hidden-confounder flag rates) | `sections/00-abstract.tex` | Rfqt-W1, xFQE-W1, AC |
| Introduction rewritten: one idea per paragraph, A→B→C running example, "what does a user get", not a confidence set | `sections/01-introduction.tex` | Rfqt-W1/W3, xFQE-W1/W2/W3, Kmr8-W2 |
| Consistent naming ("subset graphs", "passing set", "full-variable training run Ĝ = A(D_train)"), `m0` vs `m = |S|`, λ in Algorithm 1, `G ∈ C ∩ G_K`, `R_∅ := 0`, `γ ∈ (1/2,1]` | `sections/02-method.tex`, algorithm | AI-3.1, AI-3.3(c) |
| Beta law stated conditionally on D_train; `G* ∈ C_α` defined before Thm 1; "at least d_max + 2" | `sections/03-theory.tex` | AI-3.3(a), AI-3.4(a), AI-5 |
| New Proposition: false adjacencies at a positive threshold (Σ c_e ≤ m K τ) | `sections/03-theory.tex`, `appendices/theory.tex` | Kmr8-Q2, d2sj-W1 |
| Lemma 3 condition (λR_K = 0), Corollary 2 clause (Possible-D-SEP), Proposition 4 needs λ = 0 | `appendices/theory.tex` | AI-3.2 |
| Cluster-bootstrap exceedance table; run-gating table; per-run FDP table; local-error table; Figure 4 caption | `appendices/results.tex` | d2sj-W2, Kmr8-W2, d2sj-W1, AI-3.5(d) |
| Conclusion rewritten ("chunks small relative to the training split"; when to use bootstrap vs CSCD); decision claim withdrawn | `sections/06-conclusion.tex`, `tables/rev/decision.tex` | AI-3.4(b), Rfqt-W3/Q3, xFQE-W3 |
| Related work: integration methods (ION, IOD, COmbINE, Bang & Didelez) vs evaluation; Wang et al. asymptotic | `sections/07-related.tex`, `tables/related_work.tex`, `references.bib` | Rfqt-W2, AI-4, AI-3.5(c) |
| Protocol: pair packing, MB subsets, edit sampling, bootstrap own quantile, Sachs PKC, IDA on PAGs, DoWhy license | `appendices/protocol.tex`, checklist | AI-5 |

## 4. Draft responses (`reviews/aistats/drafts/`)

| Reviewer | Status | Promised for the discussion (open) |
|---|---|---|
| AC (`response-PMR.md`) | Drafted: rewrite, theory fixes, working code link, "why read it" | — |
| AI review | Drafted, point by point; Prop. 3 numbers defended as correct | Rerun of the redefined **"Conformal, no subsets"** baseline |
| d2sj | Drafted with new numbers (FDP table, cluster CIs, exact PAG dedup, bound) | **FCI-only scaling study** of local error and P(τ_α = 0) |
| Kmr8 | Drafted (bound, 50/50 rationale, gating) | **Sweep of the training fraction 0.3–0.8** |
| xFQE | Drafted (set is not a confidence set; why a scale; code link; group claims) | — (mainly writing) |
| Rfqt | Drafted (Q3 corrected; new abstract; subsets vs calibration; ION etc.) | Q1 (domain knowledge) answered only with existing numbers; "we agree the title overstates this part" |

## 5. Review points still open after the professor's work

Experimental gaps (addressed in this session, see `02_findings_report.md`):

1. **AI-3.1** – the "Conformal, no subsets" baseline is degenerate; promised rerun not done.
2. **Kmr8-Q1** – training-fraction sweep promised, not run.
3. **d2sj-W4** – FCI scaling study (local error, P(τ=0), inclusion) promised, not run; the existing n-sweep has no hidden confounders, so it cannot separate a latent-aware learner (FCI) from PC/GES.
4. **Rfqt-Q1** (reviewer at 2) – no experiment isolates what the framework adds to knowledge beyond giving it to the base algorithm; the soft penalty λ was never tuned (also listed as "not done" in October).
5. **Rfqt-W3 / Kmr8-W2 / xFQE-W3** – practical value rests on one pooled gating table across heterogeneous settings; no dose-response showing that the flag tracks the amount of systematic error.

Writing gaps (not experimental; for the professor): the full rewrite requested by xFQE/Rfqt/AC is started (abstract, introduction, conclusion) but Sections 2–4 are only edited, not rewritten; `main.tex` inputs a `revision-notes` file that must be removed before submission.
