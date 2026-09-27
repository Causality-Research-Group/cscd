import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from .config import ExperimentConfig, DomainKnowledge
from .data import (
    generate_random_dag,
    hide_latents,
    true_observed_graph,
    load_bnlearn_pgmpy_dataset,
    load_causallearn_sachs,
    load_real_csv,
    sachs_reference_graph,
    simulate_linear_sem,
)
from .utils import (
    corrupt_knowledge,
    knowledge_from_true_graph,
    make_domain_knowledge,
)
from .mixed_graph import from_dag_adjacency, to_adjacency
from .metrics import accepted_precision_recall, shd, skeleton_f1, status_counts, summarize_claims
from .decision import decision_task_summary
from .plotting import make_summary_table, plot_calibration, plot_claim_rates, plot_precision_recall, plot_raw_self_compatibility, plot_score_distributions, plot_top_claims, save_and_plot
from .selfcompat import (
    calibration_scores,
    candidate_graph_score,
    conformal_graph_set,
    conformal_threshold,
    generate_candidate_graphs,
    project_local_graphs,
    sample_variable_subsets,
    self_compatibility_score_among_local_graphs,
    train_local_graphs,
)
from .baselines import (
    ablation_columns,
    baseline_summary,
    bootstrap_compatibility,
    bootstrap_marks,
    bootstrap_threshold,
    conformal_without_subsets,
)


def _stable_method_offset(method: str) -> int:
    return sum((ord(ch) * (i + 1) for i, ch in enumerate(method))) % 100_000


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _run_one_task(task: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_one(**task)


def run_one(
    data: pd.DataFrame,
    knowledge: DomainKnowledge,
    config: ExperimentConfig,
    method: str,
    dataset: str,
    repeat: int,
    true_graph: Optional[pd.DataFrame] = None,
    distance_mode: str = "skeleton",
    rng_seed: Optional[int] = None,
    soft_knowledge: Optional[DomainKnowledge] = None,
    sem_weights: Optional[np.ndarray] = None,
    true_full_dag: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One repeat of CSCD for one base method.

    ``true_graph`` holds the true parent claims among the observed variables
    and is used for claim-level metrics.  ``true_full_dag`` is the full
    data-generating DAG, possibly with hidden variables; when given, its
    score A(G*) is computed to check whether the truth passes the threshold.
    """
    if rng_seed is None:
        rng_seed = config.random_seed + repeat * 1000 + _stable_method_offset(method)
    rng = _make_rng(rng_seed)

    lam = config.lambda_penalty if soft_knowledge is not None else 0.0

    var_names = list(data.columns)
    if config.train_size is not None:
        # Fixed absolute D_train size; D_cal gets the remainder.
        D_tr, D_cal = train_test_split(
            data,
            train_size=config.train_size,
            shuffle=True,
            random_state=config.random_seed + repeat,
        )
    else:
        D_tr, D_cal = train_test_split(
            data,
            test_size=0.5,
            shuffle=True,
            random_state=config.random_seed + repeat,
        )
    D_tr = D_tr.reset_index(drop=True)
    D_cal = D_cal.reset_index(drop=True)

    subsets = sample_variable_subsets(
        var_names,
        m=config.n_subsets,
        subset_frac=config.subset_frac,
        rng=rng,
        require_pair_coverage=config.require_pair_coverage,
    )

    local_graphs = train_local_graphs(
        D_tr,
        subsets,
        knowledge,
        method,
        config,
        rng,
    )

    local_projections = project_local_graphs(local_graphs, subsets, distance_mode)

    raw_self_compat = self_compatibility_score_among_local_graphs(
        local_graphs,
        subsets,
        distance_mode=distance_mode,
    )

    # Compute scores on N + N_test disjoint, equal-sized chunks of D_cal.
    # The first N set the conformal threshold; the held-out remainder provide
    # an honest estimate of the exceedance rate P(A_test > tau_alpha).
    # Evaluating exceedance on the calibration scores themselves would make it
    # a deterministic function of (N, alpha), identical across base methods.
    N_cal = config.n_calibration_resamples
    N_test = config.n_test_chunks
    # In equal_chunks mode each calibration chunk is exactly len(D_tr) rows so
    # scores reflect deployment-size instability.  In the default mode chunk_size
    # is None and calibration_scores uses its internal heuristic (n // n_resamples).
    if config.cal_chunk_size is not None:
        cal_chunk_size = config.cal_chunk_size
    elif config.equal_chunks:
        cal_chunk_size = len(D_tr)
    else:
        cal_chunk_size = None
    all_chunk_scores = calibration_scores(
        D_cal,
        subsets,
        local_projections,
        knowledge,
        method,
        config,
        n_resamples=N_cal + N_test,
        rng=rng,
        distance_mode=distance_mode,
        chunk_size=cal_chunk_size,
        soft_knowledge=soft_knowledge,
        lambda_penalty=lam,
    )
    # With equal_chunks the number of returned scores may be < N_cal+N_test if
    # D_cal is too small to supply that many full chunks of size len(D_tr).
    # Allocate the first ~half to calibration and the rest to test so neither
    # slice is empty.  In the normal case (enough chunks) this is identical to
    # the fixed [:N_cal] / [N_cal:] slicing.
    n_returned = len(all_chunk_scores)
    if n_returned >= N_cal + N_test:
        # Normal path: enough chunks.
        cal_scores = all_chunk_scores[:N_cal]
        test_scores = all_chunk_scores[N_cal:]
    else:
        # Fewer chunks than requested: split roughly in half, ensuring both
        # slices have at least 1 element (n_returned >= 2 is needed; if only 1
        # chunk came back, put it in cal and test is empty — exceedance will be nan).
        n_cal_eff = max(1, n_returned // 2)
        cal_scores = all_chunk_scores[:n_cal_eff]
        test_scores = all_chunk_scores[n_cal_eff:]

    candidates = generate_candidate_graphs(
        D_tr,
        knowledge,
        method,
        config,
        config.n_candidate_graphs,
        rng,
    )

    candidate_base_scores = [
        candidate_graph_score(
            G, local_projections, subsets, distance_mode,
            knowledge=soft_knowledge, lambda_penalty=lam,
        )
        for G in candidates
    ]
    # Claims, baselines, and the decision task work on 0/1 adjacency.
    candidate_adjs = [to_adjacency(G) for G in candidates]
    # The first candidate is the base algorithm on the full D_train.
    full_graph = candidate_adjs[0]

    # Score of the data-generating graph.  It is marginalized onto each subset
    # like any candidate, so hidden variables are handled correctly.
    true_marks = from_dag_adjacency(true_full_dag) if true_full_dag is not None else None
    if true_marks is not None:
        true_graph_score = candidate_graph_score(
            true_marks, local_projections, subsets, distance_mode,
            knowledge=soft_knowledge, lambda_penalty=lam,
        )
    else:
        true_graph_score = float("nan")

    bootstrap = bootstrap_marks(D_tr, knowledge, method, config, rng, config.stability_n_bootstraps)
    baseline_cols = baseline_summary(
        full_graph=full_graph,
        data_train=D_tr,
        config=config,
        rng=rng,
        candidates=candidate_adjs,
        candidate_scores=candidate_base_scores,
        raw_self_compat=raw_self_compat,
        var_names=var_names,
        true_graph=true_graph,
        bootstrap_adjs=[to_adjacency(G) for G in bootstrap],
    )

    # Localization ablations: same candidate pool, no variable subsets.
    if config.run_ablations:
        boot_compat = bootstrap_compatibility(
            bootstrap, candidates, var_names, distance_mode, true_marks
        )
        no_subsets = conformal_without_subsets(
            D_cal, candidates[0], candidates, knowledge, method, config, rng,
            distance_mode, n_chunks=N_cal, chunk_size=cal_chunk_size, true_full_dag_marks=true_marks,
        )

    summary_rows = []
    claim_rows = []
    score_rows = []

    for s in cal_scores:
        score_rows.append({
            "dataset": dataset,
            "method": method,
            "repeat": repeat,
            "score_type": "calibration",
            "score": float(s),
        })

    for s in test_scores:
        score_rows.append({
            "dataset": dataset,
            "method": method,
            "repeat": repeat,
            "score_type": "test",
            "score": float(s),
        })

    for s in candidate_base_scores:
        score_rows.append({
            "dataset": dataset,
            "method": method,
            "repeat": repeat,
            "score_type": "candidate",
            "score": float(s),
        })

    for alpha in config.alpha_values:
        tau = conformal_threshold(cal_scores, alpha)
        selected = conformal_graph_set(candidate_adjs, candidate_base_scores, tau)
        candidate_scores = candidate_base_scores

        ablation_cols: Dict[str, float] = {}
        if config.run_ablations:
            tau_boot = bootstrap_threshold(boot_compat["reference"], alpha)
            ablation_cols.update(ablation_columns(
                "bootcompat",
                conformal_graph_set(candidate_adjs, boot_compat["candidate"], tau_boot),
                float(boot_compat["truth"] <= tau_boot) if true_marks is not None else float("nan"),
                var_names, true_graph, config,
            ))
            tau_v = conformal_threshold(no_subsets["calibration"], alpha)
            ablation_cols.update(ablation_columns(
                "nosubsets",
                conformal_graph_set(candidate_adjs, no_subsets["candidate"], tau_v),
                float(no_subsets["truth"] <= tau_v) if true_marks is not None else float("nan"),
                var_names, true_graph, config,
            ))

        claims = summarize_claims(
            selected,
            var_names,
            accepted_threshold=config.accepted_threshold,
            rejected_threshold=config.rejected_threshold,
        )

        counts = status_counts(claims)
        pr = accepted_precision_recall(claims, true_graph)

        # Downstream decision task: estimate treatment-effect bias under
        # four adjustment strategies.  Only available when sem_weights and
        # true_graph are both provided (fully observed synthetic setting).
        _decision_nan: Dict[str, float] = {
            "decision_bias_oracle": float("nan"),
            "decision_bias_single": float("nan"),
            "decision_bias_accepted": float("nan"),
            "decision_bias_notrejected": float("nan"),
            "decision_bias_cscd_avg": float("nan"),
            "decision_bias_naive": float("nan"),
            "decision_n_pairs": float("nan"),
        }
        if sem_weights is not None and true_graph is not None:
            _dec_rng = _make_rng(rng_seed + 7919 + int(alpha * 1000))
            try:
                decision_cols = decision_task_summary(
                    data_eval=data,
                    W=sem_weights,
                    var_names=var_names,
                    true_graph=true_graph,
                    full_graph=full_graph,
                    claims=claims,
                    rng=_dec_rng,
                    n_pairs=config.decision_n_pairs,
                    accepted_threshold=config.accepted_threshold,
                    rejected_threshold=config.rejected_threshold,
                    selected_graphs=selected,
                )
            except Exception as _dec_exc:
                logger.warning("decision_task_summary failed: %s", _dec_exc)
                decision_cols = _decision_nan
        else:
            decision_cols = _decision_nan

        if true_graph is not None:
            graph_shd = shd(full_graph, true_graph)
            sk_prec, sk_rec, sk_f1 = skeleton_f1(full_graph, true_graph)
        else:
            graph_shd = np.nan
            sk_prec, sk_rec, sk_f1 = np.nan, np.nan, np.nan

        # Calibration check (Proposition 1): fraction of *held-out* chunk
        # scores exceeding τ_α should be ≈ α.  tau=inf when k>N, in which case
        # all scores are ≤ τ and the empirical rate is 0.
        finite_tau = np.isfinite(tau)
        if len(test_scores) > 0:
            cal_exceed_rate = float(np.mean(test_scores > tau)) if finite_tau else 0.0
        else:
            cal_exceed_rate = float("nan")
        calibration_error = cal_exceed_rate - alpha
        insample_exceed_rate = float(np.mean(cal_scores > tau)) if finite_tau else 0.0

        total_claims = len(var_names) * (len(var_names) - 1)
        summary_rows.append({
            "dataset": dataset,
            "method": method,
            "repeat": repeat,
            "alpha": alpha,
            "tau": tau if finite_tau else float("nan"),
            "raw_self_compatibility": raw_self_compat,
            "n_selected_graphs": len(selected),
            "n_candidates": len(candidates),
            "n_subsets_used": len(subsets),
            "true_graph_score": true_graph_score,
            "true_graph_included": (
                float(true_graph_score <= tau) if np.isfinite(true_graph_score) else float("nan")
            ),
            "full_graph_score": candidate_base_scores[0],
            "candidate_accept_rate": float(np.mean(np.array(candidate_scores) <= tau)),
            "cal_exceed_rate": cal_exceed_rate,
            "insample_exceed_rate": insample_exceed_rate,
            "calibration_error": calibration_error,
            "full_graph_shd": graph_shd,
            "skeleton_precision": sk_prec,
            "skeleton_recall": sk_rec,
            "skeleton_f1": sk_f1,
            **counts,
            **pr,
            "accepted_rate": counts["n_accepted"] / total_claims,
            "rejected_rate": counts["n_rejected"] / total_claims,
            "unresolved_rate": counts["n_unresolved"] / total_claims,
            "knowledge_regime": config.knowledge_regime,
            "knowledge_mode": config.knowledge_mode,
            "lambda_penalty": config.lambda_penalty,
            **baseline_cols,
            **ablation_cols,
            **decision_cols,
        })

        claims = claims.copy()
        claims["dataset"] = dataset
        claims["method"] = method
        claims["repeat"] = repeat
        claims["alpha"] = alpha
        claim_rows.append(claims)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(claim_rows, ignore_index=True),
        pd.DataFrame(score_rows),
    )


def _execute_tasks(tasks: List[Dict], n_workers: Optional[int]) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    if n_workers is None or n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(_run_one_task, task): task
                for task in tasks
            }
            results = []
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.warning(
                        "Skipping task dataset=%s method=%s repeat=%s: %s",
                        task["dataset"],
                        task["method"],
                        task["repeat"],
                        exc,
                    )
            return results
    results = []
    for task in tasks:
        try:
            results.append(run_one(**task))
        except Exception as exc:
            logger.warning(
                "Skipping task dataset=%s method=%s repeat=%s: %s",
                task["dataset"],
                task["method"],
                task["repeat"],
                exc,
            )
    return results


def run_synthetic(
    methods: List[str],
    config: ExperimentConfig,
    outdir: str,
    distance_mode: str,
    n_workers: Optional[int] = None,
) -> None:
    tasks = []
    all_summary = []
    all_claims = []
    all_scores = []

    for repeat in range(config.n_repeats):
        rng = _make_rng(config.random_seed + repeat)
        B = generate_random_dag(
            config.p,
            config.expected_degree,
            rng,
        )

        data, W_full = simulate_linear_sem(
            B,
            n=config.n,
            noise=config.noise,
            rng=rng,
            return_weights=True,
        )
        data, full_dag = hide_latents(data, B, config.latent_fraction, rng)

        # sem_weights are only passed in the fully observed setting; when
        # latents are hidden the W indexing no longer matches the observed
        # variable order so we pass None to disable the decision task.
        sem_weights_for_task = W_full if config.latent_fraction == 0.0 else None

        var_names = list(data.columns)
        true_graph = true_observed_graph(full_dag, var_names)

        # Build regime knowledge.
        empty_knowledge = make_domain_knowledge(var_names)
        if config.knowledge_regime == "none":
            regime_knowledge = empty_knowledge
        elif config.knowledge_regime == "correct":
            regime_knowledge = knowledge_from_true_graph(
                true_graph, config.knowledge_frac, rng
            )
        else:  # noisy
            correct_knowledge = knowledge_from_true_graph(
                true_graph, config.knowledge_frac, rng
            )
            regime_knowledge = corrupt_knowledge(
                correct_knowledge, config.knowledge_noise, rng
            )

        # Dispatch per knowledge_mode.
        if config.knowledge_mode == "hard":
            task_knowledge = regime_knowledge
            task_soft_knowledge = None
        else:  # soft
            task_knowledge = empty_knowledge
            task_soft_knowledge = regime_knowledge

        for method in methods:
            tasks.append({
                "data": data,
                "knowledge": task_knowledge,
                "config": config,
                "method": method,
                "dataset": "synthetic",
                "repeat": repeat,
                "true_graph": true_graph,
                "distance_mode": distance_mode,
                "rng_seed": config.random_seed + repeat * 1000 + _stable_method_offset(method),
                "soft_knowledge": task_soft_knowledge,
                "sem_weights": sem_weights_for_task,
                "true_full_dag": full_dag,
            })

    for summary, claims, scores in _execute_tasks(tasks, n_workers):
        all_summary.append(summary)
        all_claims.append(claims)
        all_scores.append(scores)

    save_and_plot(all_summary, all_claims, all_scores, outdir)


def run_named_dataset(
    dataset: str,
    methods: List[str],
    config: ExperimentConfig,
    outdir: str,
    distance_mode: str,
    n_workers: Optional[int] = None,
    real_csv: Optional[str] = None,
) -> None:
    """Run CSCD on the Sachs data, a bnlearn network, or a CSV file.

    bnlearn networks (asia, sachs_bn, child, insurance, alarm) are resampled
    for every repeat.  The Sachs data and CSV files are fixed, so repeats only
    change the train/calibration split, subsets, and candidates.
    """
    dataset = dataset.lower()
    bnlearn_names = {"asia": "asia", "sachs_bn": "sachs", "child": "child",
                     "insurance": "insurance", "alarm": "alarm"}

    def load(repeat: int):
        if dataset == "sachs_real":
            return load_causallearn_sachs()
        if dataset in bnlearn_names:
            return load_bnlearn_pgmpy_dataset(
                bnlearn_names[dataset], n_samples=config.n, random_seed=config.random_seed + repeat
            )
        if dataset == "csv":
            if real_csv is None:
                raise ValueError("--real_csv is required when --dataset csv.")
            data = load_real_csv(real_csv)
            return data, sachs_reference_graph(list(data.columns))
        raise ValueError(f"Unknown dataset {dataset!r}.")

    tasks = []
    for repeat in range(config.n_repeats):
        data, ref_graph = load(repeat)
        knowledge = make_domain_knowledge(list(data.columns))
        for method in methods:
            tasks.append({
                "data": data,
                "knowledge": knowledge,
                "config": config,
                "method": method,
                "dataset": dataset,
                "repeat": repeat,
                "true_graph": ref_graph,
                "distance_mode": distance_mode,
                "rng_seed": config.random_seed + repeat * 1000 + _stable_method_offset(method),
                "true_full_dag": ref_graph,
            })

    all_summary = []
    all_claims = []
    all_scores = []

    for summary, claims, scores in _execute_tasks(tasks, n_workers):
        all_summary.append(summary)
        all_claims.append(claims)
        all_scores.append(scores)

    save_and_plot(all_summary, all_claims, all_scores, outdir)
