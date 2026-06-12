"""
This script runs experiments for the Conformal Self-Compatible Causal Discovery (CSCD) method on both synthetic and real datasets. It allows for extensive configuration of the experimental setup, including parameters for data generation, method selection, and evaluation metrics. The results are logged to a specified output directory for further analysis.
"""

import logging
import os

from cscd.config import ExperimentConfig, parse_args
from cscd.runner import run_named_dataset, run_synthetic


def configure_logging(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )
    logging.captureWarnings(True)


def main() -> None:
    args = parse_args()
    configure_logging(args.outdir)

    config = ExperimentConfig(
        n=args.n,
        p=args.p,
        expected_degree=args.expected_degree,
        noise=args.noise,
        latent_fraction=args.latent_fraction,
        n_repeats=args.n_repeats,
        n_subsets=args.n_subsets,
        subset_frac=args.subset_frac,
        n_calibration_resamples=args.n_calibration_resamples,
        n_test_chunks=args.n_test_chunks,
        n_candidate_graphs=args.n_candidate_graphs,
        pc_alpha=args.pc_alpha,
        fci_alpha=args.fci_alpha,
        independence_test=args.independence_test,
        ges_score=args.ges_score,
        accepted_threshold=args.accepted_threshold,
        rejected_threshold=args.rejected_threshold,
        stability_n_bootstraps=args.stability_n_bootstraps,
        stability_threshold=args.stability_threshold,
        random_seed=args.random_seed,
        knowledge_regime=args.knowledge_regime,
        knowledge_mode=args.knowledge_mode,
        knowledge_frac=args.knowledge_frac,
        knowledge_noise=args.knowledge_noise,
        lambda_penalty=args.lambda_penalty,
        decision_n_pairs=args.decision_n_pairs,
    )

    methods = [m.lower() for m in args.methods]

    if args.dataset == "synthetic":
        run_synthetic(
            methods=methods,
            config=config,
            outdir=args.outdir,
            distance_mode=args.distance_mode,
            n_workers=args.n_workers,
        )
    else:
        run_named_dataset(
            dataset=args.dataset,
            methods=methods,
            config=config,
            outdir=args.outdir,
            distance_mode=args.distance_mode,
            n_workers=args.n_workers,
            real_csv=args.real_csv,
        )


if __name__ == "__main__":
    main()
