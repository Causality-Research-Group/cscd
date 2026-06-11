import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DomainKnowledge:
    order: Dict[str, int]
    forbidden_edges: Set[Tuple[str, str]]
    required_edges: Set[Tuple[str, str]]


@dataclass
class ExperimentConfig:
    n: int = 1000
    p: int = 20
    expected_degree: float = 2.0
    noise: str = "gaussian"
    latent_fraction: float = 0.0

    n_repeats: int = 5
    n_subsets: int = 30
    subset_frac: float = 0.5
    n_calibration_resamples: int = 20   # N disjoint cal chunks; paper uses 20
    n_candidate_graphs: int = 80

    alpha_values: Tuple[float, ...] = (0.05, 0.10, 0.20)
    accepted_threshold: float = 0.90
    rejected_threshold: float = 0.10

    pc_alpha: float = 0.05
    fci_alpha: float = 0.05
    independence_test: str = "fisherz"
    ges_score: str = "local_score_BIC"

    random_seed: int = 123


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["synthetic", "sachs_real", "asia", "sachs_bn", "child", "csv"],
        required=True,
    )
    parser.add_argument("--real_csv", type=str, default=None)
    parser.add_argument("--methods", nargs="+", default=["pc", "ges", "fci"])
    parser.add_argument("--outdir", type=str, default="results_cscd")
    parser.add_argument("--n_workers", type=int, default=None)

    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--expected_degree", type=float, default=2.0)
    parser.add_argument(
        "--noise",
        choices=["gaussian", "uniform", "laplace"],
        default="gaussian",
    )
    parser.add_argument("--latent_fraction", type=float, default=0.0)

    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--n_subsets", type=int, default=30)
    parser.add_argument("--subset_frac", type=float, default=0.5)
    parser.add_argument("--n_calibration_resamples", type=int, default=20)
    parser.add_argument("--n_candidate_graphs", type=int, default=80)

    parser.add_argument("--pc_alpha", type=float, default=0.05)
    parser.add_argument("--fci_alpha", type=float, default=0.05)
    parser.add_argument("--independence_test", type=str, default="fisherz")
    parser.add_argument("--ges_score", type=str, default="local_score_BIC")

    parser.add_argument("--accepted_threshold", type=float, default=0.90)
    parser.add_argument("--rejected_threshold", type=float, default=0.10)
    parser.add_argument(
        "--distance_mode", choices=["directed", "skeleton"], default="directed"
    )
    parser.add_argument("--random_seed", type=int, default=123)

    return parser.parse_args()
