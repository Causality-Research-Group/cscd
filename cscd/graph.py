"""Wrappers around the causal-learn discovery algorithms.

Every algorithm returns an endpoint-mark graph (see ``mixed_graph``):
PC and GES return CPDAGs, FCI returns a PAG, and DirectLiNGAM returns a DAG.
"""

import contextlib
import io
from typing import Optional

import numpy as np
import pandas as pd

from .mixed_graph import from_causallearn, from_dag_adjacency, to_adjacency
from .utils import apply_hard_domain_constraints

try:
    from causallearn.search.ConstraintBased.PC import pc
except Exception:
    pc = None

try:
    from causallearn.search.ConstraintBased.FCI import fci
except Exception:
    fci = None

try:
    from causallearn.search.ScoreBased.GES import ges
except Exception:
    ges = None

try:
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
except Exception:
    DirectLiNGAM = None

SUPPORTED_METHODS = ("pc", "ges", "fci", "lingam")


def _run_pc(X: np.ndarray, var_names, config, alpha: float) -> pd.DataFrame:
    if pc is None:
        raise ImportError("Could not import causal-learn PC. Install causal-learn.")
    cg = pc(
        X,
        alpha=alpha,
        indep_test=config.independence_test,
        stable=True,
        node_names=var_names,
        verbose=False,
        show_progress=False,
    )
    return from_causallearn(cg.G, var_names)


def _run_fci(X: np.ndarray, var_names, config, alpha: float) -> pd.DataFrame:
    if fci is None:
        raise ImportError("Could not import causal-learn FCI. Install causal-learn.")
    # causal-learn's FCI prints every orientation step even with verbose=False.
    with contextlib.redirect_stdout(io.StringIO()):
        g, _ = fci(
            X,
            independence_test_method=config.independence_test,
            alpha=alpha,
            verbose=False,
            node_names=var_names,
            show_progress=False,
        )
    return from_causallearn(g, var_names)


def _run_ges(X: np.ndarray, var_names, config, max_parents: Optional[float]) -> pd.DataFrame:
    # causal-learn 0.1.3.x hard-codes the BIC penalty, so the tunable GES
    # hyperparameter is the maximum number of parents (None = p / 2).
    if ges is None:
        raise ImportError("Could not import causal-learn GES. Install causal-learn.")
    record = ges(X, score_func=config.ges_score, maxP=max_parents)
    return from_causallearn(record["G"], var_names)


def _run_lingam(X: np.ndarray, var_names, random_state: int) -> pd.DataFrame:
    if DirectLiNGAM is None:
        raise ImportError("Could not import causal-learn DirectLiNGAM. Install causal-learn.")
    model = DirectLiNGAM(random_state=random_state)
    model.fit(X)
    # adjacency_matrix_[i, j] is the coefficient of j in the equation of i.
    B = (np.abs(model.adjacency_matrix_) > 0).astype(int).T
    adj = pd.DataFrame(B, index=var_names, columns=var_names)
    return from_dag_adjacency(adj)


def default_hyperparameter(method: str, config) -> Optional[float]:
    """Significance level for PC/FCI, maximum parents for GES (None = p/2)."""
    method = method.lower()
    if method == "pc":
        return config.pc_alpha
    if method == "fci":
        return config.fci_alpha
    return None


def discover_marks(
    data: pd.DataFrame,
    method: str,
    knowledge,
    config,
    random_state: int = 0,
    hyperparameter: Optional[float] = None,
) -> pd.DataFrame:
    """Run a base discovery algorithm and apply hard domain knowledge.

    ``hyperparameter`` overrides the configured significance level (PC, FCI)
    or the maximum number of parents (GES).  It is used to build the candidate family.
    """
    method = method.lower()
    X = data.values
    var_names = list(data.columns)
    if hyperparameter is None:
        hyperparameter = default_hyperparameter(method, config)

    if method == "pc":
        marks = _run_pc(X, var_names, config, alpha=hyperparameter)
    elif method == "fci":
        marks = _run_fci(X, var_names, config, alpha=hyperparameter)
    elif method == "ges":
        marks = _run_ges(X, var_names, config, max_parents=hyperparameter)
    elif method == "lingam":
        marks = _run_lingam(X, var_names, random_state)
    else:
        raise ValueError(f"Unknown method {method!r}; expected one of {SUPPORTED_METHODS}.")

    return apply_hard_domain_constraints(marks, knowledge)


def discover_graph(
    data: pd.DataFrame,
    method: str,
    knowledge,
    config,
    random_state: int = 0,
) -> pd.DataFrame:
    """Run discovery and flatten the result to a 0/1 adjacency matrix.

    Used by code that reasons about parent claims (claims, baselines, decision
    task).  Self-compatibility scoring uses ``discover_marks`` instead, because
    marginalization needs the endpoint marks.
    """
    return to_adjacency(discover_marks(data, method, knowledge, config, random_state))
