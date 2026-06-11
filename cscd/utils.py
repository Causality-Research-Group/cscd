import os
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DomainKnowledge


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    arr = StandardScaler().fit_transform(df.values)
    return pd.DataFrame(arr, columns=df.columns, index=df.index)


def make_domain_knowledge(
    var_names: List[str],
    order: Optional[List[str]] = None,
) -> "DomainKnowledge":
    # order=None → no temporal ordering constraint (empty dict).
    # Passing order=var_names would impose Xi→Xj only when i<j by variable
    # index, which is wrong for synthetic experiments where the true causal
    # order is a random permutation of the variable indices.
    order_dict = {v: i for i, v in enumerate(order)} if order is not None else {}
    return DomainKnowledge(
        order=order_dict,
        forbidden_edges=set(),
        required_edges=set(),
    )


def restrict_knowledge(
    knowledge: "DomainKnowledge",
    subset: List[str],
) -> "DomainKnowledge":
    subset_set = set(subset)
    return DomainKnowledge(
        order={v: knowledge.order[v] for v in subset if v in knowledge.order},
        forbidden_edges={(a, b) for (a, b) in knowledge.forbidden_edges
                         if a in subset_set and b in subset_set},
        required_edges={(a, b) for (a, b) in knowledge.required_edges
                        if a in subset_set and b in subset_set},
    )


def apply_hard_domain_constraints(
    A: pd.DataFrame,
    knowledge: "DomainKnowledge",
) -> pd.DataFrame:
    A = A.copy()
    vars_ = list(A.index)

    for parent in vars_:
        for child in vars_:
            if parent == child:
                A.loc[parent, child] = 0
                continue

            violates_order = (
                parent in knowledge.order
                and child in knowledge.order
                and knowledge.order[parent] >= knowledge.order[child]
            )
            forbidden = (parent, child) in knowledge.forbidden_edges

            if violates_order or forbidden:
                A.loc[parent, child] = 0

    for parent, child in knowledge.required_edges:
        if parent in A.index and child in A.columns:
            A.loc[parent, child] = 1

    return A


def empty_adj(var_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame(0, index=var_names, columns=var_names, dtype=int)


def numpy_to_adj(B: "np.ndarray", var_names: List[str]) -> pd.DataFrame:
    import numpy as np

    return pd.DataFrame(np.asarray(B, dtype=int), index=var_names, columns=var_names)
