"""Sachs et al. (2005) interventional conditions as held-out validation.

The 7466-cell data distributed with causal-learn is the concatenation of
the nine conditions of Sachs et al. in their original order, with sizes
853, 902, 911, 723, 810, 799, 848, 913, 707 (the first block is identical to
the observational CD3/CD28 file distributed by bnlearn; the per-block mean
shifts match the targets of the INT column of bnlearn's discretized
interventional file).  Six conditions act on a measured protein:

    AKT inhibitor -> Akt, G06976 -> PKC, psitectorigenin -> PIP2,
    U0126 -> Mek, PMA -> PKC, beta2cAMP -> PKA.

The inhibitors act on the activity of their target, not necessarily on its
measured phosphorylation, so we only test *other* proteins.  A pair
(target x, protein y) counts as an interventional effect if y's distribution
shifts between the observational condition and the condition acting on x:
two-sample Kolmogorov-Smirnov p-value below a Bonferroni level, and an
absolute standardized mean difference above ``min_effect``.  CSCD and the
baselines are fitted on the observational condition only, so every
interventional sample is held out.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

CONDITIONS = ["cd3cd28", "icam2", "aktinhib", "g0076", "psitect", "u0126", "ly294002", "pma", "b2camp"]
SIZES = [853, 902, 911, 723, 810, 799, 848, 913, 707]
TARGETS = {"aktinhib": "Akt", "g0076": "PKC", "psitect": "PIP2", "u0126": "Mek", "pma": "PKC", "b2camp": "PKA"}
_NAMES = {"raf": "Raf", "mek": "Mek", "plc": "Plcg", "pip2": "PIP2", "pip3": "PIP3", "erk": "Erk",
          "akt": "Akt", "pka": "PKA", "pkc": "PKC", "p38": "P38", "jnk": "Jnk"}


def load_conditions() -> Dict[str, pd.DataFrame]:
    """Log-transformed data of every condition, with the observational file's column names."""
    from causallearn.utils.Dataset import load_dataset

    X, labels = load_dataset("sachs")
    cols = [_NAMES[l.lower()] for l in labels]
    df = pd.DataFrame(np.log(np.clip(X, 1e-3, None)), columns=cols)
    bounds = np.cumsum([0] + SIZES)
    assert bounds[-1] == len(df), "unexpected size of the pooled Sachs data"
    return {c: df.iloc[bounds[i]:bounds[i + 1]].reset_index(drop=True) for i, c in enumerate(CONDITIONS)}


def interventional_effects(min_effect: float = 0.5, level: float = 0.05) -> pd.DataFrame:
    """One row per (condition, target, protein): KS p-value, standardized mean shift, effect flag."""
    cond = load_conditions()
    obs = cond["cd3cd28"]
    rows = []
    n_tests = sum(len(obs.columns) - 1 for _ in TARGETS)
    for c, x in TARGETS.items():
        for y in obs.columns:
            if y == x:
                continue
            p = ks_2samp(obs[y], cond[c][y]).pvalue
            smd = (cond[c][y].mean() - obs[y].mean()) / obs[y].std()
            rows.append(dict(condition=c, target=x, protein=y, ks_p=p, smd=smd,
                             effect=bool(p < level / n_tests and abs(smd) >= min_effect)))
    return pd.DataFrame(rows)


def validate_claims(freq: np.ndarray, names: List[str], effects: pd.DataFrame, gamma: float = 0.9) -> Dict[str, float]:
    """Held-out check of ancestral claims x in An(y) with x an intervention target.

    ``freq[i, j]`` is the frequency (or posterior) of the claim names[i] is an
    ancestor of names[j].  Accepted claims (freq >= gamma) are validated if y
    shifts under an intervention on x; accepted non-ancestral claims
    (freq <= 1 - gamma) are contradicted if it does.
    """
    idx = {v: i for i, v in enumerate(names)}
    acc = val = rej = contra = 0
    for _, r in effects.iterrows():
        f = freq[idx[r.target], idx[r.protein]]
        if f >= gamma - 1e-12:
            acc += 1
            val += int(r.effect)
        elif f <= 1 - gamma + 1e-12:
            rej += 1
            contra += int(r.effect)
    n_eff = int(effects.effect.sum())
    return {"n_accepted": acc, "validated": val / acc if acc else np.nan, "recall": val / max(n_eff, 1),
            "n_rejected": rej, "contradicted": contra / rej if rej else np.nan, "n_effects": n_eff,
            "n_pairs": len(effects)}
