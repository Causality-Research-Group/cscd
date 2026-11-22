"""LaTeX tables of the discussion-phase experiments, for the revised paper.

Usage:  python -m rebuttal.wave2_tables

Reads the CSV files written by ``rebuttal.wave2_summaries`` and
``rebuttal.nosub_baseline`` and writes, in rebuttal/out/:

* split.tex      training-fraction sweep (Kmr8-Q1)
* latent_fci.tex PC and FCI under hidden hubs as n grows (d2sj-W4)
* knowledge_local.tex  knowledge given to the full runs only (Rfqt-Q1)
* nosub.tex      corrected conformal baseline without subsets (AI-3.1)
"""

import pandas as pd

OUT = "rebuttal/out"
METHOD = {"pc": "PC", "ges": "GES", "fci": "FCI"}
SCENARIO = {"p=50, n=1000": "$p=50$, $n=1000$"}


def num(x, digits=2):
    return f"${x:.{digits}f}$"


def ci(text):
    """'[0.58, 0.92]' -> '[$0.58$, $0.92$]'."""
    lo, hi = text.strip("[]").split(",")
    return f"[${float(lo):.2f}$, ${float(hi):.2f}$]"


def write(name, header, rows, colspec):
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    lines += [r if r == "\\midrule" else r + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(f"{OUT}/{name}.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def split_table():
    df = pd.read_csv(f"{OUT}/wave2_split.csv")
    rows = []
    for method in ("pc", "ges", "fci"):
        first = True
        for _, r in df[df.method == method].sort_values("train_frac").iterrows():
            rows.append(" & ".join([
                METHOD[method] if first else "",
                f"${r.train_frac:.1f}$", f"${int(r.chunk_size)}$",
                num(r.exceed), num(r.tau, 3), num(r.gstar) + " " + ci(r.gstar_ci),
                num(r.admitted), num(r.reject_ghat), num(r.adj_prec, 3), num(r.adj_rec, 3),
                num(r.dir_def_prec, 3),
            ]))
            first = False
        if method != "fci":
            rows.append("\\midrule")
    header = ("Method & Train frac. & $n_c$ & Exceed. & $\\tau_\\alpha$ & $G^\\ast$ incl.\\ [$95\\%$] "
              "& Admitted & Rejects $\\widehat G$ & Adj.\\ prec. & Adj.\\ rec. & Def.\\ dir.\\ prec.")
    write("split", header, rows, "l" + "c" * 10)


def latent_table():
    df = pd.read_csv(f"{OUT}/wave2_latent.csv")
    rows = []
    for method in ("pc", "fci"):
        first = True
        for _, r in df[df.method == method].sort_values("n").iterrows():
            rows.append(" & ".join([
                METHOD[method] if first else "",
                f"${int(r.n):,}$".replace(",", "{,}"), f"${int(r.chunk_size)}$",
                num(r.exceed), num(r.tau, 3), num(r.gstar) + " " + ci(r.gstar_ci),
                num(r.pair_err_train, 3), num(r.any_err_train), num(r.pair_err_chunk, 3),
                num(r.frac_sufficient, 3),
            ]))
            first = False
        if method != "fci":
            rows.append("\\midrule")
    header = ("Method & $n$ & $n_c$ & Exceed. & $\\tau_\\alpha$ & $G^\\ast$ incl.\\ [$95\\%$] "
              "& Pair err.\\ (train) & Subsets with an error & Pair err.\\ (chunk) & Sufficient subsets")
    write("latent_fci", header, rows, "l" + "c" * 9)


def knowledge_table():
    df = pd.read_csv(f"{OUT}/wave2_knowledge.csv")
    rows = []
    for method in ("pc", "fci"):
        first = True
        for _, r in df[df.method == method].iterrows():
            knowledge, where = r.knowledge.split(", ", 1)
            rows.append(" & ".join([
                METHOD[method] if first else "", knowledge.capitalize().replace("%", "\\%"), where.capitalize(),
                num(r.exceed), num(r.gstar), num(r.admitted, 3), num(r.reject_ghat),
                num(r.adj_prec, 3), num(r.dir_def_prec, 3), num(r.dir_def_rec, 3),
            ]))
            first = False
        if method != "fci":
            rows.append("\\midrule")
    header = ("Method & Knowledge & Given to & Exceed. & $G^\\ast$ incl. & Admitted & Rejects $\\widehat G$ "
              "& Adj.\\ prec. & Def.\\ dir.\\ prec. & Def.\\ dir.\\ rec.")
    write("knowledge_local", header, rows, "lllccccccc")


def nosub_table():
    df = pd.read_csv(f"{OUT}/nosub_detection.csv")
    rows = []
    for scenario, group in df.groupby("scenario", sort=False):
        first = True
        for _, r in group.iterrows():
            rows.append(" & ".join([
                SCENARIO.get(scenario, scenario) if first else "", METHOD[r.method],
                num(r.flag_scenario), num(r.flag_control), num(r.auroc),
                num(r.score_median, 3), num(r.tau_median, 3),
            ]))
            first = False
    header = ("Scenario & Method & Flagged (scenario) & Flagged (control) & AUROC "
              "& Median score & Median $\\tau_\\alpha$")
    write("nosub", header, rows, "llccccc")


if __name__ == "__main__":
    split_table()
    latent_table()
    knowledge_table()
    nosub_table()
    print("wrote split, latent_fci, knowledge_local, nosub to", OUT)
