#!/bin/bash
# Rebuild report.md and the paper's tables/figures from results_v2/summary.
cd "$(dirname "$0")"
export PYTHONPATH=.
PY=.venv/bin/python
PAPER=../6ab804def2ecfc179daef8f7
$PY make_paper_tables_v2.py --out results_v2/tables > /dev/null 2>&1
$PY make_figures_v2.py --out results_v2/figures > /dev/null 2>&1
if [ -d "$PAPER" ]; then
  $PY make_paper_tables_v2.py --out "$PAPER/tables/v2" > /dev/null 2>&1
  $PY make_figures_v2.py --out "$PAPER/figures/results_v2" > /dev/null 2>&1
fi
{
  cat report_header.md report_body.md report_tail.md
  echo
  echo "## Appendix A: generated tables"
  echo
  echo "*Built by \`make_report_tables.py\` on $(date '+%Y-%m-%d %H:%M'). Mean ± standard error over repeats; α = 0.1, γ = 0.9 unless stated.*"
  $PY make_report_tables.py 2>/dev/null
  echo
  echo "### E4 — Threshold-free detection (AUROC against the matched control)"
  echo
  cat results_v2/summary/e4_auroc.md
  echo
  echo "### E11 — Paired decision-task differences"
  echo
  cat results_v2/summary/e11_paired.md
} > report.md
echo "report.md: $(wc -l < report.md) lines"
