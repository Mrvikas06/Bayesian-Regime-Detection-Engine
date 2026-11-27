"""
Phase 4 Jupyter notebook generator for Bayesian HMM diagnostics & Phase 3 comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "regime", "language": "python", "name": "regime"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 4 — Bayesian 5-State HMM (PyMC)\n",
                "MCMC diagnostics (R-hat, ESS, divergences), trace plots, transition CIs, and Phase 3 comparison.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd, json\n",
                "with open('../data/lineage/phase4_lineage.json') as f:\n",
                "    lin = json.load(f)\n",
                "print('MCMC Diagnostics:', json.dumps(lin['mcmc_diagnostics'], indent=2))\n",
                "print('OOS Metrics:', json.dumps(lin['oos_metrics'], indent=2))\n",
                "display(pd.read_csv('../reports/tables/phase4_transitions_ci.csv', index_col=0))",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Image, display\n",
                "display(Image('../reports/figures/phase4_trace_plots.png'))\n",
                "display(Image('../reports/figures/phase4_vs_phase3_comparison.png'))",
            ],
        },
    ],
}


def write_notebook(out_path: str = "notebooks/phase4_hmm_bayes.ipynb") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(NOTEBOOK, f, indent=1)
    print(f"Notebook written: {out_path}")


if __name__ == "__main__":
    write_notebook()
