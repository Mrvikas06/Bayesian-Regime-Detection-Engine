"""
Phase 3 Jupyter notebook generator for Frequentist HMM diagnostics & regime overlay.
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
                "# Phase 3 — Frequentist 5-State Gaussian HMM\n",
                "BIC comparison (3/5/7 states), transition matrix, expected durations, and Nifty 50 overlay.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd, json\n",
                "bic = pd.read_csv('../reports/tables/phase3_hmm_bic.csv')\n",
                "display(bic)\n",
                "trans = pd.read_csv('../reports/tables/phase3_hmm_transitions.csv', index_col=0)\n",
                "display(trans)",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Image, display\n",
                "display(Image('../reports/figures/phase3_hmm_overlay.png'))",
            ],
        },
    ],
}


def write_notebook(out_path: str = "notebooks/phase3_hmm_freq.ipynb") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(NOTEBOOK, f, indent=1)
    print(f"Notebook written: {out_path}")


if __name__ == "__main__":
    write_notebook()
