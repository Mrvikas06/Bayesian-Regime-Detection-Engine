"""
Phase 2 Jupyter notebook generator for feature inspection & breakpoint plots.
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
                "# Phase 2 — Feature Engineering & No-Leakage Audit\n",
                "Inspects the 40+ causal features, correlation heatmap, and historical breakpoints.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd, json\n",
                "from pathlib import Path\n",
                "df = pd.read_parquet('../data/processed/features.parquet')\n",
                "feat_cols = [c for c in df.columns if c != 'regime_label' and not c.startswith('lineage_')]\n",
                "print(f'Features: {len(feat_cols)} | Rows: {len(df)}')\n",
                "df[feat_cols].describe().T[['count', 'mean', 'std', 'min', 'max']].head(20)",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Image, display\n",
                "display(Image('../reports/figures/phase2_features_breakpoints.png'))",
            ],
        },
    ],
}


def write_notebook(out_path: str = "notebooks/phase2_features.ipynb") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(NOTEBOOK, f, indent=1)
    print(f"Notebook written: {out_path}")


if __name__ == "__main__":
    write_notebook()
