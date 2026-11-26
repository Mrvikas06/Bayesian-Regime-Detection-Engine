"""
Phase 1 Jupyter notebook for data quality visualisation.
This script generates a phase-1 notebook that can be run interactively.
Execute `jupyter nbconvert --to notebook --execute notebooks/phase1_data_quality.ipynb`
after `make phase-1` to view charts.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "regime",
            "language": "python",
            "name": "regime",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 1 — Data Quality Notebook\n",
                "Run after `make phase-1`. Shows raw series, missing-value heatmap,\n",
                "and reference label overlay on Nifty 50.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import matplotlib.dates as mdates\n",
                "import seaborn as sns\n",
                "import yaml, json\n",
                "from pathlib import Path\n",
                "import sys; sys.path.insert(0, str(Path('..').resolve()))\n",
                "\n",
                "with open('../configs/data.yaml') as f:\n",
                "    cfg = yaml.safe_load(f)\n",
                "\n",
                "df = pd.read_parquet(f\"../{cfg['output']['master_parquet']}\")\n",
                "print(f'Loaded: {df.shape[0]} rows × {df.shape[1]} cols')\n",
                "print(f'Date range: {df.index.min().date()} → {df.index.max().date()}')\n",
                "df.head(3)",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Missing value heatmap ---\n",
                "plt.figure(figsize=(14, 5))\n",
                "miss = df.isna().astype(int)\n",
                "# Downsample to monthly for readability\n",
                "miss_m = miss.resample('ME').mean()\n",
                "sns.heatmap(miss_m.T, cmap='Reds', cbar_kws={'label': 'Frac missing'},\n",
                "            xticklabels=24, yticklabels=True)\n",
                "plt.title('Missing Data Fraction (monthly aggregation)')\n",
                "plt.tight_layout()\n",
                "Path('../reports/figures').mkdir(parents=True, exist_ok=True)\n",
                "plt.savefig('../reports/figures/phase1_missing_heatmap.png', dpi=150)\n",
                "plt.show()",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Nifty 50 price with reference regime overlay ---\n",
                "from regime.data.labels import label_regimes\n",
                "labels = label_regimes(df)\n",
                "\n",
                "COLORS = {\n",
                "    'Risk-On': '#2ecc71',\n",
                "    'Risk-Off': '#e74c3c',\n",
                "    'Transitional': '#f39c12',\n",
                "    'Late-Cycle': '#9b59b6',\n",
                "    'Post-Shock': '#1abc9c',\n",
                "}\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(16, 5))\n",
                "if 'nifty50' in df.columns:\n",
                "    ax.plot(df['nifty50'], color='black', linewidth=0.8, label='Nifty 50')\n",
                "    # Shade background by regime\n",
                "    prev_regime = None\n",
                "    start_dt = None\n",
                "    for dt, reg in labels.items():\n",
                "        if reg != prev_regime:\n",
                "            if prev_regime is not None:\n",
                "                ax.axvspan(start_dt, dt, alpha=0.25, color=COLORS.get(prev_regime, 'grey'))\n",
                "            start_dt = dt\n",
                "            prev_regime = reg\n",
                "    if start_dt:\n",
                "        ax.axvspan(start_dt, labels.index[-1], alpha=0.25, color=COLORS.get(prev_regime, 'grey'))\n",
                "\n",
                "from matplotlib.patches import Patch\n",
                "legend_patches = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in COLORS.items()]\n",
                "ax.legend(handles=legend_patches, loc='upper left', fontsize=8)\n",
                "ax.set_title('Nifty 50 with Reference Regime Labels (Weak Labels — Not Ground Truth)')\n",
                "ax.set_ylabel('Index Level')\n",
                "ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))\n",
                "plt.tight_layout()\n",
                "plt.savefig('../reports/figures/phase1_nifty_regime_overlay.png', dpi=150)\n",
                "plt.show()",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Regime label distribution ---\n",
                "dist = labels.value_counts(normalize=True).rename('fraction')\n",
                "print('Reference label distribution:')\n",
                "print(dist.to_string())\n",
                "\n",
                "dist.plot(kind='bar', color=[COLORS.get(r,'grey') for r in dist.index],\n",
                "          figsize=(8,4), title='Reference Label Distribution')\n",
                "plt.ylabel('Fraction of trading days')\n",
                "plt.xticks(rotation=30)\n",
                "plt.tight_layout()\n",
                "plt.savefig('../reports/figures/phase1_label_distribution.png', dpi=150)\n",
                "plt.show()",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- Print quality summary ---\n",
                "summary_path = Path('../data/lineage/quality_summary.json')\n",
                "if summary_path.exists():\n",
                "    with open(summary_path) as f:\n",
                "        s = json.load(f)\n",
                "    print(json.dumps(s, indent=2))\n",
                "else:\n",
                "    print('Quality summary not found — run make phase-1 quality step')",
            ],
        },
    ],
}


def write_notebook(out_path: str = "notebooks/phase1_data_quality.ipynb") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(NOTEBOOK, f, indent=1)
    print(f"Notebook written: {out_path}")


if __name__ == "__main__":
    write_notebook()
