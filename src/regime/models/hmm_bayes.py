"""
regime.models.hmm_bayes
=======================
Phase 4: Bayesian 5-State Gaussian HMM in PyMC.

Key design:
- Marginalizes discrete states via an exact Numba-accelerated forward-backward
  PyTensor Op (`HMMForwardOp` + analytical `HMMGradOp`) so NUTS samples smoothly.
- Sticky Dirichlet priors on transition matrix rows.
- Ordered constraint on VIX emission means (mu_vix_0 < ... < mu_vix_4) to
  eliminate label switching across chains.
- Enforces Rule 4 MCMC diagnostics gate: max R-hat <= 1.01 and 0 divergences.
- Computes 90% credible intervals on transitions, expected durations, and
  per-day causal regime probabilities.
- Evaluates strictly causal OOS calibration (2019+) and compares vs Phase 3.

Usage
-----
    python -m regime.models.hmm_bayes --config configs/models.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.preprocessing import StandardScaler

from regime.data.labels import REGIME_NAMES
from regime.data.versioning import (
    attach_lineage,
    build_lineage_record,
    hash_config,
    hash_dataframe,
    save_lineage_json,
)
from regime.models._hmm_op import (
    HMMForwardOp,
    HMMGradOp,
    _hmm_backward_grad_nb,
    _hmm_forward_nb,
    hmm_forward_op,
)
from regime.models.hmm_freq import (
    compute_calibration_metrics,
    load_config,
    map_states_to_regimes,
    reorder_probs_by_regime,
    resolve_feature_columns,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


# Op definitions & Numba dispatch live in regime.models._hmm_op (imported above)


# ---------------------------------------------------------------------------
# PyMC Model & Sampling
# ---------------------------------------------------------------------------

def build_and_sample_bayesian_hmm(
    X_train: np.ndarray,
    feature_names: list[str],
    n_states: int = 5,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    seed: int = 42,
) -> az.InferenceData:
    """
    Build and sample a 5-state Bayesian Gaussian HMM with:
    - Sticky Dirichlet transition priors
    - Ordered VIX emission means to prevent label switching
    - Exact marginalized log-likelihood via `hmm_forward_op`
    """
    from hmmlearn.hmm import GaussianHMM

    n_obs, n_feat = X_train.shape
    vix_idx = next((i for i, c in enumerate(feature_names) if "vix" in c.lower()), 0)
    other_idxs = [i for i in range(n_feat) if i != vix_idx]

    # Warm-start from MLE diagonal Gaussian HMM ordered by VIX mean
    mle = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=150, random_state=seed).fit(X_train)
    order = np.argsort(mle.means_[:, vix_idx])
    means_init = mle.means_[order]
    # Ensure strictly increasing VIX initvals
    for k in range(1, n_states):
        if means_init[k, vix_idx] <= means_init[k - 1, vix_idx] + 0.05:
            means_init[k, vix_idx] = means_init[k - 1, vix_idx] + 0.05
    vars_diag = np.diagonal(mle.covars_, axis1=1, axis2=2)[order]
    sig_init = np.maximum(np.sqrt(vars_diag) - 0.10, 0.05)
    P_init = np.clip(mle.transmat_[np.ix_(order, order)], 0.005, 0.99)
    P_init /= P_init.sum(axis=1, keepdims=True)

    # Sticky Dirichlet concentration: 20 on diagonal, 0.5 off-diagonal
    alpha_P = np.full((n_states, n_states), 0.5)
    np.fill_diagonal(alpha_P, 20.0)

    with pm.Model() as model:
        pi0 = pm.Dirichlet("pi0", a=np.ones(n_states), initval=np.ones(n_states) / n_states)
        P = pm.Dirichlet("P", a=alpha_P, shape=(n_states, n_states), initval=P_init)

        # Ordered emission mean on VIX dimension eliminates label-switching
        mu_vix = pm.Normal(
            "mu_vix",
            mu=means_init[:, vix_idx],
            sigma=0.25,
            shape=n_states,
            transform=pm.distributions.transforms.ordered,
            initval=means_init[:, vix_idx],
        )
        # Weakly informative priors centered on MLE centroids break multimodal
        # symmetry between states with similar VIX (e.g. Risk-On vs Late-Cycle)
        mu_other = pm.Normal(
            "mu_other",
            mu=means_init[:, other_idxs],
            sigma=0.30,
            shape=(n_states, len(other_idxs)),
            initval=means_init[:, other_idxs],
        )

        # Reassemble full (n_states, n_feat) mean matrix in original column order
        cols = [None] * n_feat
        cols[vix_idx] = mu_vix[:, None]
        for pos, orig_idx in enumerate(other_idxs):
            cols[orig_idx] = mu_other[:, pos : pos + 1]
        mu = pm.Deterministic("mu", pt.concatenate(cols, axis=1))

        # Floored emission std with LogNormal prior centered on MLE scales
        sigma_raw = pm.LogNormal(
            "sigma_raw",
            mu=np.log(sig_init),
            sigma=0.25,
            shape=(n_states, n_feat),
            initval=sig_init,
        )
        sigma = pm.Deterministic("sigma", 0.10 + sigma_raw)

        # Diagonal Gaussian frame log-likelihoods: (T, K)
        X_const = pt.as_tensor_variable(X_train)
        diff = (X_const[:, None, :] - mu[None, :, :]) / sigma[None, :, :]
        log_emit = -0.5 * pt.sum(diff**2 + 2.0 * pt.log(sigma[None, :, :]) + np.log(2.0 * np.pi), axis=-1)

        loglik = hmm_forward_op(log_emit, pt.log(P), pt.log(pi0))
        pm.Potential("hmm_loglik", loglik)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(4, chains),
            init="adapt_diag",
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
        )

    return idata


# ---------------------------------------------------------------------------
# Diagnostics & Post-processing
# ---------------------------------------------------------------------------

def evaluate_mcmc_diagnostics(
    idata: az.InferenceData,
    rhat_threshold: float = 1.01,
    max_divergences: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compute R-hat, ESS (bulk & tail), and divergences.
    Enforces Rule 4: raises RuntimeError if thresholds are violated.
    """
    summary = az.summary(idata, var_names=["mu_vix", "mu_other", "sigma", "P"])
    max_rhat = float(summary["r_hat"].max())
    min_ess_bulk = float(summary["ess_bulk"].min())
    min_ess_tail = float(summary["ess_tail"].min())
    n_div = int(idata.sample_stats["diverging"].sum().values)

    diag = {
        "max_rhat": round(max_rhat, 4),
        "min_ess_bulk": round(min_ess_bulk, 1),
        "min_ess_tail": round(min_ess_tail, 1),
        "n_divergences": n_div,
        "rhat_pass": max_rhat <= rhat_threshold,
        "div_pass": n_div <= max_divergences,
    }
    logger.info("MCMC Diagnostics: %s", diag)

    if not diag["rhat_pass"] or not diag["div_pass"]:
        raise RuntimeError(
            f"Rule 4 MCMC diagnostics gate FAILED: max_rhat={max_rhat:.4f} "
            f"(limit {rhat_threshold}), divergences={n_div} (limit {max_divergences})"
        )
    return summary, diag


def posterior_causal_filter(
    X: np.ndarray,
    idata: az.InferenceData,
    n_posterior_draws: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run causal forward filter across `n_posterior_draws` posterior parameter samples.
    Returns (mean_probs, q05_probs, q95_probs) each of shape (T, K).
    """
    rng = np.random.default_rng(seed)
    post = idata.posterior
    mu_all = post["mu"].values.reshape(-1, *post["mu"].shape[2:])
    sig_all = post["sigma"].values.reshape(-1, *post["sigma"].shape[2:])
    P_all = post["P"].values.reshape(-1, *post["P"].shape[2:])
    pi0_all = post["pi0"].values.reshape(-1, *post["pi0"].shape[2:])

    idx = rng.choice(len(mu_all), size=min(n_posterior_draws, len(mu_all)), replace=False)
    T, D = X.shape
    K = mu_all.shape[1]
    all_alpha = np.empty((len(idx), T, K), dtype=np.float64)

    for s_i, d_idx in enumerate(idx):
        mu_d = mu_all[d_idx]
        sig_d = sig_all[d_idx]
        # Convert weekly transition matrix P_w to daily transition matrix P_d
        # via first-order generator approximation: P_d = 0.8 * I + 0.2 * P_w
        P_d = 0.8 * np.eye(K) + 0.2 * P_all[d_idx]
        log_P_d = np.log(np.clip(P_d, 1e-12, 1.0))
        log_pi0_d = np.log(np.clip(pi0_all[d_idx], 1e-12, 1.0))

        diff = (X[:, None, :] - mu_d[None, :, :]) / sig_d[None, :, :]
        log_emit = -0.5 * np.sum(
            diff**2 + 2.0 * np.log(sig_d[None, :, :]) + np.log(2.0 * np.pi), axis=-1
        )
        _, alpha_d, _ = _hmm_forward_nb(log_emit, log_P_d, log_pi0_d)
        all_alpha[s_i] = alpha_d

    mean_p = all_alpha.mean(axis=0)
    q05_p = np.quantile(all_alpha, 0.05, axis=0)
    q95_p = np.quantile(all_alpha, 0.95, axis=0)
    return mean_p, q05_p, q95_p


def plot_trace_diagnostics(
    idata: az.InferenceData,
    out_path: str | Path = "reports/figures/phase4_trace_plots.png",
) -> None:
    """Save trace & posterior density plots for ordered VIX means and diagonal stickiness."""
    mu_vix = idata.posterior["mu_vix"].values  # (chains, draws, 5)
    P_vals = idata.posterior["P"].values       # (chains, draws, 5, 5)
    n_chains, n_draws, n_states = mu_vix.shape
    chain_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    # Top-left: mu_vix traces
    for c in range(n_chains):
        for k in range(n_states):
            axes[0, 0].plot(mu_vix[c, :, k], color=chain_colors[c % 4], alpha=0.45, lw=0.5)
    axes[0, 0].set_title("Trace: Ordered VIX Means (mu_vix)")
    axes[0, 0].set_xlabel("Draw")
    axes[0, 0].set_ylabel("Standardised VIX")

    # Top-right: mu_vix posterior distributions
    for k in range(n_states):
        vals = mu_vix[:, :, k].ravel()
        axes[0, 1].hist(vals, bins=40, density=True, alpha=0.55, label=f"State {k}")
    axes[0, 1].set_title("Posterior: Ordered VIX Means (mu_vix)")
    axes[0, 1].legend(fontsize=8)

    # Bottom-left: diagonal stickiness traces
    for c in range(n_chains):
        for k in range(n_states):
            axes[1, 0].plot(P_vals[c, :, k, k], color=chain_colors[c % 4], alpha=0.45, lw=0.5)
    axes[1, 0].set_title("Trace: Transition Stickiness P[k, k]")
    axes[1, 0].set_xlabel("Draw")
    axes[1, 0].set_ylabel("Probability")

    # Bottom-right: diagonal stickiness posteriors
    for k in range(n_states):
        vals = P_vals[:, :, k, k].ravel()
        axes[1, 1].hist(vals, bins=40, density=True, alpha=0.55, label=f"State {k}")
    axes[1, 1].set_title("Posterior: Transition Stickiness P[k, k]")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Phase 4 — Bayesian HMM NUTS Trace & Posterior Diagnostics (4 Chains)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Trace plots saved: %s", out_path)


def plot_phase4_vs_phase3(
    bayes_probs: pd.DataFrame,
    freq_path: str | Path = "data/processed/model_outputs/hmm_freq_probs.parquet",
    out_path: str | Path = "reports/figures/phase4_vs_phase3_comparison.png",
) -> None:
    """Compare Bayesian HMM Risk-Off & Risk-On probabilities (+90% CI) against Phase 3."""
    if not Path(freq_path).exists():
        logger.warning("Phase 3 output not found at %s; skipping comparison plot", freq_path)
        return
    freq = pd.read_parquet(freq_path)
    common = bayes_probs.index.intersection(freq.index)

    fig, axes = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
    for ax, reg, color in zip(axes, ["Risk-On", "Risk-Off"], ["#27ae60", "#c0392b"]):
        ax.plot(common, freq.loc[common, reg], color="grey", alpha=0.65, lw=0.9, label="Phase 3 (Freq HMM)")
        ax.plot(common, bayes_probs.loc[common, reg], color=color, lw=1.1, label="Phase 4 (Bayes HMM mean)")
        if f"{reg}_q05" in bayes_probs.columns and f"{reg}_q95" in bayes_probs.columns:
            ax.fill_between(
                common,
                bayes_probs.loc[common, f"{reg}_q05"],
                bayes_probs.loc[common, f"{reg}_q95"],
                color=color,
                alpha=0.2,
                label="90% Credible Interval",
            )
        ax.set_ylabel(f"P({reg})")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper left", ncol=3, fontsize=8)
        ax.grid(alpha=0.25)

    axes[0].set_title("Phase 4 Bayesian HMM vs Phase 3 Frequentist HMM (Causal Probabilities)", fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Phase 4 vs Phase 3 comparison plot saved: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Bayesian HMM (PyMC)")
    parser.add_argument("--config", default="configs/models.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    b_cfg = cfg["hmm_bayes"]
    seed = int(cfg.get("seed", 42))

    df = pd.read_parquet(cfg["features_parquet"])
    feat_cols, proxy_notes = resolve_feature_columns(df, b_cfg["feature_subset"])
    df_clean = df[feat_cols + ["regime_label"]].dropna().copy()

    # Strictly causal train/test split: train <= 2018-12-31, OOS >= 2019-01-01
    # Fit posterior on weekly (W-FRI) training observations (~620 weeks) to reduce
    # autocorrelation and enable fast 4-chain NUTS convergence, then filter daily.
    train_df_daily = df_clean[df_clean.index.year <= 2018]
    train_df_weekly = train_df_daily[feat_cols].resample("W-FRI").last().dropna()
    scaler = StandardScaler()
    scaler.fit(train_df_daily[feat_cols].values)
    X_train_weekly = scaler.transform(train_df_weekly.values)
    X_train_daily = scaler.transform(train_df_daily[feat_cols].values)
    X_all = scaler.transform(df_clean[feat_cols].values)

    logger.info(
        "Sampling Bayesian 5-state HMM on weekly train (%d weeks from <=2018) with %d draws × %d chains …",
        len(train_df_weekly),
        b_cfg["draws"],
        b_cfg["chains"],
    )
    idata = build_and_sample_bayesian_hmm(
        X_train=X_train_weekly,
        feature_names=feat_cols,
        n_states=int(b_cfg.get("n_states", 5)),
        draws=int(b_cfg.get("draws", 2000)),
        tune=int(b_cfg.get("tune", 1000)),
        chains=int(b_cfg.get("chains", 4)),
        target_accept=float(b_cfg.get("target_accept", 0.95)),
        seed=seed,
    )

    # Rule 4 Diagnostics Gate
    summary_df, diag = evaluate_mcmc_diagnostics(
        idata,
        rhat_threshold=float(b_cfg.get("rhat_threshold", 1.01)),
        max_divergences=int(b_cfg.get("max_divergences", 0)),
    )
    # Save InferenceData before plotting so MCMC results are never lost
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        idata.to_netcdf(out_dir / "hmm_bayes_idata.nc")
    except Exception as exc:  # noqa: BLE001
        import pickle
        logger.warning("NetCDF export unavailable (%s); saving pickle fallback", exc)
        with open(out_dir / "hmm_bayes_idata.pkl", "wb") as pf:
            pickle.dump(idata, pf)

    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    summary_df.to_csv("reports/tables/phase4_mcmc_diagnostics.csv")
    plot_trace_diagnostics(idata)

    # Posterior causal forward filter over entire daily timeline (train + OOS)
    mean_p, q05_p, q95_p = posterior_causal_filter(X_all, idata, n_posterior_draws=200, seed=seed)
    train_mask = df_clean.index.year <= 2018
    state_map = map_states_to_regimes(X_train_daily, mean_p[train_mask], feat_cols)
    logger.info("Bayesian HMM state-to-regime mapping: %s", state_map)

    probs_df = reorder_probs_by_regime(mean_p, state_map)
    probs_df.index = df_clean.index
    q05_df = reorder_probs_by_regime(q05_p, state_map).set_axis([f"{r}_q05" for r in REGIME_NAMES], axis=1)
    q95_df = reorder_probs_by_regime(q95_p, state_map).set_axis([f"{r}_q95" for r in REGIME_NAMES], axis=1)
    q05_df.index = df_clean.index
    q95_df.index = df_clean.index

    # Transition matrix & expected durations with 90% CI
    P_draws = idata.posterior["P"].values.reshape(-1, 5, 5)
    inv_map = {reg: st for st, reg in state_map.items()}
    order_idx = [inv_map[r] for r in REGIME_NAMES]
    P_ord = P_draws[:, order_idx, :][:, :, order_idx]
    P_mean = P_ord.mean(axis=0)
    diag_draws = np.diagonal(P_ord, axis1=1, axis2=2)
    dur_draws = 1.0 / np.clip(1.0 - diag_draws, 1e-6, 1.0)

    trans_ci = pd.DataFrame(P_mean, index=REGIME_NAMES, columns=REGIME_NAMES)
    trans_ci["stickiness_mean"] = diag_draws.mean(axis=0)
    trans_ci["stickiness_q05"] = np.quantile(diag_draws, 0.05, axis=0)
    trans_ci["stickiness_q95"] = np.quantile(diag_draws, 0.95, axis=0)
    trans_ci["duration_mean_days"] = dur_draws.mean(axis=0)
    trans_ci["duration_q05_days"] = np.quantile(dur_draws, 0.05, axis=0)
    trans_ci["duration_q95_days"] = np.quantile(dur_draws, 0.95, axis=0)
    trans_ci.round(4).to_csv("reports/tables/phase4_transitions_ci.csv")

    # OOS Calibration Metrics (2019+)
    oos_probs = probs_df[probs_df.index.year >= 2019]
    oos_metrics = compute_calibration_metrics(oos_probs, df_clean["regime_label"])
    logger.info("Bayesian HMM OOS Calibration (2019+): %s", oos_metrics)

    # Combine probabilities + credible intervals + lineage
    full_out = pd.concat([probs_df, q05_df, q95_df], axis=1)
    full_out["is_oos"] = full_out.index.year >= 2019
    full_out["predicted_regime"] = full_out[REGIME_NAMES].idxmax(axis=1)
    full_out["conviction"] = full_out[REGIME_NAMES].max(axis=1)

    data_hash = str(df["lineage_data_version_hash"].iloc[0]) if "lineage_data_version_hash" in df.columns else "unknown"
    feat_hash = str(df["lineage_feature_set_version"].iloc[0]) if "lineage_feature_set_version" in df.columns else "unknown"
    lineage = build_lineage_record(
        phase=4,
        data_hash=data_hash,
        feature_hash=feat_hash,
        model_hash=hash_dataframe(probs_df),
        config_hash=hash_config(cfg),
        extra={
            "model": "hmm_bayes_pymc",
            "features_used": feat_cols,
            "proxy_notes": proxy_notes,
            "state_map": state_map,
            "mcmc_diagnostics": diag,
            "oos_metrics": oos_metrics,
        },
    )
    full_out = attach_lineage(full_out, lineage)
    full_out.to_parquet(out_dir / "hmm_bayes_probs.parquet")
    save_lineage_json(lineage, "data/lineage/phase4_lineage.json")

    plot_phase4_vs_phase3(full_out)
    logger.info("Phase 4 Bayesian HMM complete.")


if __name__ == "__main__":
    main()
