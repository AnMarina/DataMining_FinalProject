"""
plots.py — Visualization helpers for the UF-SSL experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_vpl_dynamics(history, metrics_sup):
    """Plot Vanilla Pseudo-Labeling training dynamics."""
    import pandas as pd

    if not history:
        print("No VPL history to plot.")
        return

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Pseudo-label count by round
    axes[0].bar(df["round"], df["selected"], color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Pseudo-labels added")
    axes[0].set_title("Pseudo-labels selected per round")
    axes[0].grid(True, alpha=0.3)

    # Accuracy by round
    axes[1].plot(df["round"], df["test_acc"], marker="o", color="steelblue", label="VPL accuracy")
    axes[1].axhline(metrics_sup["accuracy"], color="gray", linestyle="--", label="Supervised baseline")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Test accuracy over rounds")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # FNR by round
    axes[2].plot(df["round"], df["test_fnr"], marker="s", color="coral", label="FNR (VPL)")
    axes[2].axhline(metrics_sup["fnr"], color="gray", linestyle="--", label="Supervised baseline FNR")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("False negative rate")
    axes[2].set_title("FNR over rounds\n(lower = fewer missed malignant cases)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Vanilla Pseudo-Labeling (tau=0.95) — Training Dynamics", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_calibration(y_test, y_prob_sup):
    """Plot a reliability diagram for the supervised MLP baseline."""
    fig, ax = plt.subplots(figsize=(6, 6))

    frac_pos, mean_pred = calibration_curve(y_test, y_prob_sup, n_bins=10)

    ax.plot(mean_pred, frac_pos, "s-", color="steelblue", label="MLP supervised only")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.05, color="coral", label="Overconfident region")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram\nPoints below diagonal = model is overconfident")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_entropy_distribution(probs_unlab, tau=0.95):
    """Plot predictive entropy on the unlabeled pool."""
    from src.model import predictive_entropy

    entropy = predictive_entropy(probs_unlab)
    max_conf_unlab = probs_unlab.max(axis=1)
    accepted_mask = max_conf_unlab >= tau

    fig, ax = plt.subplots(figsize=(8, 4))

    # Rejected vs accepted by tau
    ax.hist(
        entropy[~accepted_mask],
        bins=25,
        color="steelblue",
        alpha=0.7,
        label=f"Rejected (conf < {tau})"
    )
    ax.hist(
        entropy[accepted_mask],
        bins=25,
        color="coral",
        alpha=0.7,
        label=f"Accepted (conf >= {tau})"
    )

    if accepted_mask.any():
        mean_h = entropy[accepted_mask].mean()
        ax.axvline(
            mean_h,
            color="darkred",
            linestyle="--",
            label=f"Mean accepted entropy = {mean_h:.3f}"
        )

    ax.set_xlabel("Predictive entropy H(p)")
    ax.set_ylabel("Number of unlabeled samples")
    ax.set_title(
        "Predictive entropy of unlabeled samples\n"
        "Accepted pseudo-labels are high-confidence, but not perfectly certain"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if accepted_mask.any():
        print(f"\nEntropy stats for accepted samples (conf >= {tau}):")
        print(f"  Count: {accepted_mask.sum()}")
        print(f"  Mean entropy: {entropy[accepted_mask].mean():.4f}")
        print(f"  Max entropy: {entropy[accepted_mask].max():.4f}")
        pct = (entropy[accepted_mask] > 0.10).mean() * 100
        print(f"  % with H > 0.10: {pct:.1f}%")
    else:
        print(f"\nNo unlabeled samples passed tau={tau}.")