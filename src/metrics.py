import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)


def _calibration_setup(y_true, y_prob_pos, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob_pos = np.asarray(y_prob_pos, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob_pos, bins) - 1, 0, n_bins - 1)

    return y_true, y_prob_pos, bin_ids


def expected_calibration_error(y_true, y_prob_pos, n_bins=15):
    y_true, y_prob_pos, bin_ids = _calibration_setup(y_true, y_prob_pos, n_bins)

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            acc_bin = y_true[mask].mean()          # empirical positive rate
            conf_bin = y_prob_pos[mask].mean()     # mean predicted probability
            ece += (mask.sum() / n) * abs(acc_bin - conf_bin)

    return float(ece)



def maximum_calibration_error(y_true, y_prob_pos, n_bins=15):
    y_true, y_prob_pos, bin_ids = _calibration_setup(y_true, y_prob_pos, n_bins)

    worst = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            acc_bin = y_true[mask].mean()
            conf_bin = y_prob_pos[mask].mean()
            worst = max(worst, abs(acc_bin - conf_bin))

    return float(worst)


def compute_metrics(y_true, y_pred, y_prob_pos):
    # Confusion matrix in [benign, malignant] label order
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # False negative rate = missed malignant cases
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "auc": float(roc_auc_score(y_true, y_prob_pos)),
        "fnr": float(fnr),
        "cm": cm,
        "brier": float(brier_score_loss(y_true, y_prob_pos)),
        "ece": float(expected_calibration_error(y_true, y_prob_pos)),
        "mce": float(maximum_calibration_error(y_true, y_prob_pos)),
    }


def print_results(label, m):
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Accuracy  : {m['accuracy']:.4f}")
    print(f"F1        : {m['f1']:.4f}")
    print(f"AUC       : {m['auc']:.4f}")
    print(f"FNR       : {m['fnr']:.4f}")
    print(f"Brier     : {m['brier']:.4f}")
    print(f"ECE       : {m['ece']:.4f}")
    print(f"MCE       : {m['mce']:.4f}")