# test_ablation.py
# Ablation study for AML-Detect.
#
# Purpose:
#   Determine which detector contributes positively to the ensemble and which
#   does not. The previous version reported the ensemble's internal scoring
#   value, which is not a standard classification metric and cannot support
#   claims about component contribution. This version reports precision,
#   recall, F1 and PR-AUC.
#
# Configurations tested:
#   - Full ensemble
#   - Without autoencoder
#   - Without GNN
#   - Without OCSVM
#   - Autoencoder alone
#   - GNN alone
#   - OCSVM alone
#
# Output:
#   results/metrics/ablation_metrics.csv
#
# Usage:
#   python src/ensemble/test_ablation.py

import os
import sys
import csv
import numpy as np

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_balanced_sample(n_samples=5000, seed=42):
    """Load a reproducible balanced sample of normal and attack observations."""
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    rng = np.random.default_rng(seed)

    normal_idx = np.where(y_test == 0)[0]
    attack_idx = np.where(y_test == 1)[0]
    n_each = min(n_samples // 2, len(normal_idx), len(attack_idx))

    normal_sample = rng.choice(normal_idx, size=n_each, replace=False)
    attack_sample = rng.choice(attack_idx, size=n_each, replace=False)

    idx = np.concatenate([normal_sample, attack_sample])
    rng.shuffle(idx)

    return X_test[idx], y_test[idx]


# ----------------------------------------------------------------------
# Score computation per configuration
# ----------------------------------------------------------------------

def compute_configuration_scores(ensemble, X_eval, config):
    """
    Compute ensemble scores for a given ablation configuration.

    Args:
        ensemble: EnsembleDetector instance
        X_eval:   feature matrix
        config:   one of "full", "without_autoencoder", "without_gnn",
                  "without_ocsvm", "autoencoder_alone", "gnn_alone",
                  "ocsvm_alone"

    Returns:
        scores array of length len(X_eval)
    """
    # Store original availability
    saved_ae = ensemble.ae
    saved_gnn = ensemble.gnn
    saved_ocsvm = ensemble.ocsvm
    saved_weights = list(ensemble.weights)

    try:
        if config == "full":
            pass

        elif config == "without_autoencoder":
            ensemble.ae = None
            ensemble.weights = [0.0, 0.5, 0.5]

        elif config == "without_gnn":
            ensemble.gnn = None
            ensemble.weights = [0.5, 0.0, 0.5]

        elif config == "without_ocsvm":
            ensemble.ocsvm = None
            ensemble.weights = [0.5, 0.5, 0.0]

        elif config == "autoencoder_alone":
            ensemble.gnn = None
            ensemble.ocsvm = None
            ensemble.weights = [1.0, 0.0, 0.0]

        elif config == "gnn_alone":
            ensemble.ae = None
            ensemble.ocsvm = None
            ensemble.weights = [0.0, 1.0, 0.0]

        elif config == "ocsvm_alone":
            ensemble.ae = None
            ensemble.gnn = None
            ensemble.weights = [0.0, 0.0, 1.0]

        else:
            raise ValueError(f"Unknown configuration: {config}")

        # Re-normalise weights over the available components
        weights = np.array(ensemble.weights[:3], dtype=float)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        ensemble.weights = weights.tolist()

        scores = np.zeros(len(X_eval))
        for i, x in enumerate(X_eval):
            ensemble_score, _, _, _ = ensemble.predict(x, mode="fixed")
            scores[i] = ensemble_score

        return scores

    finally:
        # Restore original state
        ensemble.ae = saved_ae
        ensemble.gnn = saved_gnn
        ensemble.ocsvm = saved_ocsvm
        ensemble.weights = saved_weights


def compute_metrics(y_true, scores):
    """Compute precision, recall, F1, PR-AUC and ROC-AUC at threshold 0.5."""
    preds = (scores >= 0.5).astype(int)

    precision = float(precision_score(y_true, preds, zero_division=0))
    recall = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    if len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, scores))
        roc_auc = float(roc_auc_score(y_true, scores))
    else:
        pr_auc = 0.0
        roc_auc = 0.0

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
    }


# ----------------------------------------------------------------------
# Main ablation
# ----------------------------------------------------------------------

def run_ablation():
    from ensemble import EnsembleDetector

    print("=" * 60)
    print("Ablation Study")
    print("=" * 60)

    X_test = np.load("data/processed/X_test.npy")
    input_dim = X_test.shape[1]

    print("\nLoading ensemble...")
    ensemble = EnsembleDetector(input_dim, enable_mtd=False)

    print("\nLoading balanced sample of 5,000 observations...")
    X_eval, y_eval = load_balanced_sample(n_samples=5000, seed=42)
    print(f"   X_eval: {X_eval.shape}")
    print(f"   y_eval: normal={(y_eval == 0).sum()}, attack={(y_eval == 1).sum()}")

    configs = [
        "full",
        "without_autoencoder",
        "without_gnn",
        "without_ocsvm",
        "autoencoder_alone",
        "gnn_alone",
        "ocsvm_alone",
    ]

    rows = []
    print("\nRunning configurations:")
    for config in configs:
        print(f"   {config}...")
        scores = compute_configuration_scores(ensemble, X_eval, config)
        metrics = compute_metrics(y_eval, scores)
        row = {"configuration": config}
        row.update(metrics)
        rows.append(row)
        print(f"      precision={metrics['precision']:.4f}  "
              f"recall={metrics['recall']:.4f}  "
              f"f1={metrics['f1']:.4f}  "
              f"pr_auc={metrics['pr_auc']:.4f}  "
              f"roc_auc={metrics['roc_auc']:.4f}")

    # Write CSV
    os.makedirs("results/metrics", exist_ok=True)
    out_path = "results/metrics/ablation_metrics.csv"
    fieldnames = ["configuration", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nAblation results saved to {out_path}")

    # Interpretation
    print("\nInterpretation:")
    full_f1 = next(r["f1"] for r in rows if r["configuration"] == "full")
    for r in rows:
        if r["configuration"] == "full":
            continue
        if r["configuration"].startswith("without_"):
            component = r["configuration"].replace("without_", "")
            delta = r["f1"] - full_f1
            if delta < -0.01:
                verdict = "contributes positively"
            elif delta > 0.01:
                verdict = "detrimental under this configuration"
            else:
                verdict = "redundant"
            print(f"   {component}: F1 delta = {delta:+.4f} ({verdict})")

    print("\nAblation study complete.")
    print("=" * 60)

    return rows


if __name__ == "__main__":
    run_ablation()