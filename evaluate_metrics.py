# evaluate_metrics.py
# Held-out classification evaluation for AML-Detect.
#
# Location:
#   Project root: C:\ResearchP\aml-detect\evaluate_metrics.py
#   Resolves project root and imports EnsembleDetector directly from
#   src/ensemble/ensemble.py so that no __init__.py is required.
#
# Purpose:
#   Produce every record that Section 4.8 reports:
#     - metrics on the balanced sample and the imbalanced test subset
#     - bootstrap confidence intervals
#     - per-class metrics
#     - threshold sweep
#     - reliability diagram data
#     - confusion matrix, PR curve, ROC curve, reliability diagram
#     - structured JSON record
#     - no placeholder detection delay (stated as not measured)
#
# Outputs (results/metrics/):
#   evaluation_balanced.csv
#   evaluation_imbalanced.csv
#   evaluation_ci.csv
#   evaluation_per_class.csv
#   evaluation_threshold_sweep.csv
#   evaluation_calibration.csv
#   evaluation_ensemble_stats.csv
#   evaluation_results.json
#
# Outputs (results/figures/):
#   confusion_matrix.png
#   precision_recall_curve.png
#   roc_curve.png
#   reliability_diagram.png
#
# Usage:
#   From project root:  python evaluate_metrics.py

import os
import sys
import csv
import json
import time
import importlib.util
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


# ----------------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------------

_this_file = os.path.abspath(__file__)
_this_dir = os.path.dirname(_this_file)


def _find_project_root(start):
    current = os.path.abspath(start)
    for _ in range(6):
        if os.path.isdir(os.path.join(current, "data")) and \
           os.path.isdir(os.path.join(current, "src")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.getcwd()


PROJECT_ROOT = _find_project_root(_this_dir)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
ENSEMBLE_DIR = os.path.join(SRC_DIR, "ensemble")
RISK_DIR = os.path.join(SRC_DIR, "risk_engine")
MTD_DIR = os.path.join(SRC_DIR, "mtd_coordinator")

for _p in (SRC_DIR, ENSEMBLE_DIR, RISK_DIR, MTD_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "results", "models")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Robust import of EnsembleDetector
# ----------------------------------------------------------------------

def _load_ensemble_class():
    ensemble_file = os.path.join(ENSEMBLE_DIR, "ensemble.py")
    if not os.path.isfile(ensemble_file):
        raise FileNotFoundError(
            f"ensemble.py not found at {ensemble_file}. "
            f"Confirm the repository layout."
        )
    spec = importlib.util.spec_from_file_location("aml_ensemble_module", ensemble_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aml_ensemble_module"] = module
    spec.loader.exec_module(module)
    return module.EnsembleDetector


EnsembleDetector = _load_ensemble_class()


# ----------------------------------------------------------------------
# Ensemble and scoring
# ----------------------------------------------------------------------

def load_ensemble(input_dim):
    """Load the ensemble with MTD disabled for deterministic scoring."""
    return EnsembleDetector(input_dim, enable_mtd=False)


def score_samples(ensemble, X):
    """Score every sample with the fixed-weight path, no MTD."""
    scores = np.zeros(len(X))
    for i, x in enumerate(X):
        ensemble_score, _, _, _ = ensemble.predict(x, mode="fixed")
        scores[i] = ensemble_score
    return scores


# ----------------------------------------------------------------------
# Metrics helpers
# ----------------------------------------------------------------------

def compute_metrics(y_true, scores, threshold=0.5):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    n = len(y_true)
    return {
        "accuracy": float((preds == y_true).mean()),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total_samples": int(n),
    }


def write_metrics_csv(path, metrics, label):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["evaluation_set", label])
        for k, v in metrics.items():
            writer.writerow([k, v if isinstance(v, str) else round(float(v), 6)])


def bootstrap_ci(y_true, scores, n_boot=1000, seed=42, threshold=0.5):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metrics = {"precision": [], "recall": [], "f1": [], "roc_auc": [], "pr_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        preds = (s_b >= threshold).astype(int)
        metrics["precision"].append(precision_score(y_b, preds, zero_division=0))
        metrics["recall"].append(recall_score(y_b, preds, zero_division=0))
        metrics["f1"].append(f1_score(y_b, preds, zero_division=0))
        metrics["roc_auc"].append(roc_auc_score(y_b, s_b))
        metrics["pr_auc"].append(average_precision_score(y_b, s_b))
    out = {}
    for k, vals in metrics.items():
        vals = np.array(vals)
        if len(vals) == 0:
            out[k] = (0.0, 0.0)
        else:
            out[k] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return out


def write_ci_csv(path, ci):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "ci_lower", "ci_upper"])
        for k, (lo, hi) in ci.items():
            writer.writerow([k, round(lo, 6), round(hi, 6)])


def per_class_metrics(y_true, scores, threshold=0.5):
    preds = (scores >= threshold).astype(int)
    rows = []
    for cls, name in [(0, "Normal"), (1, "Attack")]:
        mask_true = (y_true == cls)
        mask_pred = (preds == cls)
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())
        support = int(mask_true.sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "class": name,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        })
    return rows


def write_per_class_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def threshold_sweep(y_true, scores, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    rows = []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 6),
            "f1": round(float(f1_score(y_true, preds, zero_division=0)), 6),
        })
    return rows


def write_threshold_sweep_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "f1"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def reliability_diagram_data(y_true, scores, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            rows.append({
                "bin_lower": round(float(lo), 2),
                "bin_upper": round(float(hi), 2),
                "mean_predicted": 0.0,
                "observed_frequency": 0.0,
                "count": 0,
            })
            continue
        rows.append({
            "bin_lower": round(float(lo), 2),
            "bin_upper": round(float(hi), 2),
            "mean_predicted": round(float(scores[mask].mean()), 6),
            "observed_frequency": round(float((y_true[mask] == 1).mean()), 6),
            "count": int(mask.sum()),
        })
    return rows


def write_calibration_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin_lower", "bin_upper", "mean_predicted", "observed_frequency", "count"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def ensemble_stats_by_class(y_true, scores):
    normal = scores[y_true == 0]
    attack = scores[y_true == 1]
    return [
        {"statistic": "mean", "normal": round(float(normal.mean()), 6),
         "attack": round(float(attack.mean()), 6),
         "difference": round(float(attack.mean() - normal.mean()), 6)},
        {"statistic": "std", "normal": round(float(normal.std(ddof=1)), 6),
         "attack": round(float(attack.std(ddof=1)), 6),
         "difference": round(float(attack.std(ddof=1) - normal.std(ddof=1)), 6)},
    ]


def write_ensemble_stats_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["statistic", "normal", "attack", "difference"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted Normal", "Predicted Attack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True Normal", "True Attack"])
    ax.set_title("Confusion matrix (balanced sample)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pr_curve(y_true, scores, path):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="steelblue", label=f"PR (AUC = {pr_auc:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (balanced sample)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, scores, path):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve (balanced sample)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_reliability(calibration_rows, path):
    xs = [r["mean_predicted"] for r in calibration_rows if r["count"] > 0]
    ys = [r["observed_frequency"] for r in calibration_rows if r["count"] > 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.plot(xs, ys, marker="o", color="steelblue", label="Observed")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed attack frequency")
    ax.set_title("Reliability diagram (balanced sample)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Held-Out Classification Evaluation")
    print("=" * 60)

    print("\nLoading data...")
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"   X_test: {X_test.shape}")
    print(f"   y_test: normal={(y_test == 0).sum()}, attack={(y_test == 1).sum()}")

    input_dim = X_test.shape[1]

    print("\nLoading ensemble...")
    ensemble = load_ensemble(input_dim)

    # --- Balanced sample -------------------------------------------------
    print("\nDrawing balanced sample of 5,000 observations (seed 42)...")
    rng = np.random.default_rng(42)
    normal_idx = np.where(y_test == 0)[0]
    attack_idx = np.where(y_test == 1)[0]
    bal_idx = np.concatenate([
        rng.choice(normal_idx, size=2500, replace=False),
        rng.choice(attack_idx, size=2500, replace=False),
    ])
    rng.shuffle(bal_idx)
    X_bal = X_test[bal_idx]
    y_bal = y_test[bal_idx]

    print("\nScoring balanced sample...")
    t0 = time.time()
    scores_bal = score_samples(ensemble, X_bal)
    elapsed_bal = time.time() - t0
    print(f"   Scored {len(X_bal)} in {elapsed_bal:.2f}s "
          f"({elapsed_bal / len(X_bal) * 1000:.2f} ms/sample)")

    metrics_bal = compute_metrics(y_bal, scores_bal)
    write_metrics_csv(os.path.join(METRICS_DIR, "evaluation_balanced.csv"),
                      metrics_bal, "balanced_5000")

    print("\nBalanced sample metrics:")
    for k, v in metrics_bal.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    # --- Imbalanced subset ----------------------------------------------
    print("\nScoring imbalanced subset of the test set (20,000 samples)...")
    imb_size = 20000
    imb_idx = rng.choice(len(X_test), size=imb_size, replace=False)
    X_imb = X_test[imb_idx]
    y_imb = y_test[imb_idx]

    t0 = time.time()
    scores_imb = score_samples(ensemble, X_imb)
    elapsed_imb = time.time() - t0
    print(f"   Scored {len(X_imb)} in {elapsed_imb:.2f}s")
    print(f"   Imbalanced subset: normal={(y_imb == 0).sum()}, "
          f"attack={(y_imb == 1).sum()}")

    metrics_imb = compute_metrics(y_imb, scores_imb)
    write_metrics_csv(os.path.join(METRICS_DIR, "evaluation_imbalanced.csv"),
                      metrics_imb, "imbalanced_subset_20000")

    print("\nImbalanced subset metrics:")
    for k, v in metrics_imb.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    # --- Baselines -------------------------------------------------------
    print("\nComputing baselines...")
    majority_acc = float((y_bal == 0).mean()) if (y_bal == 0).sum() > (y_bal == 1).sum() else float((y_bal == 1).mean())
    random_auc = 0.5
    random_pr_auc = float((y_bal == 1).mean())

    print(f"   Majority-class accuracy: {majority_acc:.4f}")
    print(f"   Random ROC-AUC: {random_auc:.4f}")
    print(f"   Random PR-AUC (positive rate): {random_pr_auc:.4f}")

    # --- Bootstrap confidence intervals ----------------------------------
    print("\nComputing bootstrap confidence intervals (1,000 resamples)...")
    ci = bootstrap_ci(y_bal, scores_bal, n_boot=1000, seed=42)
    write_ci_csv(os.path.join(METRICS_DIR, "evaluation_ci.csv"), ci)
    for k, (lo, hi) in ci.items():
        print(f"   {k}: [{lo:.4f}, {hi:.4f}]")

    # --- Per-class metrics ----------------------------------------------
    print("\nComputing per-class metrics...")
    pc = per_class_metrics(y_bal, scores_bal)
    write_per_class_csv(os.path.join(METRICS_DIR, "evaluation_per_class.csv"), pc)
    for r in pc:
        print(f"   {r['class']}: precision={r['precision']:.4f} "
              f"recall={r['recall']:.4f} f1={r['f1']:.4f} support={r['support']}")

    # --- Threshold sweep -------------------------------------------------
    print("\nRunning threshold sweep...")
    sweep = threshold_sweep(y_bal, scores_bal)
    write_threshold_sweep_csv(os.path.join(METRICS_DIR, "evaluation_threshold_sweep.csv"), sweep)
    for r in sweep:
        print(f"   t={r['threshold']:.2f}  precision={r['precision']:.4f}  "
              f"recall={r['recall']:.4f}  f1={r['f1']:.4f}")

    # --- Reliability diagram data ---------------------------------------
    print("\nComputing reliability diagram data...")
    cal = reliability_diagram_data(y_bal, scores_bal, bins=10)
    write_calibration_csv(os.path.join(METRICS_DIR, "evaluation_calibration.csv"), cal)

    # --- Ensemble score statistics by class -----------------------------
    print("\nComputing ensemble score statistics by class...")
    ens_stats = ensemble_stats_by_class(y_bal, scores_bal)
    write_ensemble_stats_csv(os.path.join(METRICS_DIR, "evaluation_ensemble_stats.csv"), ens_stats)
    for r in ens_stats:
        print(f"   {r['statistic']}: normal={r['normal']} attack={r['attack']} "
              f"difference={r['difference']}")

    # --- Figures ---------------------------------------------------------
    print("\nPlotting figures...")
    cm = confusion_matrix(y_bal, (scores_bal >= 0.5).astype(int))
    plot_confusion(cm, os.path.join(FIGURES_DIR, "confusion_matrix.png"))
    plot_pr_curve(y_bal, scores_bal, os.path.join(FIGURES_DIR, "precision_recall_curve.png"))
    plot_roc_curve(y_bal, scores_bal, os.path.join(FIGURES_DIR, "roc_curve.png"))
    plot_reliability(cal, os.path.join(FIGURES_DIR, "reliability_diagram.png"))
    print(f"   Figures written to {FIGURES_DIR}")

    # --- Structured JSON record -----------------------------------------
    summary = {
        "balanced_sample": {
            "seed": 42,
            "normal": int((y_bal == 0).sum()),
            "attack": int((y_bal == 1).sum()),
            "metrics": metrics_bal,
        },
        "imbalanced_subset": {
            "sample_size": imb_size,
            "normal": int((y_imb == 0).sum()),
            "attack": int((y_imb == 1).sum()),
            "metrics": metrics_imb,
        },
        "baselines": {
            "majority_class_accuracy": majority_acc,
            "random_roc_auc": random_auc,
            "random_pr_auc": random_pr_auc,
        },
        "confidence_intervals": {
            k: {"lower": lo, "upper": hi} for k, (lo, hi) in ci.items()
        },
        "per_class": pc,
        "threshold_sweep": sweep,
        "calibration": cal,
        "ensemble_stats_by_class": ens_stats,
        "timings": {
            "balanced_total_seconds": round(elapsed_bal, 4),
            "balanced_inference_ms": round(elapsed_bal / len(X_bal) * 1000, 4),
            "imbalanced_total_seconds": round(elapsed_imb, 4),
            "imbalanced_inference_ms": round(elapsed_imb / len(X_imb) * 1000, 4),
        },
        "detection_delay": {
            "measured": False,
            "reason": "The Kaggle derivative does not preserve attack onset "
                      "timestamps. Detection delay is not measured.",
        },
        "notes": [
            "Balanced sample changes class prevalence to 50%. Precision, F1 "
            "and PR-AUC are prevalence-sensitive.",
            "Imbalanced metrics are computed on a 20,000-sample subset of the "
            "full test set, not the full 341,919 rows.",
            "The artefact is a research prototype. Current performance does "
            "not support deployment.",
        ],
    }
    json_path = os.path.join(METRICS_DIR, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   JSON summary written to {json_path}")

    print("\nHeld-out evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()