# test_adversarial.py
# FGSM adversarial evaluation for AML-Detect.
#
# Location:
#   This script sits at the project root: C:\ResearchP\aml-detect\test_adversarial.py
#   It resolves the project root and adds src/ and src/ensemble/ to sys.path
#   so that autoencoder.py and other modules import correctly regardless of
#   the current working directory.
#
# Purpose:
#   Measure whether the autoencoder-based detector degrades under FGSM
#   perturbations at increasing epsilon values. Verify perturbation
#   application, clip to valid sensor ranges, exclude immutable variables
#   and report attack success rate per epsilon.
#
# Outputs:
#   results/metrics/adversarial_results.csv
#   results/figures/adversarial_score_distribution.png
#
# Usage:
#   From project root:  python test_adversarial.py

import os
import sys
import csv
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Path setup (fixed)
# ----------------------------------------------------------------------
# The script may sit at the project root or inside src/ensemble/.
# Either way, add src/ and src/ensemble/ to sys.path so that
# `from autoencoder import Autoencoder` resolves.

_this_file = os.path.abspath(__file__)
_this_dir = os.path.dirname(_this_file)

# Walk up to find the directory that contains both data/ and src/
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
# Load data and models
# ----------------------------------------------------------------------

def load_data(n_samples=500, seed=42):
    """Load a balanced sample from the test partition."""
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    rng = np.random.default_rng(seed)
    normal_idx = np.where(y_test == 0)[0]
    attack_idx = np.where(y_test == 1)[0]

    n_each = min(n_samples // 2, len(normal_idx), len(attack_idx))
    normal_sample = rng.choice(normal_idx, size=n_each, replace=False)
    attack_sample = rng.choice(attack_idx, size=n_each, replace=False)

    idx = np.concatenate([normal_sample, attack_sample])
    rng.shuffle(idx)

    return X_test[idx], y_test[idx]


def load_autoencoder(input_dim):
    """Load the trained autoencoder in eval mode."""
    from autoencoder import Autoencoder

    model = Autoencoder(input_dim, latent_dim=128)
    state_path = os.path.join(MODELS_DIR, "autoencoder.pt")
    model.load_state_dict(torch.load(state_path, map_location=torch.device("cpu")))
    model.eval()  # Critical: disable dropout and batch-norm updates

    threshold_path = os.path.join(DATA_DIR, "autoencoder_threshold.txt")
    with open(threshold_path, "r") as f:
        first = f.readline().strip()
    # Accept either "95th percentile: 0.759046" or a bare float
    if ":" in first:
        threshold = float(first.split(":")[1].strip())
    else:
        threshold = float(first)

    return model, threshold


# ----------------------------------------------------------------------
# Sensor range constraints
# ----------------------------------------------------------------------

def compute_feature_ranges(X_train):
    """Per-feature min and max from the training partition."""
    return X_train.min(axis=0), X_train.max(axis=0)


def identify_immutable_features(X_train, std_threshold=1e-6):
    """Features with near-zero variance are treated as immutable."""
    stds = X_train.std(axis=0)
    return stds < std_threshold


# ----------------------------------------------------------------------
# Autoencoder scoring
# ----------------------------------------------------------------------

def autoencoder_scores(model, X):
    """Reconstruction error per sample."""
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        recon = model(x_tensor)
        mse = torch.mean((recon - x_tensor) ** 2, dim=1).numpy()
    return mse


def ensemble_score_from_ae(error, threshold):
    """Bounded score in [0, 1] from reconstruction error."""
    if threshold is None or threshold <= 0:
        return float(np.clip(error, 0.0, 1.0))
    return float(np.clip(error / threshold, 0.0, 1.0))


def predict_from_ae(model, X, threshold):
    """Binary predictions using the reconstruction-error threshold."""
    errors = autoencoder_scores(model, X)
    return (errors >= threshold).astype(int), errors


# ----------------------------------------------------------------------
# FGSM attack
# ----------------------------------------------------------------------

def fgsm_attack(model, X, epsilon, feature_min, feature_max, immutable_mask):
    """
    Apply FGSM perturbation to the input.

    Model must be in eval mode. Perturbation is added to the input, zeroed
    on immutable features, then clipped to training ranges.
    """
    x_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)

    recon = model(x_tensor)
    loss = torch.mean((recon - x_tensor) ** 2)

    model.zero_grad()
    loss.backward()

    if x_tensor.grad is None:
        raise RuntimeError(
            "Gradient is None. Confirm the model is in eval mode and the "
            "loss depends on the input tensor."
        )

    grad = x_tensor.grad.detach().numpy()
    perturbation = epsilon * np.sign(grad)
    perturbation[:, immutable_mask] = 0.0

    X_adv = X + perturbation
    X_adv = np.clip(X_adv, feature_min, feature_max)

    return X_adv.astype(np.float32)


# ----------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------

def run_adversarial_evaluation():
    print("=" * 60)
    print("FGSM Adversarial Evaluation")
    print("=" * 60)

    print("\nLoading data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_eval, y_eval = load_data(n_samples=500, seed=42)
    input_dim = X_eval.shape[1]
    print(f"   X_train: {X_train.shape}")
    print(f"   X_eval:  {X_eval.shape}")

    print("\nComputing feature ranges and immutable mask...")
    feature_min, feature_max = compute_feature_ranges(X_train)
    immutable_mask = identify_immutable_features(X_train)
    print(f"   Immutable features: {int(immutable_mask.sum())} of {input_dim}")

    print("\nLoading autoencoder (eval mode)...")
    model, threshold = load_autoencoder(input_dim)
    print(f"   Threshold: {threshold:.6f}")

    print("\nEvaluating clean predictions...")
    clean_preds, clean_errors = predict_from_ae(model, X_eval, threshold)
    clean_acc = float((clean_preds == y_eval).mean())
    clean_scores = np.array([ensemble_score_from_ae(e, threshold) for e in clean_errors])
    print(f"   Clean accuracy: {clean_acc:.4f}")

    epsilons = [0.01, 0.03, 0.05, 0.10]
    rows = []
    adversarial_scores_max_eps = None

    for eps in epsilons:
        print(f"\nRunning FGSM at epsilon = {eps}...")

        X_adv = fgsm_attack(
            model, X_eval, eps, feature_min, feature_max, immutable_mask
        )

        delta = np.abs(X_adv - X_eval)
        changed = int((delta > 1e-9).sum())
        print(f"   Perturbed values changed: {changed} of {delta.size}")

        adv_preds, adv_errors = predict_from_ae(model, X_adv, threshold)
        adv_acc = float((adv_preds == y_eval).mean())

        clean_correct = (clean_preds == y_eval)
        adv_wrong = (adv_preds != y_eval)
        attack_success = int((clean_correct & adv_wrong).sum())
        attack_success_rate = attack_success / max(1, int(clean_correct.sum()))

        robust_samples = int((adv_preds == y_eval).sum())
        total_samples = len(y_eval)

        rows.append({
            "epsilon": eps,
            "clean_accuracy": round(clean_acc, 6),
            "adversarial_accuracy": round(adv_acc, 6),
            "attack_success_rate": round(attack_success_rate, 6),
            "attack_success_count": attack_success,
            "robust_samples": robust_samples,
            "total_samples": total_samples,
        })

        print(f"   Adversarial accuracy: {adv_acc:.4f}")
        print(f"   Attack success rate:  {attack_success_rate:.4f}")
        print(f"   Robust samples:       {robust_samples}/{total_samples}")

        if eps == epsilons[-1]:
            adversarial_scores_max_eps = np.array(
                [ensemble_score_from_ae(e, threshold) for e in adv_errors]
            )

    out_path = os.path.join(METRICS_DIR, "adversarial_results.csv")
    fieldnames = [
        "epsilon", "clean_accuracy", "adversarial_accuracy",
        "attack_success_rate", "attack_success_count",
        "robust_samples", "total_samples",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nAdversarial results saved to {out_path}")

    print("\nPlotting clean versus adversarial score distributions...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clean_scores, bins=40, alpha=0.6, label="Clean", color="steelblue")
    if adversarial_scores_max_eps is not None:
        ax.hist(
            adversarial_scores_max_eps, bins=40, alpha=0.6,
            label=f"Adversarial (epsilon={epsilons[-1]})", color="crimson"
        )
    ax.set_xlabel("Ensemble score (normalised reconstruction error)")
    ax.set_ylabel("Count")
    ax.set_title("Clean versus adversarial score distribution")
    ax.legend()
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "adversarial_score_distribution.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"   Figure saved to {fig_path}")

    print("\nInterpretation:")
    if all(r["attack_success_rate"] == 0.0 for r in rows):
        print("   Attack success rate is zero at every epsilon.")
        print("   Verify model is in eval mode and perturbations are applied.")
    elif all(
        abs(rows[0]["attack_success_rate"] - r["attack_success_rate"]) < 1e-9
        for r in rows
    ):
        print("   Attack success rate is identical across epsilon.")
        print("   This indicates saturation. Report as not adversarially robust.")
    else:
        print("   Attack success rate increases with epsilon.")
        print("   Report as not adversarially robust under the tested configuration.")

    print("\nAdversarial evaluation complete.")
    print("=" * 60)

    return rows


if __name__ == "__main__":
    run_adversarial_evaluation()