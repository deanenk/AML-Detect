# ocsvm.py
# One-Class SVM using SGD (Stochastic Gradient Descent) for large-scale data
# Fast alternative to standard OneClassSVM which is O(n^2) complexity
#
# Normalisation method:
#   Raw decision_function scores are mapped to [0, 1] using min-max scaling
#   fitted on the training partition:
#       score_norm = (score_raw - min_train) / (max_train - min_train)
#   where min_train and max_train are the min and max decision-function
#   outputs observed on the training set for the selected nu.
#   Scores below min_train are clipped to 0; scores above max_train are
#   clipped to 1. This preserves the ranking of raw scores but rescales them
#   into a bounded interval for ensemble weighting.

import numpy as np
from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import OneClassSVM
import joblib
import os
import time
import csv

def train_ocsvm_fast(X_train, nu=0.05, max_iter=1000, seed=42, verbose=1):
    """
    Train fast One-Class SVM using SGD for large datasets.

    SGDOneClassSVM scales linearly with the number of samples (O(n)),
    making it suitable for datasets with ~1 million samples.

    Args:
        X_train: Training data (n_samples, n_features)
        nu: Upper bound on fraction of training errors
        max_iter: Maximum number of iterations
        seed: Random seed for reproducibility
        verbose: Verbosity level passed to SGDOneClassSVM
    """
    print("Training Fast One-Class SVM (SGD)...")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Nu (outlier fraction): {nu}")
    print(f"   Max iterations: {max_iter}")

    start_time = time.time()

    np.random.seed(seed)

    ocsvm = SGDOneClassSVM(
        nu=nu,
        max_iter=max_iter,
        random_state=seed,
        shuffle=True,
        tol=1e-4,
        learning_rate='optimal',
        verbose=verbose
    )

    ocsvm.fit(X_train)

    training_time = time.time() - start_time
    print(f"   Training completed in {training_time:.2f} seconds")

    train_scores = ocsvm.decision_function(X_train)
    print(f"   Training score range: [{train_scores.min():.4f}, {train_scores.max():.4f}]")
    print(f"   Training score mean: {train_scores.mean():.4f}")
    print(f"   Training score std: {train_scores.std():.4f}")

    return ocsvm, training_time, train_scores

def normalise_scores(scores, min_train, max_train):
    """Min-max normalise raw decision-function scores to [0, 1]."""
    if max_train - min_train == 0:
        return np.zeros_like(scores)
    norm = (scores - min_train) / (max_train - min_train)
    return np.clip(norm, 0.0, 1.0)

def run_nu_sensitivity(X_train, nu_values=(0.01, 0.05, 0.1, 0.2),
                       max_iter=1000, seed=42, test_size=100):
    """Run nu sensitivity analysis and log results to CSV."""
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    log_path = "results/metrics/ocsvm_nu_sensitivity.csv"
    rows = []

    for nu in nu_values:
        print("\n" + "-" * 40)
        print(f"Nu sensitivity run: nu = {nu}")
        print("-" * 40)

        model, t_time, train_scores = train_ocsvm_fast(
            X_train, nu=nu, max_iter=max_iter, seed=seed, verbose=0
        )

        min_train = float(train_scores.min())
        max_train = float(train_scores.max())
        mean_train = float(train_scores.mean())
        std_train = float(train_scores.std())

        # Quick test on held-out sample (first test_size rows)
        sample = X_train[:test_size]
        raw_scores = model.decision_function(sample)
        preds = model.predict(sample)
        n_anom = int(np.sum(preds == -1))
        pct_anom = n_anom / test_size * 100.0

        # Normalised scores
        norm_scores = normalise_scores(raw_scores, min_train, max_train)

        print(f"   Training time: {t_time:.2f} s")
        print(f"   Train score range: [{min_train:.4f}, {max_train:.4f}]")
        print(f"   Train score mean: {mean_train:.4f}")
        print(f"   Train score std: {std_train:.4f}")
        print(f"   Test anomalies: {n_anom}/{test_size} ({pct_anom:.1f}%)")
        print(f"   Normalised score range: [{norm_scores.min():.4f}, {norm_scores.max():.4f}]")

        rows.append({
            "nu": nu,
            "training_time_s": round(t_time, 4),
            "train_score_min": round(min_train, 6),
            "train_score_max": round(max_train, 6),
            "train_score_mean": round(mean_train, 6),
            "train_score_std": round(std_train, 6),
            "test_anomalies": n_anom,
            "test_anomaly_pct": round(pct_anom, 2),
            "norm_score_min": round(float(norm_scores.min()), 6),
            "norm_score_max": round(float(norm_scores.max()), 6),
        })

        # Save each model separately for traceability
        joblib.dump(model, f"results/models/ocsvm_nu_{nu}.pkl")

    # Write CSV
    fieldnames = [
        "nu", "training_time_s",
        "train_score_min", "train_score_max", "train_score_mean", "train_score_std",
        "test_anomalies", "test_anomaly_pct",
        "norm_score_min", "norm_score_max",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nNu sensitivity log saved to: {log_path}")
    return rows

def train_ocsvm_subsample(X_train, nu=0.05, sample_size=50000, seed=42):
    """Alternative: train standard OneClassSVM on a subsample."""
    print("Training Subsampled One-Class SVM...")
    print(f"   Full dataset: {X_train.shape[0]:,} samples")
    print(f"   Subsample: {sample_size:,} samples")

    np.random.seed(seed)
    indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    X_subsample = X_train[indices]

    start_time = time.time()
    ocsvm = OneClassSVM(nu=nu, gamma='scale')
    ocsvm.fit(X_subsample)
    training_time = time.time() - start_time

    print(f"   Training completed in {training_time:.2f} seconds")
    os.makedirs("results/models", exist_ok=True)
    joblib.dump(ocsvm, 'results/models/ocsvm.pkl')
    print(f"   Model saved to: results/models/ocsvm.pkl")

    return ocsvm, training_time

def predict_ocsvm(model, X):
    """Returns 1 for normal, -1 for anomaly."""
    return model.predict(X)

def anomaly_score_ocsvm(model, X):
    """Decision-function score. Higher = more normal."""
    return model.decision_function(X)

def quick_test(model, X, num_samples=100, min_train=None, max_train=None):
    """Quick test on a small sample."""
    sample = X[:num_samples]
    scores = anomaly_score_ocsvm(model, sample)
    predictions = predict_ocsvm(model, sample)

    print(f"\n   Quick test on {num_samples} samples:")
    print(f"   Anomaly scores range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   Number of anomalies detected: {np.sum(predictions == -1)}")
    print(f"   Percentage anomalies: {np.sum(predictions == -1) / num_samples * 100:.1f}%")

    if min_train is not None and max_train is not None:
        norm = normalise_scores(scores, min_train, max_train)
        print(f"   Normalised score range: [{norm.min():.4f}, {norm.max():.4f}]")

    return scores, predictions

if __name__ == "__main__":
    print("=" * 60)
    print("One-Class SVM Training (Fast Version)")
    print("=" * 60)

    print("Loading training data...")
    X_train = np.load('data/processed/X_train.npy')
    print(f"   X_train shape: {X_train.shape}")
    print(f"   Total samples: {X_train.shape[0]:,}")
    print(f"   Features: {X_train.shape[1]}")

    # Nu sensitivity analysis over 0.01, 0.05, 0.1, 0.2
    rows = run_nu_sensitivity(
        X_train,
        nu_values=(0.01, 0.05, 0.1, 0.2),
        max_iter=1000,
        seed=42,
        test_size=100
    )

    # Select the default nu = 0.05 model for the ensemble
    model_default, time_default, train_scores_default = train_ocsvm_fast(
        X_train, nu=0.05, max_iter=1000, seed=42, verbose=1
    )
    min_train = float(train_scores_default.min())
    max_train = float(train_scores_default.max())

    joblib.dump(model_default, 'results/models/ocsvm.pkl')
    print(f"   Default model (nu=0.05) saved to: results/models/ocsvm.pkl")

    quick_test(model_default, X_train, num_samples=100,
               min_train=min_train, max_train=max_train)

    print("\nOCSVM training complete!")
    print("=" * 60)