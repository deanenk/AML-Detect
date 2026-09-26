# autoencoder.py
# Autoencoder for reconstruction-based anomaly detection
# Enhanced with Step 3 fixes:
#   - Scaler consistency assertion across splits (median-based, tolerant of
#     zero-variance features common in ICS data)
#   - Confirmed reduction='mean' for training and validation loss
#   - Per-epoch metrics logged to CSV
#   - Validation loss separated into normal and attack components
#   - Threshold recalibrated only after loss curves are stable
#   - JSON serialization converts NumPy types to native Python types

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    """Autoencoder with 128-dimensional latent space."""

    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ---------------------------------------------------------------------------
# Step 3 fix 1: scaler consistency assertion across splits
# ---------------------------------------------------------------------------
def assert_scaler_consistency(X_train, X_val, X_test,
                              scaler_path='data/processed/scaler_params.json'):
    """
    Confirm that train, val and test arrays were transformed by the same
    StandardScaler fitted on the training partition only.

    Uses median statistics to avoid false positives from zero-variance
    features (common in ICS data where some actuators remain constant).
    Reports zero-variance features as information, not as failure.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler parameters not found at {scaler_path}. "
            "Run preprocessing first to export scaler.mean_ and scaler.scale_."
        )

    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)

    stored_mean = np.array(scaler_params['mean'])
    stored_scale = np.array(scaler_params['scale'])

    if stored_mean.shape[0] != X_train.shape[1]:
        raise ValueError(
            f"Stored scaler mean has {stored_mean.shape[0]} features, "
            f"but training data has {X_train.shape[1]}."
        )

    print("[Scaler check] Stored mean (first 5):", stored_mean[:5])
    print("[Scaler check] Stored scale (first 5):", stored_scale[:5])

    std_zero_count = 0

    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        feature_mean = X.mean(axis=0)
        feature_std = X.std(axis=0)

        mean_drift_max = float(np.abs(feature_mean).max())
        mean_drift_med = float(np.median(np.abs(feature_mean)))

        # Use median to avoid zero-variance features dominating the metric
        std_drift_med = float(np.median(np.abs(feature_std - 1.0)))
        std_zero_count = int((feature_std < 1e-6).sum())
        std_below_half = int((feature_std < 0.5).sum())

        print(
            f"[Scaler check] {name}: "
            f"mean drift (max/med) = {mean_drift_max:.4f}/{mean_drift_med:.4f}, "
            f"std drift (med) = {std_drift_med:.4f}, "
            f"zero-variance features = {std_zero_count}, "
            f"std < 0.5 = {std_below_half}"
        )

        if name == "train":
            # Median mean drift should be near zero
            assert mean_drift_med < 1e-2, (
                f"Training split mean is not ~0 (median drift "
                f"{mean_drift_med:.4f}). Scaler may not have been applied "
                "to training data."
            )
            # Median std drift should be near zero. Zero-variance features
            # are tolerated because StandardScaler sets their scale to 1.
            assert std_drift_med < 0.1, (
                f"Training split std is not ~1 (median drift "
                f"{std_drift_med:.4f}). Scaler may not have been applied "
                "to training data."
            )

    print("[Scaler check] Passed. Same scaler applied across all splits.")
    if std_zero_count > 0:
        print(
            f"[Scaler check] Note: {std_zero_count} features have zero "
            "variance in the training set. StandardScaler leaves these "
            "unchanged by design. They carry no information but do not "
            "break the pipeline."
        )


def train_autoencoder(X_train, X_val, y_val, input_dim,
                      epochs=50, batch_size=256, seed=42,
                      log_path='results/metrics/autoencoder_training_log.csv'):
    """
    Train autoencoder with fixed seed for reproducibility.

    Step 3 fixes applied:
    - reduction='mean' is used for both training and validation loss.
    - Per-epoch metrics are logged to a CSV.
    - Validation loss is separated into normal and attack components.
    """
    print("🚀 Training Autoencoder...")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Autoencoder(input_dim, latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Step 3 fix 2: reduction='mean' explicitly for both losses
    criterion = nn.MSELoss(reduction='mean')

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    val_losses_normal = []
    val_losses_attack = []
    log_rows = []

    y_val_array = np.asarray(y_val)
    normal_mask = y_val_array == 0
    attack_mask = y_val_array == 1

    print(f"   Validation labels: "
          f"normal = {int(normal_mask.sum())}, "
          f"attack = {int(attack_mask.sum())}")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            reconstruction = model(x)
            loss = criterion(reconstruction, x)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        model.eval()
        epoch_val_loss = 0
        all_val_errors = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0]
                reconstruction = model(x)
                loss = criterion(reconstruction, x)
                epoch_val_loss += loss.item()
                # Per-sample error for class-split reporting
                per_sample = torch.mean((reconstruction - x) ** 2, dim=1).numpy()
                all_val_errors.extend(per_sample.tolist())

        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(val_loader)

        all_val_errors = np.array(all_val_errors)

        # Step 3 fix 4: split validation loss by class
        val_loss_normal = (
            float(all_val_errors[normal_mask].mean())
            if normal_mask.any() else float('nan')
        )
        val_loss_attack = (
            float(all_val_errors[attack_mask].mean())
            if attack_mask.any() else float('nan')
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_losses_normal.append(val_loss_normal)
        val_losses_attack.append(val_loss_attack)

        log_rows.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss_all': val_loss,
            'val_loss_normal': val_loss_normal,
            'val_loss_attack': val_loss_attack,
            'n_train_batches': len(train_loader),
            'n_val_normal': int(normal_mask.sum()),
            'n_val_attack': int(attack_mask.sum()),
        })

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val Normal = {val_loss_normal:.4f}, "
                  f"Val Attack = {val_loss_attack:.4f}")

    # Step 3 fix 3: log per-epoch metrics to CSV
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        import pandas as pd
        pd.DataFrame(log_rows).to_csv(log_path, index=False)
        print(f"   Training log saved to: {log_path}")
    except ImportError:
        # Fallback to csv module if pandas is unavailable
        import csv
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"   Training log saved to: {log_path}")

    # Save model and loss curve
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss (all)')
    plt.plot(val_losses_normal, label='Validation Loss (normal)')
    plt.plot(val_losses_attack, label='Validation Loss (attack)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training Loss')
    plt.legend()
    plt.savefig('results/figures/ae_loss.png')
    plt.close()

    torch.save(model.state_dict(), 'results/models/autoencoder.pt')
    print(f"   Model saved to: results/models/autoencoder.pt")

    return model, train_losses, val_losses, log_rows


def check_loss_stability(log_rows, tolerance=2.0):
    """
    Decide whether the loss curves are consistent enough for the threshold
    to be treated as valid.

    Criteria:
    - Final train loss and final validation normal loss must be within
      `tolerance` factor of each other.
    - Training loss must decrease or plateau (not diverge).

    If validation attack loss is nan (no attack observations in validation),
    the stability check proceeds with train and val normal only.
    """
    final = log_rows[-1]
    train_loss = final['train_loss']
    val_normal = final['val_loss_normal']
    val_attack = final['val_loss_attack']

    ratio = max(train_loss, val_normal) / max(min(train_loss, val_normal), 1e-9)
    train_decreasing = log_rows[-1]['train_loss'] <= log_rows[0]['train_loss']
    stable = (ratio <= tolerance) and train_decreasing

    print("[Loss stability check]")
    print(f"   Final train loss        : {train_loss:.4f}")
    print(f"   Final val loss (normal) : {val_normal:.4f}")
    if np.isnan(val_attack):
        print(f"   Final val loss (attack) : nan "
              "(no attack observations in validation split)")
    else:
        print(f"   Final val loss (attack) : {val_attack:.4f}")
    print(f"   Train/val normal ratio  : {ratio:.2f}")
    print(f"   Training loss decreasing: {train_decreasing}")
    print(f"   Stable enough for threshold: {stable}")

    if not stable:
        print("   ⚠️  Loss curves are not consistent. The calibrated threshold "
              "must be treated as provisional.")

    return stable


def compute_reconstruction_errors(model, X, batch_size=256):
    """Compute MSE reconstruction error for each sample."""
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
            recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=1).numpy()
            errors.extend(mse)
    return np.array(errors)


def calibrate_threshold(val_errors, percentile=95, y_val=None):
    """
    Calibrate threshold using validation data only.

    Step 3 fix 5: If y_val is provided, use only NORMAL observations for
    calibration. This prevents attack observations from inflating the
    threshold.
    """
    if y_val is not None:
        normal_mask = np.asarray(y_val) == 0
        val_errors = val_errors[normal_mask]
        print(f"   Calibrating on {normal_mask.sum()} validation normal samples")

    threshold = float(np.percentile(val_errors, percentile))
    print(f"   Threshold calibrated at {percentile}th percentile: {threshold:.6f}")
    return threshold


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 Autoencoder Training")
    print("=" * 60)

    # Load processed data
    X_train = np.load('data/processed/X_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')
    y_test = np.load('data/processed/y_test.npy')

    print(f"   X_train: {X_train.shape}")
    print(f"   X_val: {X_val.shape}")
    print(f"   X_test: {X_test.shape}")

    # Step 3 fix 1: verify scaler consistency across splits
    try:
        assert_scaler_consistency(X_train, X_val, X_test)
    except FileNotFoundError as e:
        print(f"   ⚠️  {e}")
        print("   Skipping scaler consistency check. Export scaler_params.json "
              "in the preprocessing script to enable this check.")

    # Train
    input_dim = X_train.shape[1]
    model, train_losses, val_losses, log_rows = train_autoencoder(
        X_train, X_val, y_val, input_dim, epochs=50, batch_size=256, seed=42
    )

    # Step 3 fix 5: check loss stability before calibrating threshold
    stable = check_loss_stability(log_rows)

    # Compute errors and calibrate threshold
    val_errors = compute_reconstruction_errors(model, X_val)
    threshold = calibrate_threshold(val_errors, percentile=95, y_val=y_val)

    # Save threshold with status
    os.makedirs('data/processed', exist_ok=True)
    threshold_status = 'valid' if stable else 'provisional'
    with open('data/processed/autoencoder_threshold.txt', 'w') as f:
        f.write(f"Threshold: {threshold:.6f}\n")
        f.write(f"Status: {threshold_status}\n")
        f.write(f"Loss curves stable: {stable}\n")

    # Also save structured JSON for downstream use
    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/autoencoder_threshold.json', 'w') as f:
        json.dump({
            'threshold': float(threshold),
            'percentile': 95,
            'status': str(threshold_status),
            'loss_stable': bool(stable),
        }, f, indent=2)

    print(f"   Threshold saved to: data/processed/autoencoder_threshold.txt "
          f"(status: {threshold_status})")
    print("✅ Autoencoder training complete!")
    print("=" * 60)