# preprocess.py
# Data loading, validation, chronological partitioning and leakage-controlled scaling.
#
# Inputs  : data/raw/normal.csv, data/raw/attack.csv, data/raw/merged.csv
# Outputs : data/processed/X_train.npy, X_val.npy, X_test.npy
#           data/processed/y_train.npy, y_val.npy, y_test.npy
#           data/processed/scaler_params.json
#           results/metrics/preprocess_log.json

import os
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
METRICS_DIR = 'results/metrics'

NORMAL_CSV = os.path.join(RAW_DIR, 'normal.csv')
ATTACK_CSV = os.path.join(RAW_DIR, 'attack.csv')
MERGED_CSV = os.path.join(RAW_DIR, 'merged.csv')

# Column names in the Kaggle SWaT derivative
TIMESTAMP_COL = 'timestamp'
LABEL_COL = 'Normal/Attack'

# Chronological split boundaries, matching Chapter 4 Table 4.2
TRAIN_END = 965880
VAL_END = TRAIN_END + 133920  # 1,099,800

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sha256_of_file(path, chunk_size=8192):
    """Return the SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def log_step(message):
    print(f"[Preprocess] {message}")


def normalise_label(value):
    """Map raw label values to 0 (Normal) or 1 (Attack)."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip().lower()
    if text in {'normal', '0', 'false', 'benign', 'no', 'n'}:
        return 0
    if text in {'attack', '1', 'true', 'anomaly', 'yes', 'y'}:
        return 1
    raise ValueError(f"Unrecognised label value: {value!r}")


# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------
def load_raw():
    """
    Load the Kaggle SWaT derivative.

    Strategy:
    - Prefer merged.csv when it contains both normal and attack rows.
    - Otherwise concatenate normal.csv and attack.csv, assigning labels.
    - Handle the case where the timestamp is stored as the DataFrame index.
    """
    log_step("Loading raw data...")

    if os.path.exists(MERGED_CSV):
        df = pd.read_csv(MERGED_CSV)
        log_step(f"Merged file loaded: {df.shape}")

        # If the timestamp is stored as the index, promote it to a column
        if TIMESTAMP_COL not in df.columns:
            df = df.reset_index()
            if 'index' in df.columns:
                df = df.rename(columns={'index': TIMESTAMP_COL})
            log_step(f"Promoted index to column '{TIMESTAMP_COL}'.")

        # If the label column is missing, try common alternatives
        if LABEL_COL not in df.columns:
            for alt in ('Label', 'label', 'Attack', 'class', 'Class', 'Target'):
                if alt in df.columns:
                    log_step(f"Using alternative label column: '{alt}'")
                    df = df.rename(columns={alt: LABEL_COL})
                    break

        if LABEL_COL not in df.columns:
            raise ValueError(
                f"Merged file is missing the '{LABEL_COL}' column. "
                f"Available columns: {list(df.columns)}"
            )

        # Normalise label values to {0, 1}
        df[LABEL_COL] = df[LABEL_COL].apply(normalise_label)
        return df

    if os.path.exists(NORMAL_CSV) and os.path.exists(ATTACK_CSV):
        normal_df = pd.read_csv(NORMAL_CSV)
        attack_df = pd.read_csv(ATTACK_CSV)

        normal_df[LABEL_COL] = 0
        attack_df[LABEL_COL] = 1

        df = pd.concat([normal_df, attack_df], ignore_index=True)
        log_step(f"Concatenated normal and attack files: {df.shape}")
        return df

    raise FileNotFoundError(
        "No raw data found. Expected merged.csv or normal.csv + attack.csv "
        f"in {RAW_DIR}."
    )


# ---------------------------------------------------------------------------
# Step 2: Validate
# ---------------------------------------------------------------------------
def validate(df):
    """
    Validate schema, timestamp order, duplicates, missingness and labels.

    Returns a log dictionary.
    """
    log_step("Validating schema and content...")
    log = {}

    # Timestamp
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Missing timestamp column '{TIMESTAMP_COL}'. "
            f"Available columns: {list(df.columns)}"
        )

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors='coerce')
    if df[TIMESTAMP_COL].isna().any():
        raise ValueError("Timestamp parsing produced NaT values.")

    if not df[TIMESTAMP_COL].is_monotonic_increasing:
        log_step("Timestamp not monotonic. Sorting chronologically.")
        df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    log['timestamp_monotonic'] = bool(df[TIMESTAMP_COL].is_monotonic_increasing)

    # Duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log['duplicates_removed'] = before - len(df)
    log_step(f"Duplicates removed: {log['duplicates_removed']}")

    # Label
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Missing label column '{LABEL_COL}'. "
            f"Available columns: {list(df.columns)}"
        )

    unique_labels = set(df[LABEL_COL].unique().tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels are not in {{0, 1}}: {unique_labels}")

    log['label_counts'] = {
        'normal': int((df[LABEL_COL] == 0).sum()),
        'attack': int((df[LABEL_COL] == 1).sum()),
    }

    # Feature columns: numeric only, excluding timestamp and label
    feature_cols = [
        c for c in df.columns
        if c not in {TIMESTAMP_COL, LABEL_COL}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    log['feature_count'] = len(feature_cols)
    log['feature_names'] = feature_cols
    log_step(f"Numeric features retained: {len(feature_cols)}")

    if len(feature_cols) != 51:
        log_step(f"WARNING: expected 51 features, found {len(feature_cols)}")

    # Missing values
    missing = df[feature_cols].isna().sum()
    log['missing_total'] = int(missing.sum())
    log['missing_per_feature'] = {
        c: int(v) for c, v in missing.items() if v > 0
    }
    log_step(f"Total missing values in features: {log['missing_total']}")

    return df, feature_cols, log


# ---------------------------------------------------------------------------
# Step 3: Impute
# ---------------------------------------------------------------------------
def impute(df, feature_cols):
    """
    Replace missing feature values with zero.

    This is a documented limitation. Alternatives (forward fill, interpolation,
    missingness indicators) are noted for future work in Chapter 4 Section 4.3.
    """
    log_step("Imputing missing values with zero...")
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Step 4: Partition chronologically
# ---------------------------------------------------------------------------
def partition(df):
    """
    Split chronologically into 70% train, 15% validation, 15% test.
    Uses the fixed boundaries recorded in Chapter 4 Table 4.2.
    """
    log_step("Partitioning chronologically...")
    n = len(df)
    if n < VAL_END:
        raise ValueError(
            f"Dataset has {n} rows, but partition boundaries expect "
            f"at least {VAL_END}. Adjust TRAIN_END and VAL_END."
        )

    train_df = df.iloc[:TRAIN_END].copy()
    val_df = df.iloc[TRAIN_END:VAL_END].copy()
    test_df = df.iloc[VAL_END:].copy()

    log_step(
        f"Train: {len(train_df)} | "
        f"Val: {len(val_df)} | "
        f"Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Step 5: Scale with leakage control
# ---------------------------------------------------------------------------
def scale(train_df, val_df, test_df, feature_cols):
    """
    Fit StandardScaler on training partition only.
    Apply the same scaler to validation and test.

    Export scaler mean and scale for downstream verification.
    """
    log_step("Fitting StandardScaler on training partition only...")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    # Export scaler parameters
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_names': feature_cols,
        'fitted_on': 'train',
        'n_train_samples': int(len(train_df)),
        'exported_at': datetime.utcnow().isoformat() + 'Z',
    }
    scaler_path = os.path.join(PROCESSED_DIR, 'scaler_params.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    log_step(f"Scaler parameters saved to: {scaler_path}")

    # Verification of training-only fit
    train_mean_max = float(np.abs(X_train.mean(axis=0)).max())
    train_std_max = float(np.abs(X_train.std(axis=0) - 1.0).max())
    log_step(
        f"Train check: max |mean| = {train_mean_max:.6f}, "
        f"max |std - 1| = {train_std_max:.6f}"
    )

    return X_train, X_val, X_test, scaler


# ---------------------------------------------------------------------------
# Step 6: Save
# ---------------------------------------------------------------------------
def save_arrays(X_train, X_val, X_test, y_train, y_val, y_test):
    log_step("Saving processed arrays...")
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)
    log_step("Arrays saved.")


# ---------------------------------------------------------------------------
# Step 7: Provenance record
# ---------------------------------------------------------------------------
def record_provenance(log):
    """Write a preprocessing log for traceability."""
    log['raw_checksums'] = {}
    for path in (NORMAL_CSV, ATTACK_CSV, MERGED_CSV):
        if os.path.exists(path):
            log['raw_checksums'][os.path.basename(path)] = sha256_of_file(path)

    log['processed_at'] = datetime.utcnow().isoformat() + 'Z'
    log['seed'] = SEED

    out_path = os.path.join(METRICS_DIR, 'preprocess_log.json')
    with open(out_path, 'w') as f:
        json.dump(log, f, indent=2)
    log_step(f"Provenance log saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_raw()
    print(f"   Raw shape: {df.shape}")

    # 2. Validate
    df, feature_cols, log = validate(df)
    print(f"   Validated shape: {df.shape}")

    # 3. Impute
    df = impute(df, feature_cols)

    # 4. Partition
    train_df, val_df, test_df = partition(df)

    # 5. Scale
    X_train, X_val, X_test, scaler = scale(
        train_df, val_df, test_df, feature_cols
    )

    y_train = train_df[LABEL_COL].values.astype(np.int64)
    y_val = val_df[LABEL_COL].values.astype(np.int64)
    y_test = test_df[LABEL_COL].values.astype(np.int64)

    # 6. Save
    save_arrays(X_train, X_val, X_test, y_train, y_val, y_test)

    # Partition summary
    log['partitions'] = {
        'train': {
            'n': int(len(y_train)),
            'normal': int((y_train == 0).sum()),
            'attack': int((y_train == 1).sum()),
        },
        'val': {
            'n': int(len(y_val)),
            'normal': int((y_val == 0).sum()),
            'attack': int((y_val == 1).sum()),
        },
        'test': {
            'n': int(len(y_test)),
            'normal': int((y_test == 0).sum()),
            'attack': int((y_test == 1).sum()),
        },
    }
    print("[Preprocess] Partition summary:")
    for split, counts in log['partitions'].items():
        print(f"   {split:5s}: n={counts['n']:>7d}, "
              f"normal={counts['normal']:>7d}, attack={counts['attack']:>6d}")

    # 7. Provenance
    record_provenance(log)

    print("=" * 60)
    print("Preprocessing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()