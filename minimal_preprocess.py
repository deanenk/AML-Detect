print("🚀 Script started!")
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

print("✅ All imports successful!")

raw_path = "data/raw/"
print(f"Loading from: {raw_path}")

try:
    normal_df = pd.read_csv(os.path.join(raw_path, "normal.csv"))
    print(f"✅ Normal loaded: {normal_df.shape}")
except Exception as e:
    print(f"❌ Error loading normal: {e}")
    exit(1)

try:
    attack_df = pd.read_csv(os.path.join(raw_path, "attack.csv"))
    print(f"✅ Attack loaded: {attack_df.shape}")
except Exception as e:
    print(f"❌ Error loading attack: {e}")
    exit(1)

print("Finding timestamp column...")
timestamp_col = None
for col in normal_df.columns:
    col_clean = col.strip()
    if col_clean.lower() in ['timestamp', 'time', 'date', 'datetime']:
        timestamp_col = col
        print(f"✅ Found timestamp column: '{timestamp_col}'")
        break

if timestamp_col is None:
    print("❌ No timestamp column found!")
    print(f"Columns: {normal_df.columns.tolist()[:10]}")
    exit(1)

# Combine data
print("Combining data...")
normal_df['label'] = 0
attack_df['label'] = 1
combined_df = pd.concat([normal_df, attack_df], ignore_index=True)
print(f"✅ Combined: {combined_df.shape}")

print("Sorting by timestamp...")
if timestamp_col != timestamp_col.strip():
    clean_name = timestamp_col.strip()
    combined_df.rename(columns={timestamp_col: clean_name}, inplace=True)
    timestamp_col = clean_name

# Parse dates - FIRST strip whitespace, THEN parse
print("Parsing dates...")
combined_df[timestamp_col] = combined_df[timestamp_col].str.strip()
combined_df[timestamp_col] = pd.to_datetime(
    combined_df[timestamp_col], 
    format='%d/%m/%Y %I:%M:%S %p',
    dayfirst=True
)

combined_df = combined_df.sort_values(timestamp_col).reset_index(drop=True)
print(f"✅ Sorted! Date range: {combined_df[timestamp_col].min()} to {combined_df[timestamp_col].max()}")

# Split chronologically
print("Splitting data...")
min_date = combined_df[timestamp_col].min()
train_cutoff = min_date + pd.Timedelta(days=7)

train_df = combined_df[combined_df[timestamp_col] < train_cutoff].copy()
test_df = combined_df[combined_df[timestamp_col] >= train_cutoff].copy()

# Validation split from train
val_cutoff = train_df[timestamp_col].min() + pd.Timedelta(days=5.6)
val_df = train_df[train_df[timestamp_col] >= val_cutoff].copy()
train_df = train_df[train_df[timestamp_col] < val_cutoff].copy()

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Separate features and labels - KEEP ONLY NUMERIC
exclude_cols = ['label', timestamp_col]
feature_cols = [col for col in combined_df.columns if col not in exclude_cols]
print(f"Features before cleaning: {len(feature_cols)}")

# Keep only numeric columns
numeric_cols = []
for col in feature_cols:
    try:
        # Test if column can be converted to numeric
        pd.to_numeric(combined_df[col].iloc[:100], errors='raise')
        numeric_cols.append(col)
    except:
        print(f"   Skipping non-numeric column: '{col}'")
        continue

feature_cols = numeric_cols
print(f"Features after cleaning: {len(feature_cols)}")

# Extract features and labels
X_train = train_df[feature_cols].values.astype(float)
y_train = train_df['label'].values
X_val = val_df[feature_cols].values.astype(float)
y_val = val_df['label'].values
X_test = test_df[feature_cols].values.astype(float)
y_test = test_df['label'].values

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Check for any remaining NaN or infinite values
print("Checking for NaN/Inf values...")
print(f"  X_train NaN count: {np.isnan(X_train).sum()}")
print(f"  X_val NaN count: {np.isnan(X_val).sum()}")
print(f"  X_test NaN count: {np.isnan(X_test).sum()}")

# Handle any NaN values (replace with 0)
if np.isnan(X_train).sum() > 0:
    print("  Replacing NaN with 0...")
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"X_train_scaled: {X_train_scaled.shape}, y_train: {y_train.shape}")
print(f"X_val_scaled: {X_val_scaled.shape}, y_val: {y_val.shape}")
print(f"X_test_scaled: {X_test_scaled.shape}, y_test: {y_test.shape}")

# Save
print("Saving processed data...")
os.makedirs("data/processed/", exist_ok=True)
np.save("data/processed/X_train.npy", X_train_scaled)
np.save("data/processed/X_val.npy", X_val_scaled)
np.save("data/processed/X_test.npy", X_test_scaled)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_val.npy", y_val)
np.save("data/processed/y_test.npy", y_test)

with open("data/processed/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

# Save feature columns
with open("data/processed/feature_columns.txt", 'w') as f:
    for col in feature_cols:
        f.write(f"{col}\n")

# Save split info
with open("data/processed/split_info.txt", 'w') as f:
    f.write("SWaT Dataset Chronological Split\n")
    f.write("="*50 + "\n")
    f.write(f"Train: {len(train_df)} samples\n")
    f.write(f"  Normal: {sum(y_train == 0)}, Attack: {sum(y_train == 1)}\n")
    f.write(f"Validation: {len(val_df)} samples\n")
    f.write(f"  Normal: {sum(y_val == 0)}, Attack: {sum(y_val == 1)}\n")
    f.write(f"Test: {len(test_df)} samples\n")
    f.write(f"  Normal: {sum(y_test == 0)}, Attack: {sum(y_test == 1)}\n")
    f.write(f"Features: {len(feature_cols)}\n")
    if timestamp_col:
        f.write(f"Timestamp column: {timestamp_col}\n")

print("✅ Done!")
print("="*50)
print("Processed files saved to: data/processed/")