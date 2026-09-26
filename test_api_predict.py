# test_api_predict.py
# Test the AML-Detect API prediction endpoint

import requests
import json
import numpy as np

print("="*60)
print("🔧 AML-Detect API Prediction Test")
print("="*60)

# API endpoint
url = "http://localhost:8000/predict"

# Load a sample from test data
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Test 5 samples
print("\n📊 Testing 5 samples:")
print("-"*60)

for i in range(5):
    features = X_test[i].tolist()
    true_label = "Attack" if y_test[i] == 1 else "Normal"
    
    response = requests.post(url, json={"features": features})
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nSample {i+1}:")
        print(f"   True Label: {true_label}")
        print(f"   Ensemble Score: {result['ensemble_score']:.4f}")
        print(f"   Risk Score: {result['risk_score']:.4f}")
        print(f"   Weights: {[f'{w:.3f}' for w in result['weights']]}")
        print(f"   Component Scores: {[f'{s:.3f}' for s in result['component_scores']]}")
        print(f"   MTD Active: {result['mtd_state']['active'] if result['mtd_state'] else False}")
    else:
        print(f"❌ Sample {i+1}: Error {response.status_code}")

print("\n✅ API prediction test complete!")
print("="*60)