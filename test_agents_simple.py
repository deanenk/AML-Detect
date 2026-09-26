# test_agents_simple.py
# Test agent workflow without XMPP (simulated)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from ensemble.ensemble import EnsembleDetector
from mtd_coordinator.mtd import MTDCoordinator

print("="*60)
print("🤖 Simulated Multi-Agent System Test (No XMPP)")
print("="*60)

# Load ensemble
print("\n📂 Loading ensemble...")
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')
ensemble = EnsembleDetector(X_test.shape[1], enable_mtd=True)

# Initialize MTD
mtd = MTDCoordinator()

print("\n📊 Simulating agent workflow:")
print("-"*60)

# Agent states
alert_count = 0
escalation_count = 0
active_threshold = ensemble.threshold
risk_threshold = 0.7

# Process 50 samples
for i in range(min(50, len(X_test))):
    # Reactive agent detects
    ensemble_score, risk, weights, scores, risk_components, mtd_state = ensemble.predict_with_mtd(X_test[i])
    
    if ensemble_score > 0.6:
        alert_count += 1
        print(f"\n🔔 Alert #{alert_count}: Score={ensemble_score:.3f}, Risk={risk:.3f}")
        print(f"   True Label: {'Attack' if y_test[i] == 1 else 'Normal'}")
        
        # Supervisor checks risk
        if risk > risk_threshold:
            escalation_count += 1
            print(f"   ⚠️ Escalation #{escalation_count}: Risk={risk:.3f}")
            
            # Request deep analysis (simulated)
            print(f"   🧠 Deep analysis requested...")
            
            # MTD adapts
            current_state = {'threshold': ensemble.threshold, 'weights': ensemble.weights}
            adaptation = mtd.adapt(risk, current_state)
            if adaptation and adaptation.get('action') == 'increase_threshold':
                ensemble.threshold = adaptation['new_threshold']
                print(f"   ⚡ MTD: Threshold increased to {ensemble.threshold:.4f}")
            elif adaptation and adaptation.get('action') == 'rollback':
                ensemble.threshold = mtd.state['threshold']
                print(f"   ↩️ MTD: Threshold restored to {ensemble.threshold:.4f}")

print("\n" + "-"*60)
print("📊 Summary:")
print(f"   Total samples processed: 50")
print(f"   Alerts: {alert_count}")
print(f"   Escalations: {escalation_count}")
print(f"   Final MTD State: {mtd.get_state()}")
print(f"   Final Ensemble Threshold: {ensemble.threshold:.4f}")

print("\n✅ Agent workflow simulation complete!")
print("="*60)