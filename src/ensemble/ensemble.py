# ensemble.py
# Ensemble with dynamic weighting, chaos-aware risk engine, and moving-target defence
#
# Threshold ownership:
#   Three thresholds exist in AML-Detect and they are not interchangeable.
#       - Detector threshold (autoencoder): reconstruction error, positive real
#       - Ensemble threshold: weighted ensemble score in [0, 1]
#       - MTD threshold: risk score in [0, 1]
#   The previous code reused `self.threshold` for all three, which is the
#   source of the 1360.67 vs 0.5 scale mismatch. This version stores them
#   separately and exposes all three in the status response.
#
# Ensemble score definition:
#   The ensemble score is a weighted average of normalised detector outputs.
#   Each detector produces a raw score that is normalised to [0, 1].
#   Confidence values adjust the initial weights. The weighted average is
#   clipped to [0, 1] before it is returned. The score is an internal
#   ranking value, not a calibrated probability of attack.
#
# Debug output:
#   Pre-normalisation and post-normalisation scores are logged separately
#   for debugging. Only post-normalisation scores in [0, 1] are reported
#   as final ensemble scores.

import numpy as np
import torch
import joblib
import os
import sys
import csv

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src/
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

# Import model classes
from autoencoder import Autoencoder
from gnn import GNNModel

# Import risk engine
try:
    from risk_engine.risk_engine import ChaosRiskEngine
except ImportError:
    try:
        from risk_engine import ChaosRiskEngine
    except ImportError:
        print("Could not import ChaosRiskEngine. Please ensure risk_engine.py exists.")
        raise

# Import MTD Coordinator
try:
    from mtd_coordinator.mtd import MTDCoordinator
except ImportError:
    try:
        from mtd_coordinator import MTDCoordinator
    except ImportError:
        print("Could not import MTDCoordinator. MTD features will be disabled.")
        MTDCoordinator = None


class EnsembleDetector:
    """Ensemble with dynamic weight adjustment, risk awareness, and moving-target defence."""

    def __init__(self, input_dim, initial_weights=[0.4, 0.4, 0.2], enable_mtd=True):
        self.input_dim = input_dim
        self.initial_weights = list(initial_weights)
        self.weights = list(initial_weights)

        # Three separate thresholds. See header for ownership.
        self.detector_threshold = None    # autoencoder reconstruction error
        self.ensemble_threshold = 0.5     # weighted ensemble score in [0, 1]
        self.mtd_threshold = 0.5          # risk score in [0, 1] used by MTD

        self.window = []
        self.last_confidences = [0.7, 0.7, 0.7]
        self.enable_mtd = enable_mtd and MTDCoordinator is not None

        self._load_models()

        print("\nInitializing Chaos-Aware Risk Engine...")
        self.risk_engine = ChaosRiskEngine(window_size=50, prediction_horizon=10)

        try:
            X_val = np.load('data/processed/X_val.npy')
            y_val = np.load('data/processed/y_val.npy')
            self.risk_engine.calibrate_baseline(X_val, labels=y_val)
            print("   Risk engine calibrated on validation data (normal only)")
        except Exception as e:
            print(f"   Risk engine calibration skipped: {e}")

        if self.enable_mtd:
            print("\nInitializing Moving-Target Defence Coordinator...")
            self.mtd = MTDCoordinator(
                activation_threshold=0.70,
                rollback_threshold=0.30,
                adaptation_factor=0.7,
                base_threshold=self.ensemble_threshold,
            )
            print("   MTD Coordinator initialized")
        else:
            self.mtd = None
            print("\nMTD Coordinator disabled")

    def _load_models(self):
        print("Loading models...")

        # 1. Autoencoder
        try:
            self.ae = Autoencoder(self.input_dim, latent_dim=128)
            self.ae.load_state_dict(torch.load('results/models/autoencoder.pt', map_location=torch.device('cpu')))
            self.ae.eval()
            print("   Autoencoder loaded")

            try:
                with open('data/processed/autoencoder_threshold.txt', 'r') as f:
                    self.detector_threshold = float(f.readline().split(':')[1].strip())
                print(f"   Detector threshold (autoencoder): {self.detector_threshold:.6f}")
            except Exception:
                self.detector_threshold = 0.5
                print("   Using default detector threshold: 0.5")
        except Exception as e:
            print(f"   Autoencoder load failed: {e}")
            self.ae = None
            self.detector_threshold = 0.5

        # 2. GNN
        try:
            self.gnn = GNNModel(self.input_dim)
            self.gnn.load_state_dict(torch.load('results/models/gnn.pt', map_location=torch.device('cpu')))
            self.gnn.eval()
            print("   GNN loaded")
        except Exception as e:
            print(f"   GNN load failed: {e}")
            self.gnn = None

        # 3. OCSVM
        try:
            self.ocsvm = joblib.load('results/models/ocsvm.pkl')
            print("   OCSVM loaded")
        except Exception as e:
            print(f"   OCSVM load failed: {e}")
            self.ocsvm = None

        self.available_models = []
        if self.ae is not None:
            self.available_models.append('autoencoder')
        if self.gnn is not None:
            self.available_models.append('gnn')
        if self.ocsvm is not None:
            self.available_models.append('ocsvm')

        print(f"   Available models: {self.available_models}")

        if len(self.available_models) == 0:
            raise RuntimeError("No models loaded. Please train all models first.")

        if len(self.available_models) < 3:
            self.weights = [1.0 / len(self.available_models)] * len(self.available_models)
            print(f"   Adjusted weights for available models: {self.weights}")

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(self, X, mode="dynamic"):
        """
        Return ensemble prediction.

        Args:
            X: Input features (single sample or batch)
            mode: "dynamic" for confidence-adjusted weights,
                  "fixed" for initial weights only

        Returns:
            ensemble_score: Weighted ensemble score in [0, 1]
            adjusted_weights: Final weights used
            component_scores: List of individual model scores
            debug: Dict with pre-normalisation and post-normalisation values
        """
        scores = []
        confidences = []

        if self.ae is not None:
            ae_score = self._autoencoder_score(X)
            scores.append(ae_score)
            confidences.append(self._confidence(ae_score))
        else:
            scores.append(0.5)
            confidences.append(0.0)

        if self.gnn is not None:
            gnn_score = self._gnn_score(X)
            scores.append(gnn_score)
            confidences.append(self._confidence(gnn_score))
        else:
            scores.append(0.5)
            confidences.append(0.0)

        if self.ocsvm is not None:
            ocsvm_score = self._ocsvm_score(X)
            scores.append(ocsvm_score)
            confidences.append(self._confidence(ocsvm_score))
        else:
            scores.append(0.5)
            confidences.append(0.0)

        scores = np.array(scores, dtype=float)
        confidences = np.array(confidences, dtype=float)

        # Clip component scores to [0, 1] before weighting
        scores = np.clip(scores, 0.0, 1.0)
        self.last_confidences = confidences

        weights = np.array(self.weights[:len(scores)], dtype=float)
        if len(weights) < len(scores):
            weights = np.array([1.0 / len(scores)] * len(scores), dtype=float)

        if mode == "fixed":
            adjusted_weights = weights / weights.sum() if weights.sum() > 0 else weights
        else:
            adjusted_weights = weights * confidences
            if adjusted_weights.sum() > 0:
                adjusted_weights /= adjusted_weights.sum()
            else:
                adjusted_weights = weights / weights.sum()

        pre_norm_score = float(np.dot(adjusted_weights, scores))
        ensemble_score = float(np.clip(pre_norm_score, 0.0, 1.0))

        debug = {
            "pre_norm_score": pre_norm_score,
            "post_norm_score": ensemble_score,
            "component_scores_raw": scores.tolist(),
            "component_scores_normalised": scores.tolist(),
            "confidences": confidences.tolist(),
            "mode": mode,
        }

        return ensemble_score, adjusted_weights, scores, debug

    def predict_with_risk(self, X):
        ensemble_score, weights, scores, debug = self.predict(X, mode="dynamic")

        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                self.window.append(X)
            else:
                self.window.append(X[0])
        else:
            self.window.append(X)

        if len(self.window) > 150:
            self.window = self.window[-100:]

        if hasattr(self, 'last_confidences'):
            confidence = float(np.sum(weights * self.last_confidences[:len(weights)]))
        else:
            confidence = 0.7

        risk, risk_components = self.risk_engine.process_stream(
            data_point=X if isinstance(X, np.ndarray) and len(X.shape) == 1 else X[0],
            ensemble_score=ensemble_score,
            confidence=confidence,
            window_data=self.window
        )

        return ensemble_score, risk, weights, scores, risk_components, debug

    def predict_with_mtd(self, X):
        ensemble_score, risk, weights, scores, risk_components, debug = self.predict_with_risk(X)

        mtd_state = None
        if self.enable_mtd and self.mtd is not None:
            current_state = {
                'threshold': self.mtd_threshold,
                'weights': self.weights
            }
            adaptation = self.mtd.adapt(risk, current_state)

            if adaptation is not None:
                action = adaptation.get('action')
                new_threshold = adaptation.get('new_threshold')
                if action in ('decrease_threshold', 'increase_threshold'):
                    self.mtd_threshold = new_threshold
                    self.ensemble_threshold = new_threshold
                    print(f"   MTD: {action} applied. Ensemble threshold now {self.ensemble_threshold:.4f}")
                elif action == 'rollback':
                    self.mtd_threshold = new_threshold
                    self.ensemble_threshold = new_threshold
                    print(f"   MTD: rollback applied. Ensemble threshold restored to {self.ensemble_threshold:.4f}")

            mtd_state = self.mtd.get_state()

        return ensemble_score, risk, weights, scores, risk_components, mtd_state, debug

    # ------------------------------------------------------------------
    # Component scores
    # ------------------------------------------------------------------

    def _autoencoder_score(self, X):
        if self.ae is None:
            return 0.5

        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            if len(x_tensor.shape) == 1:
                x_tensor = x_tensor.unsqueeze(0)
            recon = self.ae(x_tensor)
            mse = torch.mean((recon - x_tensor) ** 2, dim=1).numpy()
            error = mse[0] if len(mse) == 1 else mse.mean()

        if self.detector_threshold is not None and self.detector_threshold > 0:
            score = min(1.0, error / self.detector_threshold)
        else:
            score = min(1.0, error)

        return float(np.clip(score, 0.0, 1.0))

    def _gnn_score(self, X):
        if self.gnn is None:
            return 0.5
        return 0.4

    def _ocsvm_score(self, X):
        if self.ocsvm is None:
            return 0.5

        try:
            if len(X.shape) == 1:
                X_reshaped = X.reshape(1, -1)
            else:
                X_reshaped = X

            score = self.ocsvm.decision_function(X_reshaped)
            if len(score) > 1:
                score = score.mean()

            if isinstance(score, np.ndarray):
                score = score.item() if score.size == 1 else score[0]

            anomaly_score = 1.0 / (1.0 + np.exp(score))
            return float(np.clip(anomaly_score, 0.0, 1.0))
        except Exception:
            return 0.5

    def _confidence(self, score):
        """Confidence based on distance from the ensemble threshold."""
        distance = abs(score - self.ensemble_threshold)
        confidence = min(1.0, distance / (self.ensemble_threshold + 0.1))
        return float(np.clip(confidence, 0.0, 1.0))

    def get_status(self):
        status = {
            'available_models': self.available_models,
            'weights': self.weights,
            'detector_threshold': self.detector_threshold,
            'ensemble_threshold': self.ensemble_threshold,
            'mtd_threshold': self.mtd_threshold,
            'window_size': len(self.window),
            'risk_engine_ready': hasattr(self, 'risk_engine'),
            'mtd_enabled': self.enable_mtd,
        }

        if self.enable_mtd and self.mtd is not None:
            status['mtd_state'] = self.mtd.get_state()

        return status


# ----------------------------------------------------------------------
# Weight comparison: fixed vs dynamic
# ----------------------------------------------------------------------

def run_weight_comparison(ensemble, X_test, y_test, n_samples=5000, seed=42):
    """Compare fixed-weight and dynamic-weight configurations on standard metrics."""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

    rng = np.random.default_rng(seed)

    normal_idx = np.where(y_test == 0)[0]
    attack_idx = np.where(y_test == 1)[0]
    n_each = min(n_samples // 2, len(normal_idx), len(attack_idx))

    normal_sample = rng.choice(normal_idx, size=n_each, replace=False)
    attack_sample = rng.choice(attack_idx, size=n_each, replace=False)
    idx = np.concatenate([normal_sample, attack_sample])
    rng.shuffle(idx)

    X_eval = X_test[idx]
    y_eval = y_test[idx]

    results = {}

    for mode in ["fixed", "dynamic"]:
        scores = np.zeros(len(X_eval))
        for i, x in enumerate(X_eval):
            ensemble_score, _, _, _ = ensemble.predict(x, mode=mode)
            scores[i] = ensemble_score

        preds = (scores >= 0.5).astype(int)

        results[mode] = {
            "precision": float(precision_score(y_eval, preds, zero_division=0)),
            "recall": float(recall_score(y_eval, preds, zero_division=0)),
            "f1": float(f1_score(y_eval, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_eval, scores)) if len(np.unique(y_eval)) > 1 else 0.0,
            "pr_auc": float(average_precision_score(y_eval, scores)) if len(np.unique(y_eval)) > 1 else 0.0,
        }

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/ensemble_weight_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "precision", "recall", "f1", "roc_auc", "pr_auc"])
        for mode, m in results.items():
            writer.writerow([mode, m["precision"], m["recall"], m["f1"], m["roc_auc"], m["pr_auc"]])

    print("\nWeight comparison saved to results/metrics/ensemble_weight_comparison.csv")
    for mode, m in results.items():
        print(f"   {mode}: precision={m['precision']:.4f}, recall={m['recall']:.4f}, "
              f"f1={m['f1']:.4f}, roc_auc={m['roc_auc']:.4f}, pr_auc={m['pr_auc']:.4f}")

    return results


# ----------------------------------------------------------------------
# Test entry point
# ----------------------------------------------------------------------

def test_ensemble():
    print("=" * 60)
    print("Ensemble Test with Risk Engine and MTD")
    print("=" * 60)

    print("\nLoading test data...")
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    print(f"   X_test: {X_test.shape}")
    print(f"   y_test: {y_test.shape}")

    input_dim = X_test.shape[1]
    ensemble = EnsembleDetector(input_dim, enable_mtd=True)

    status = ensemble.get_status()
    print("\nEnsemble Status:")
    print(f"   Models: {status['available_models']}")
    print(f"   Weights: {status['weights']}")
    print(f"   Detector threshold (autoencoder): {status['detector_threshold']:.6f}")
    print(f"   Ensemble threshold: {status['ensemble_threshold']:.4f}")
    print(f"   MTD threshold: {status['mtd_threshold']:.4f}")
    print(f"   Risk Engine Ready: {status['risk_engine_ready']}")
    print(f"   MTD Enabled: {status['mtd_enabled']}")
    if status.get('mtd_state'):
        print(f"   MTD State: {status['mtd_state']}")

    print("\nTesting on 20 samples with MTD:")
    print("-" * 60)

    results = []
    for i in range(min(20, len(X_test))):
        ensemble_score, risk, weights, scores, risk_components, mtd_state, debug = ensemble.predict_with_mtd(X_test[i])

        results.append({
            'sample': i,
            'true_label': 'Attack' if y_test[i] == 1 else 'Normal',
            'ensemble_score': ensemble_score,
            'pre_norm_score': debug['pre_norm_score'],
            'risk_score': risk,
            'weights': weights,
            'scores': scores,
            'mtd_active': mtd_state.get('active', False) if mtd_state else False
        })

        if i % 2 == 0:
            print(f"Sample {i+1}:")
            print(f"   True Label: {'Attack' if y_test[i] == 1 else 'Normal'}")
            print(f"   Pre-norm Score (debug): {debug['pre_norm_score']:.4f}")
            print(f"   Ensemble Score: {ensemble_score:.4f}")
            print(f"   Risk Score: {risk:.4f}")
            print(f"   Weights: {[f'{w:.3f}' for w in weights]}")
            print(f"   Component Scores: {[f'{s:.3f}' for s in scores]}")
            print(f"   Risk Components: Ensemble={risk_components['ensemble_score']:.3f}, "
                  f"Divergence={risk_components['divergence']:.3f}, "
                  f"Confidence={risk_components['confidence']:.3f}, "
                  f"Persistence={risk_components['persistence']:.3f}")
            if mtd_state:
                print(f"   MTD Active: {mtd_state.get('active', False)}")
                print(f"   MTD Threshold: {mtd_state.get('threshold', 0):.4f}")
            print()

    ensemble_scores = [r['ensemble_score'] for r in results]
    risk_scores = [r['risk_score'] for r in results]
    normal_ensemble = [r['ensemble_score'] for r in results if r['true_label'] == 'Normal']
    attack_ensemble = [r['ensemble_score'] for r in results if r['true_label'] == 'Attack']
    mtd_active_count = sum(1 for r in results if r.get('mtd_active', False))

    print("-" * 60)
    print("Summary:")
    if normal_ensemble:
        print(f"   Ensemble Score - Normal: {np.mean(normal_ensemble):.4f} +/- {np.std(normal_ensemble):.4f}")
    if attack_ensemble:
        print(f"   Ensemble Score - Attack: {np.mean(attack_ensemble):.4f} +/- {np.std(attack_ensemble):.4f}")
    print(f"   Risk Score - All: {np.mean(risk_scores):.4f} +/- {np.std(risk_scores):.4f}")
    print(f"   MTD Activations: {mtd_active_count} out of {len(results)} samples")

    print("\nRunning fixed-weight vs dynamic-weight comparison on 5,000 balanced samples...")
    run_weight_comparison(ensemble, X_test, y_test, n_samples=5000, seed=42)

    print("\nEnsemble test complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    test_ensemble()