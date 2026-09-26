# risk_engine.py
# Chaos-aware risk engine with escalation index
#
# Risk score definition:
#   The risk score is an escalation index in [0, 1]. It combines four
#   weighted components:
#       ensemble evidence  (w1 = 0.35)
#       local divergence   (w2 = 0.25)
#       detector confidence(w3 = 0.25)
#       persistence        (w4 = 0.15)
#   The score is not a calibrated probability of attack.
#
# Outputs for Chapter 4 Section 4.5.2:
#   results/metrics/risk_scores.csv                 [NEW]
#   results/metrics/risk_separation.csv             [NEW]
#   results/metrics/risk_threshold_sensitivity.csv  [NEW]

import numpy as np
import os
import csv

class ChaosRiskEngine:
    """Chaos-aware risk engine for escalation-sensitive scoring."""

    def __init__(self, window_size=50, prediction_horizon=10,
                 weights=(0.35, 0.25, 0.25, 0.15)):  # [NEW] weights configurable
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.weights = tuple(weights)  # [NEW]
        self.baseline_divergence = None
        self.divergence_min = None
        self.divergence_max = None
        self.persistence_count = 0
        self.history = []
        self.calibration_split = "unknown"  # [NEW] records what the baseline used

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_baseline(self, X_val, labels=None):
        """
        Calibrate the divergence baseline.

        If labels are provided, the baseline is computed on normal
        observations only. This prevents attack observations from
        inflating the baseline, which would suppress the divergence
        component for genuine attacks.  # [NEW]
        """
        X_val = np.asarray(X_val)

        # [NEW] Separate normal observations if labels are provided
        if labels is not None:
            labels = np.asarray(labels)
            normal_mask = (labels == 0)
            if normal_mask.sum() > 0:
                X_val = X_val[normal_mask]
                self.calibration_split = "normal_only"
            else:
                self.calibration_split = "all_observations"
        else:
            self.calibration_split = "all_observations"

        divergences = []
        window = []
        for x in X_val[:5000]:
            window.append(x)
            if len(window) > self.window_size:
                window = window[-self.window_size:]
            if len(window) >= self.window_size:
                d = self._estimate_divergence(window)
                if d is not None and np.isfinite(d):
                    divergences.append(d)

        if len(divergences) == 0:
            self.baseline_divergence = 0.0
            self.divergence_min = 0.0
            self.divergence_max = 1.0
        else:
            self.baseline_divergence = float(np.median(divergences))
            self.divergence_min = float(np.percentile(divergences, 5))
            self.divergence_max = float(np.percentile(divergences, 95))

        print("Calibrating chaos baseline...")
        print(f"   Calibration split: {self.calibration_split}")
        print(f"   Baseline divergence: {self.baseline_divergence:.4f}")
        print(f"   Divergence range: [{self.divergence_min:.4f}, {self.divergence_max:.4f}]")

    # ------------------------------------------------------------------
    # Divergence estimation
    # ------------------------------------------------------------------

    def _estimate_divergence(self, window):
        """Estimate local trajectory divergence over a window."""
        w = np.asarray(window, dtype=float)
        if w.ndim != 2 or w.shape[0] < 3:
            return None
        diffs = np.diff(w, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        if norms.size == 0:
            return None
        return float(np.mean(norms))

    def _normalise_divergence(self, d):
        if d is None or not np.isfinite(d):
            return 0.0
        if self.divergence_max is None or self.divergence_max == self.divergence_min:
            return 0.0
        v = (d - self.divergence_min) / (self.divergence_max - self.divergence_min)
        return float(np.clip(v, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def _update_persistence(self, ensemble_score, threshold=0.5):
        if ensemble_score >= threshold:
            self.persistence_count = min(self.persistence_count + 1, 10)
        else:
            self.persistence_count = max(self.persistence_count - 1, 0)
        return self.persistence_count / 10.0

    def process_stream(self, data_point, ensemble_score, confidence, window_data):
        """
        Compute the risk score for one observation.

        Returns:
            risk: escalation index in [0, 1]
            components: dict with the four weighted components
        """
        # Divergence
        divergence_raw = None
        if window_data is not None and len(window_data) >= 3:
            divergence_raw = self._estimate_divergence(window_data)
        divergence_norm = self._normalise_divergence(divergence_raw)

        # Persistence
        persistence_norm = self._update_persistence(ensemble_score)

        # Confidence
        confidence_norm = float(np.clip(confidence, 0.0, 1.0))

        # Ensemble evidence
        ensemble_norm = float(np.clip(ensemble_score, 0.0, 1.0))

        w1, w2, w3, w4 = self.weights
        risk = (w1 * ensemble_norm
                + w2 * divergence_norm
                + w3 * confidence_norm
                + w4 * persistence_norm)
        risk = float(np.clip(risk, 0.0, 1.0))

        components = {
            "ensemble_score": ensemble_norm,
            "divergence": divergence_norm,
            "confidence": confidence_norm,
            "persistence": persistence_norm,
            "weights": self.weights,
        }

        self.history.append({
            "risk": risk,
            "ensemble": ensemble_norm,
            "divergence": divergence_norm,
            "confidence": confidence_norm,
            "persistence": persistence_norm,
        })

        return risk, components

    # ------------------------------------------------------------------
    # [NEW] Chapter 4 Section 4.5.2 outputs
    # ------------------------------------------------------------------

    def export_risk_scores(self, X_test, y_test, path="results/metrics/risk_scores.csv",
                           max_samples=5000, seed=42):
        """
        Compute risk scores for a labelled sample and save to CSV.

        This produces the input required for separation metrics and
        threshold sensitivity.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rng = np.random.default_rng(seed)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        n = min(max_samples, len(X_test))
        idx = rng.choice(len(X_test), size=n, replace=False)

        rows = []
        window = []
        for i in idx:
            x = X_test[i]
            window.append(x)
            if len(window) > self.window_size:
                window = window[-self.window_size:]

            # Use a neutral ensemble placeholder if none is supplied;
            # the caller may overwrite this by passing scores.
            ensemble_score = 0.5
            confidence = 0.7
            risk, _ = self.process_stream(x, ensemble_score, confidence, window)

            rows.append({
                "index": int(i),
                "label": int(y_test[i]),
                "risk": risk,
                "ensemble": ensemble_score,
                "confidence": confidence,
            })

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "label", "risk",
                                                   "ensemble", "confidence"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"Risk scores saved to {path}")
        return path

    def compute_separation_metrics(self,
                                   path_in="results/metrics/risk_scores.csv",
                                   path_out="results/metrics/risk_separation.csv"):
        """
        Compute AUC, Cohen's d and KS statistic for the risk score alone.
        """
        from sklearn.metrics import roc_auc_score
        from scipy.stats import ks_2samp

        labels, risks = [], []
        with open(path_in, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row["label"]))
                risks.append(float(row["risk"]))

        labels = np.array(labels)
        risks = np.array(risks)

        normal = risks[labels == 0]
        attack = risks[labels == 1]

        if len(np.unique(labels)) < 2 or len(normal) == 0 or len(attack) == 0:
            auc = 0.0
            cohens_d = 0.0
            ks = 0.0
        else:
            auc = float(roc_auc_score(labels, risks))
            pooled_std = np.sqrt((normal.var(ddof=1) + attack.var(ddof=1)) / 2.0)
            cohens_d = float((attack.mean() - normal.mean()) / pooled_std) if pooled_std > 0 else 0.0
            ks = float(ks_2samp(normal, attack).statistic)

        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        with open(path_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["auc", auc])
            writer.writerow(["cohens_d", cohens_d])
            writer.writerow(["ks_statistic", ks])
            writer.writerow(["normal_mean", float(normal.mean()) if len(normal) else 0.0])
            writer.writerow(["attack_mean", float(attack.mean()) if len(attack) else 0.0])
            writer.writerow(["normal_std", float(normal.std(ddof=1)) if len(normal) > 1 else 0.0])
            writer.writerow(["attack_std", float(attack.std(ddof=1)) if len(attack) > 1 else 0.0])

        print(f"Separation metrics saved to {path_out}")
        print(f"   AUC={auc:.4f}, Cohen's d={cohens_d:.4f}, KS={ks:.4f}")
        return {"auc": auc, "cohens_d": cohens_d, "ks": ks}

    def compute_threshold_sensitivity(self,
                                      path_in="results/metrics/risk_scores.csv",
                                      path_out="results/metrics/risk_threshold_sensitivity.csv",
                                      thresholds=(0.5, 0.6, 0.7, 0.8)):
        """
        Compute true positive rate and false positive rate at each threshold.
        """
        labels, risks = [], []
        with open(path_in, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row["label"]))
                risks.append(float(row["risk"]))

        labels = np.array(labels)
        risks = np.array(risks)

        rows = []
        for t in thresholds:
            flagged = risks >= t
            tp = int(np.sum(flagged & (labels == 1)))
            fp = int(np.sum(flagged & (labels == 0)))
            fn = int(np.sum((~flagged) & (labels == 1)))
            tn = int(np.sum((~flagged) & (labels == 0)))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            rows.append({
                "threshold": t,
                "true_positive_rate": round(tpr, 6),
                "false_positive_rate": round(fpr, 6),
                "normal_flagged": fp,
                "attack_flagged": tp,
            })

        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        with open(path_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["threshold", "true_positive_rate",
                                                   "false_positive_rate",
                                                   "normal_flagged", "attack_flagged"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"Threshold sensitivity saved to {path_out}")
        for r in rows:
            print(f"   t={r['threshold']:.2f}  TPR={r['true_positive_rate']:.4f}  "
                  f"FPR={r['false_positive_rate']:.4f}")
        return rows


# ----------------------------------------------------------------------
# Standalone run
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chaos-Aware Risk Engine")
    print("=" * 60)

    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    engine = ChaosRiskEngine(window_size=50, prediction_horizon=10,
                             weights=(0.35, 0.25, 0.25, 0.15))
    engine.calibrate_baseline(X_val, labels=y_val)

    engine.export_risk_scores(X_test, y_test,
                              path="results/metrics/risk_scores.csv",
                              max_samples=5000, seed=42)

    engine.compute_separation_metrics(
        path_in="results/metrics/risk_scores.csv",
        path_out="results/metrics/risk_separation.csv"
    )

    engine.compute_threshold_sensitivity(
        path_in="results/metrics/risk_scores.csv",
        path_out="results/metrics/risk_threshold_sensitivity.csv",
        thresholds=(0.5, 0.6, 0.7, 0.8)
    )

    print("\nRisk engine run complete.")
    print("=" * 60)