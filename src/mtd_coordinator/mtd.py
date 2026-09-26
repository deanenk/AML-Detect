# mtd.py
# Moving-Target Defence Coordinator
#
# Threshold ownership:
#   The MTD threshold is a risk-score threshold in [0, 1]. It is not the
#   autoencoder reconstruction-error threshold and not the ensemble score
#   threshold. Three thresholds exist in AML-Detect and they are not
#   interchangeable:
#       - Detector threshold (autoencoder): reconstruction error, positive real
#       - Ensemble threshold: weighted ensemble score in [0, 1]
#       - MTD threshold: risk score in [0, 1]
#
# Adaptation direction:
#   When risk is high, scrutiny should increase, not decrease. The MTD
#   coordinator therefore DECREASES the detection threshold during high-risk
#   periods, making the detector more sensitive. It restores the threshold
#   when risk subsides. The previous implementation increased the threshold,
#   which reduced sensitivity when scrutiny should increase. This reversal
#   is the core fix in this version.
#
# Hysteresis:
#   Activation at 0.70 and rollback at 0.30 create a 0.40 gap. The gap
#   prevents rapid state oscillation when the risk score fluctuates near a
#   single value. The specific values are initial, not calibrated.
#
# Logs:
#   results/metrics/mtd_adaptations.csv   one row per activation or rollback
#   results/metrics/mtd_counts.csv        activation and rollback totals

import os
import csv
import time


class MTDCoordinator:
    """Moving-target defence coordinator with reversible threshold adaptation."""

    def __init__(self,
                 activation_threshold=0.70,
                 rollback_threshold=0.30,
                 adaptation_factor=0.7,
                 base_threshold=0.5,
                 log_dir="results/metrics"):
        """
        Args:
            activation_threshold: risk score above which MTD activates
            rollback_threshold:   risk score below which MTD rolls back
            adaptation_factor:    multiplier applied to the base threshold on
                                  activation. Values < 1 decrease the threshold
                                  (increase sensitivity). Values > 1 increase
                                  the threshold (decrease sensitivity).
                                  Default 0.7 is a decrease, consistent with
                                  the security rationale.
            base_threshold:       detection threshold in [0, 1] to adapt
            log_dir:              directory for adaptation logs
        """
        self.activation_threshold = activation_threshold
        self.rollback_threshold = rollback_threshold
        self.adaptation_factor = adaptation_factor
        self.base_threshold = base_threshold

        self.state = {
            "active": False,
            "threshold": base_threshold,
            "original_threshold": base_threshold,
        }
        self.current_adaptation = None
        self.adaptation_history = []

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.adaptations_path = os.path.join(self.log_dir, "mtd_adaptations.csv")
        self.counts_path = os.path.join(self.log_dir, "mtd_counts.csv")
        self._init_adaptations_log()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_adaptations_log(self):
        if not os.path.exists(self.adaptations_path):
            with open(self.adaptations_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "event_id", "timestamp", "action", "reason",
                    "risk_score", "original_threshold", "new_threshold"
                ])

    def _log_adaptation(self, action, reason, risk_score,
                        original_threshold, new_threshold):
        event_id = len(self.adaptation_history)
        with open(self.adaptations_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                event_id,
                round(time.time(), 4),
                action,
                reason,
                round(float(risk_score), 6),
                round(float(original_threshold), 6),
                round(float(new_threshold), 6),
            ])

    def _write_counts(self):
        activations = sum(1 for h in self.adaptation_history if h["type"] == "apply")
        rollbacks = sum(1 for h in self.adaptation_history if h["type"] == "rollback")
        with open(self.counts_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["activations", activations])
            writer.writerow(["rollbacks", rollbacks])
            writer.writerow(["final_state_active", int(self.state["active"])])
            writer.writerow(["final_threshold", round(self.state["threshold"], 6)])

    # ------------------------------------------------------------------
    # Core adaptation
    # ------------------------------------------------------------------

    def adapt(self, risk_score, current_state=None):
        """
        Decide whether to activate, roll back or hold.

        Args:
            risk_score: escalation index in [0, 1]
            current_state: optional dict with current threshold

        Returns:
            dict describing the adaptation, or None if no change
        """
        if current_state is not None and "threshold" in current_state:
            self.base_threshold = current_state["threshold"]

        # Activate: risk above activation threshold and not already active
        if risk_score > self.activation_threshold and not self.state["active"]:
            new_threshold = self.base_threshold * self.adaptation_factor
            new_threshold = float(max(0.0, min(1.0, new_threshold)))

            adaptation = {
                "id": len(self.adaptation_history) + 1,
                "action": "decrease_threshold",
                "original_threshold": self.base_threshold,
                "new_threshold": new_threshold,
                "timestamp": time.time(),
                "reason": f"Risk score {risk_score:.4f} exceeded activation threshold "
                          f"{self.activation_threshold:.2f}. Threshold decreased to "
                          f"increase sensitivity.",
            }
            self.apply_adaptation(adaptation)
            self._log_adaptation("apply", adaptation["reason"], risk_score,
                                 adaptation["original_threshold"],
                                 adaptation["new_threshold"])
            self._write_counts()
            return adaptation

        # Roll back: risk below rollback threshold and currently active
        if risk_score < self.rollback_threshold and self.state["active"]:
            original = self.state["original_threshold"]
            adaptation = {
                "id": len(self.adaptation_history) + 1,
                "action": "rollback",
                "original_threshold": self.state["threshold"],
                "new_threshold": original,
                "timestamp": time.time(),
                "reason": f"Risk score {risk_score:.4f} fell below rollback threshold "
                          f"{self.rollback_threshold:.2f}. Threshold restored.",
            }
            self.apply_adaptation(adaptation)
            self._log_adaptation("rollback", adaptation["reason"], risk_score,
                                 adaptation["original_threshold"],
                                 adaptation["new_threshold"])
            self._write_counts()
            return adaptation

        return None

    def apply_adaptation(self, adaptation):
        """Apply the adaptation to the MTD state."""
        action = adaptation["action"]

        if action in ("decrease_threshold", "increase_threshold"):
            self.state["original_threshold"] = adaptation["original_threshold"]
            self.state["threshold"] = adaptation["new_threshold"]
            self.state["active"] = True
        elif action == "rollback":
            self.state["threshold"] = adaptation["new_threshold"]
            self.state["active"] = False

        self.current_adaptation = adaptation
        self.adaptation_history.append({
            "timestamp": adaptation["timestamp"],
            "type": "apply" if action != "rollback" else "rollback",
            "action": action,
            "original_threshold": adaptation["original_threshold"],
            "new_threshold": adaptation["new_threshold"],
            "reason": adaptation["reason"],
        })

        print(f"   MTD applied: {action} -> threshold {self.state['threshold']:.4f}")

    def rollback(self):
        """Explicit rollback to the original threshold."""
        if not self.state["active"]:
            return False

        original = self.state["original_threshold"]
        adaptation = {
            "id": len(self.adaptation_history) + 1,
            "action": "rollback",
            "original_threshold": self.state["threshold"],
            "new_threshold": original,
            "timestamp": time.time(),
            "reason": "Explicit rollback requested.",
        }
        self.apply_adaptation(adaptation)
        self._log_adaptation("rollback", adaptation["reason"], 0.0,
                             adaptation["original_threshold"],
                             adaptation["new_threshold"])
        self._write_counts()
        return True

    def get_state(self):
        """Return the current MTD state."""
        return {
            "active": self.state["active"],
            "threshold": self.state["threshold"],
            "original_threshold": self.state["original_threshold"],
            "activation_threshold": self.activation_threshold,
            "rollback_threshold": self.rollback_threshold,
            "adaptation_factor": self.adaptation_factor,
            "adaptation_count": len(self.adaptation_history),
        }


# ----------------------------------------------------------------------
# Standalone run
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Moving-Target Defence Coordinator")
    print("=" * 60)

    mtd = MTDCoordinator(
        activation_threshold=0.70,
        rollback_threshold=0.30,
        adaptation_factor=0.7,
        base_threshold=0.5,
    )

    print("\nSimulating a risk sequence...")
    risk_sequence = [0.2, 0.4, 0.75, 0.8, 0.6, 0.4, 0.25, 0.1]

    for i, risk in enumerate(risk_sequence):
        result = mtd.adapt(risk)
        state = mtd.get_state()
        print(f"  Step {i+1}: risk={risk:.2f}, "
              f"active={state['active']}, "
              f"threshold={state['threshold']:.4f}, "
              f"adaptations={state['adaptation_count']}")

    print("\nFinal state:")
    for k, v in mtd.get_state().items():
        print(f"   {k}: {v}")

    print("\nLogs written to:")
    print(f"   {mtd.adaptations_path}")
    print(f"   {mtd.counts_path}")
    print("=" * 60)