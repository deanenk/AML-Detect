# reactive_agent.py
# Reactive agent for low-latency monitoring

from spade.agent import Agent
from spade.behaviour import PeriodicBehaviour
from spade.message import Message
import json
import time
import numpy as np

class ReactiveAgent(Agent):
    """Reactive agent for low-latency anomaly monitoring."""
    
    def __init__(self, jid, password, ensemble=None):
        super().__init__(jid, password)
        self.ensemble = ensemble
        self.alerts_sent = 0
        self.alert_threshold = 0.6
        self.test_data = None
        self.test_labels = None
        self.current_index = 0
    
    class MonitorBehaviour(PeriodicBehaviour):
        async def run(self):
            if self.agent.ensemble is None:
                return
            
            # Load test data if not loaded
            if self.agent.test_data is None:
                try:
                    self.agent.test_data = np.load('data/processed/X_test.npy')
                    self.agent.test_labels = np.load('data/processed/y_test.npy')
                    print(f"   📊 Reactive Agent: Loaded {len(self.agent.test_data)} test samples")
                except Exception as e:
                    print(f"   ⚠️ Reactive Agent: Could not load data: {e}")
                    return
            
            # Process one sample per tick (simulating streaming)
            if self.agent.current_index < len(self.agent.test_data):
                X = self.agent.test_data[self.agent.current_index]
                y = self.agent.test_labels[self.agent.current_index]
                
                # Get prediction
                ensemble_score, risk, weights, scores, risk_components = self.agent.ensemble.predict_with_mtd(X)
                
                # Check if anomaly detected
                if ensemble_score > self.agent.alert_threshold:
                    # Send alert to supervisor
                    msg = Message(to="supervisor@localhost")
                    msg.body = json.dumps({
                        "type": "anomaly_alert",
                        "timestamp": time.time(),
                        "sample_id": self.agent.current_index,
                        "ensemble_score": float(ensemble_score),
                        "risk_score": float(risk),
                        "true_label": "Attack" if y == 1 else "Normal"
                    })
                    await self.send(msg)
                    self.agent.alerts_sent += 1
                    print(f"   🔔 Reactive Agent: Alert #{self.agent.alerts_sent} (score={ensemble_score:.3f})")
                
                self.agent.current_index += 1
                
                # Pause briefly to simulate real-time
                await self.sleep(0.1)
    
    async def setup(self):
        print(f"🤖 Reactive Agent {self.jid} starting...")
        monitor = self.MonitorBehaviour(period=1.0)
        self.add_behaviour(monitor)