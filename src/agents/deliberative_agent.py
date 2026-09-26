# deliberative_agent.py
# Deliberative agent for deep ensemble and risk analysis

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
import json
import time
import numpy as np

class DeliberativeAgent(Agent):
    """Deliberative agent for deeper ensemble and risk analysis."""
    
    def __init__(self, jid, password, ensemble=None):
        super().__init__(jid, password)
        self.ensemble = ensemble
        self.analysis_count = 0
    
    class AnalyseBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=5)
            if msg:
                try:
                    request = json.loads(msg.body)
                    if request.get('type') == 'deep_analysis':
                        # Perform deep analysis
                        sample = request.get('sample', [])
                        
                        if self.agent.ensemble and len(sample) > 0:
                            sample = np.array(sample)
                            ensemble_score, risk, weights, scores, risk_components = self.agent.ensemble.predict_with_mtd(sample)
                            
                            result = {
                                "type": "analysis_result",
                                "timestamp": time.time(),
                                "ensemble_score": float(ensemble_score),
                                "risk_score": float(risk),
                                "weights": [float(w) for w in weights],
                                "component_scores": [float(s) for s in scores],
                                "risk_components": {k: float(v) for k, v in risk_components.items()}
                            }
                            
                            response = Message(to=msg.sender)
                            response.body = json.dumps(result)
                            await self.send(response)
                            self.agent.analysis_count += 1
                            print(f"   🧠 Deliberative Agent: Analysis #{self.agent.analysis_count} sent")
                except Exception as e:
                    print(f"   ⚠️ Deliberative Agent error: {e}")
    
    async def setup(self):
        print(f"🧠 Deliberative Agent {self.jid} starting...")
        analyse = self.AnalyseBehaviour()
        self.add_behaviour(analyse)