# supervisor.py
# Supervisor agent for orchestration and escalation control

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
import json
import time

class SupervisorAgent(Agent):
    """Supervisor agent for orchestration and escalation control."""
    
    def __init__(self, jid, password, ensemble=None, mtd=None):
        super().__init__(jid, password)
        self.ensemble = ensemble
        self.mtd = mtd
        self.alert_count = 0
        self.escalation_count = 0
        self.risk_threshold = 0.7
        self.state = {
            'active_alerts': [],
            'last_escalation': None,
            'pending_requests': []
        }
    
    class OrchestrateBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if msg:
                try:
                    data = json.loads(msg.body)
                    
                    if data.get('type') == 'anomaly_alert':
                        self.agent.alert_count += 1
                        print(f"   📨 Supervisor: Received alert #{self.agent.alert_count}")
                        
                        # Store alert
                        self.agent.state['active_alerts'].append(data)
                        if len(self.agent.state['active_alerts']) > 100:
                            self.agent.state['active_alerts'] = self.agent.state['active_alerts'][-100:]
                        
                        # Check if risk exceeds threshold
                        risk = data.get('risk_score', 0)
                        if risk > self.agent.risk_threshold:
                            # Request deep analysis from deliberative agent
                            await self.request_deep_analysis(data)
                    
                    elif data.get('type') == 'analysis_result':
                        print(f"   📊 Supervisor: Received analysis result")
                        
                        # Check if escalation needed
                        risk = data.get('risk_score', 0)
                        if risk > self.agent.risk_threshold:
                            await self.escalate(data)
                            
                except Exception as e:
                    print(f"   ⚠️ Supervisor error: {e}")
        
        async def request_deep_analysis(self, alert):
            """Request deep analysis from deliberative agent."""
            # Store pending request
            self.agent.state['pending_requests'].append(alert)
            
            msg = Message(to="deliberative@localhost")
            msg.body = json.dumps({
                "type": "deep_analysis",
                "sample": []  # In production, this would contain the actual sample
            })
            await self.send(msg)
            print(f"   🔍 Supervisor: Requested deep analysis")
        
        async def escalate(self, analysis):
            """Escalate to MTD coordinator."""
            self.agent.escalation_count += 1
            self.agent.state['last_escalation'] = {
                'timestamp': time.time(),
                'risk': analysis.get('risk_score', 0),
                'reason': 'Risk threshold exceeded'
            }
            
            # Send to MTD if available
            if self.agent.mtd:
                msg = Message(to="mtd@localhost")
                msg.body = json.dumps({
                    "type": "escalate",
                    "risk_score": analysis.get('risk_score', 0),
                    "ensemble_score": analysis.get('ensemble_score', 0),
                    "timestamp": time.time()
                })
                await self.send(msg)
                print(f"   ⚠️ Supervisor: Escalation #{self.agent.escalation_count} sent to MTD")
    
    async def setup(self):
        print(f"👤 Supervisor Agent {self.jid} starting...")
        orchestrate = self.OrchestrateBehaviour()
        self.add_behaviour(orchestrate)