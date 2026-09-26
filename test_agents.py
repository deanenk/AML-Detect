# test_agents.py
# Test the multi-agent system

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from ensemble.ensemble import EnsembleDetector
from mtd_coordinator.mtd import MTDCoordinator
from agents.reactive_agent import ReactiveAgent
from agents.deliberative_agent import DeliberativeAgent
from agents.supervisor import SupervisorAgent

async def main():
    print("="*60)
    print("🤖 AML-Detect Multi-Agent System Test")
    print("="*60)
    
    # Load ensemble
    print("\n📂 Loading ensemble...")
    X_test = np.load('data/processed/X_test.npy')
    ensemble = EnsembleDetector(X_test.shape[1], enable_mtd=True)
    
    # Initialize MTD
    mtd = MTDCoordinator()
    
    print("\n🤖 Starting agents (requires XMPP server)...")
    print("   Note: Agents need an XMPP server to communicate.")
    print("   For testing without XMPP, use test_agents_simple.py")
    print("\n" + "="*60)
    print("To run agents, you need:")
    print("1. An XMPP server (e.g., Openfire or ejabberd)")
    print("2. User accounts: reactive@localhost, deliberative@localhost, supervisor@localhost")
    print("3. Uncomment the agent creation code below")
    print("="*60)
    
    # Uncomment these when XMPP server is ready:
    # reactive = ReactiveAgent("reactive@localhost", "password", ensemble)
    # deliberative = DeliberativeAgent("deliberative@localhost", "password", ensemble)
    # supervisor = SupervisorAgent("supervisor@localhost", "password", ensemble, mtd)
    # 
    # await reactive.start()
    # await deliberative.start()
    # await supervisor.start()
    # 
    # print("\n✅ Agents started! Running for 30 seconds...")
    # await asyncio.sleep(30)
    # 
    # print("\n📊 Summary:")
    # print(f"   Reactive alerts: {reactive.alerts_sent}")
    # print(f"   Deliberative analyses: {deliberative.analysis_count}")
    # print(f"   Supervisor alerts: {supervisor.alert_count}")
    # print(f"   Supervisor escalations: {supervisor.escalation_count}")
    # 
    # await reactive.stop()
    # await deliberative.stop()
    # await supervisor.stop()
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(main())