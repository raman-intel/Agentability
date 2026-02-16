"""Multi-agent system example with conflict detection.

This example demonstrates tracking multiple agents with conflicting goals.
"""

from agentability import Tracer, DecisionType, ConflictType
from uuid import uuid4


def main():
    """Run multi-agent example."""
    tracer = Tracer(offline_mode=True, database_path="multi_agent_example.db")
    
    print("🤖 Multi-Agent System Example")
    print("=" * 60)
    print("Scenario: Customer support automation with routing and quality agents")
    
    session_id = str(uuid4())
    
    # Agent 1: Routing Agent (prioritizes speed)
    print("\n\n📱 Agent 1: Routing Agent")
    print("-" * 60)
    
    with tracer.trace_decision(
        agent_id="routing_agent",
        session_id=session_id,
        decision_type=DecisionType.DELEGATION,
        input_data={"ticket": "My order is late", "priority": "high"},
        tags=["routing", "customer_support"]
    ) as routing_decision_id:
        
        # Routing agent wants fast resolution
        tracer.record_decision(
            output={"route_to": "bot", "escalate": False},
            confidence=0.78,
            reasoning=[
                "Simple delivery inquiry",
                "Bot can handle in <60 seconds",
                "No human needed for efficiency"
            ],
            assumptions=["Customer wants fast response over personalized service"]
        )
        
        print(f"✅ Routing Decision: {routing_decision_id}")
        print(f"   Route to: BOT")
        print(f"   Goal: Maximize speed")
    
    # Agent 2: Quality Agent (prioritizes satisfaction)
    print("\n\n⭐ Agent 2: Quality Agent")
    print("-" * 60)
    
    with tracer.trace_decision(
        agent_id="quality_agent",
        session_id=session_id,
        decision_type=DecisionType.COORDINATION,
        input_data={"ticket": "My order is late", "customer_tier": "premium"},
        tags=["quality_control", "customer_support"]
    ) as quality_decision_id:
        
        # Quality agent wants high satisfaction
        tracer.record_decision(
            output={"route_to": "human", "escalate": True},
            confidence=0.92,
            reasoning=[
                "Premium customer detected",
                "Delivery issues need empathy",
                "Human agent provides better experience"
            ],
            assumptions=["Customer values personalized service over speed"]
        )
        
        print(f"✅ Quality Decision: {quality_decision_id}")
        print(f"   Route to: HUMAN")
        print(f"   Goal: Maximize satisfaction")
    
    # Detect and record conflict
    print("\n\n⚠️  Conflict Detected!")
    print("-" * 60)
    
    conflict_id = tracer.record_conflict(
        session_id=session_id,
        conflict_type=ConflictType.GOAL_CONFLICT,
        involved_agents=["routing_agent", "quality_agent"],
        agent_positions={
            "routing_agent": {
                "route_to": "bot",
                "goal": "minimize_response_time",
                "utility": 0.95  # High if routed to bot
            },
            "quality_agent": {
                "route_to": "human",
                "goal": "maximize_satisfaction",
                "utility": 0.90  # High if routed to human
            }
        },
        severity=0.75,  # High severity conflict
        resolution_strategy="quality_override",
        nash_equilibrium={
            "strategy": "route_to_human",
            "routing_utility": 0.40,
            "quality_utility": 0.90
        },
        pareto_optimal=True
    )
    
    print(f"✅ Conflict recorded: {conflict_id}")
    print(f"   Type: GOAL_CONFLICT")
    print(f"   Severity: 0.75 (High)")
    print(f"   Resolution: Quality agent wins (premium customer)")
    
    # Agent 3: Supervisor Agent (resolves conflict)
    print("\n\n👔 Agent 3: Supervisor Agent")
    print("-" * 60)
    
    with tracer.trace_decision(
        agent_id="supervisor_agent",
        session_id=session_id,
        decision_type=DecisionType.COORDINATION,
        parent_decision_id=routing_decision_id,
        input_data={
            "conflict_id": str(conflict_id),
            "agents": ["routing_agent", "quality_agent"]
        },
        tags=["conflict_resolution", "customer_support"]
    ) as supervisor_decision_id:
        
        tracer.record_decision(
            output={
                "final_route": "human",
                "override_agent": "routing_agent",
                "reason": "premium_customer_policy"
            },
            confidence=0.95,
            reasoning=[
                "Company policy: premium customers → human agents",
                "Customer satisfaction > speed for high-value customers",
                "Quality agent's assessment is correct"
            ],
            constraints_checked=["premium_customer_policy"],
            data_sources=["conflict_analysis", "customer_database"]
        )
        
        print(f"✅ Supervisor Decision: {supervisor_decision_id}")
        print(f"   Final Route: HUMAN")
        print(f"   Overrode: routing_agent")
        print(f"   Policy Applied: Premium customer protection")
    
    # Show decision hierarchy
    print("\n\n🌳 Decision Tree")
    print("-" * 60)
    print(f"├── {routing_decision_id} (routing_agent)")
    print(f"│   └── Route to BOT")
    print(f"├── {quality_decision_id} (quality_agent)")
    print(f"│   └── Route to HUMAN")
    print(f"└── {supervisor_decision_id} (supervisor_agent)")
    print(f"    └── RESOLVED: Route to HUMAN")
    
    # Query all decisions for this session
    decisions = tracer.query_decisions(session_id=session_id)
    
    print("\n\n📊 Session Summary")
    print("-" * 60)
    print(f"Session ID: {session_id}")
    print(f"Total Decisions: {len(decisions)}")
    print(f"Agents Involved: 3")
    print(f"Conflicts Detected: 1")
    print(f"Final Resolution: Route to human agent")
    
    tracer.close()
    print("\n✅ Multi-agent example completed!")


if __name__ == "__main__":
    main()
