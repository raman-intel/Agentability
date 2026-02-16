"""Basic usage example for Agentability.

This example shows how to instrument a simple AI agent with Agentability.
"""

from agentability import Tracer, DecisionType, MemoryType, MemoryOperation
import random


def main():
    """Run basic example."""
    # Initialize tracer in offline mode (uses SQLite)
    tracer = Tracer(offline_mode=True, database_path="example.db")
    
    print("🚀 Agentability Basic Example")
    print("=" * 50)
    
    # Example 1: Track a simple classification decision
    with tracer.trace_decision(
        agent_id="risk_classifier",
        decision_type=DecisionType.CLASSIFICATION,
        input_data={"loan_amount": 50000, "credit_score": 720, "income": 85000},
        tags=["loan_processing", "risk_assessment"]
    ) as decision_id:
        
        print(f"\n📋 Decision ID: {decision_id}")
        
        # Simulate agent reasoning
        credit_score = 720
        loan_amount = 50000
        income = 85000
        
        reasoning_steps = []
        
        # Check credit score
        if credit_score >= 700:
            reasoning_steps.append(f"Credit score {credit_score} meets minimum 700")
        else:
            reasoning_steps.append(f"Credit score {credit_score} below minimum 700")
        
        # Check debt-to-income ratio
        dti_ratio = loan_amount / income
        if dti_ratio < 0.43:
            reasoning_steps.append(f"DTI ratio {dti_ratio:.2f} is acceptable (<0.43)")
        else:
            reasoning_steps.append(f"DTI ratio {dti_ratio:.2f} exceeds limit 0.43")
        
        # Make decision
        approved = credit_score >= 700 and dti_ratio < 0.43
        confidence = 0.85 if approved else 0.72
        
        # Record the decision with full provenance
        tracer.record_decision(
            output={"approved": approved, "amount": loan_amount if approved else 0},
            confidence=confidence,
            reasoning=reasoning_steps,
            uncertainties=[
                "Employment stability not verified",
                "No rental payment history available"
            ],
            assumptions=[
                "Income figures are current and accurate",
                "No undisclosed debts exist"
            ],
            constraints_checked=[
                "Credit score >= 700",
                "DTI ratio < 0.43"
            ],
            data_sources=["credit_bureau", "income_verification"]
        )
        
        print(f"✅ Decision recorded: {'APPROVED' if approved else 'DENIED'}")
        print(f"   Confidence: {confidence:.2%}")
    
    # Example 2: Track a memory retrieval operation
    print("\n\n💾 Memory Operation Example")
    print("=" * 50)
    
    # Simulate vector memory retrieval (RAG)
    operation_id = tracer.record_memory_operation(
        agent_id="risk_classifier",
        memory_type=MemoryType.VECTOR,
        operation=MemoryOperation.RETRIEVE,
        latency_ms=42.5,
        items_processed=10,
        
        # Vector-specific metrics
        vector_dimension=1536,
        similarity_threshold=0.75,
        top_k=10,
        avg_similarity=0.82,
        min_similarity=0.76,
        max_similarity=0.91,
        
        # Quality metrics
        retrieval_precision=0.85,
        retrieval_recall=0.78,
    )
    
    print(f"✅ Memory operation recorded: {operation_id}")
    print(f"   Retrieval precision: 85%")
    print(f"   Average similarity: 0.82")
    
    # Example 3: Track an LLM call
    print("\n\n🤖 LLM Call Example")
    print("=" * 50)
    
    llm_call_id = tracer.record_llm_call(
        agent_id="risk_classifier",
        provider="anthropic",
        model="claude-sonnet-4",
        prompt_tokens=1500,
        completion_tokens=800,
        latency_ms=1250.0,
        cost_usd=0.0435,  # Auto-calculated if not provided
        finish_reason="stop",
        temperature=0.7,
        max_tokens=2000
    )
    
    print(f"✅ LLM call recorded: {llm_call_id}")
    print(f"   Total tokens: 2,300")
    print(f"   Cost: $0.0435")
    
    # Query decisions
    print("\n\n🔍 Querying Decisions")
    print("=" * 50)
    
    decisions = tracer.query_decisions(
        agent_id="risk_classifier",
        limit=10
    )
    
    print(f"Found {len(decisions)} decision(s)")
    for decision in decisions:
        print(f"  • {decision.decision_id}: {decision.decision_type.value}")
        print(f"    Confidence: {decision.confidence:.2%}")
        print(f"    Reasoning steps: {len(decision.reasoning)}")
    
    # Close tracer
    tracer.close()
    print("\n\n✅ Example completed successfully!")
    print(f"   Data saved to: example.db")


if __name__ == "__main__":
    main()
