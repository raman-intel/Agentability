# 🚀 AGENTABILITY

**The Observability Standard for Production AI Agents**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

> "While competitors show WHAT agents did, Agentability shows WHY they did it, HOW they thought about it, and WHAT capabilities they have."

## 📋 Table of Contents

- [Why Agentability?](#why-agentability)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Pricing](#pricing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Why Agentability?

Multi-agent systems are becoming production-critical, but existing observability tools fall short:

- **❌ Langfuse** - Great for prompts, not agent-native
- **❌ AgentOps** - SaaS-only, expensive at scale
- **❌ W&B** - Experiment tracking, not real-time observability
- **❌ Arize Phoenix** - Complex setup, GPU-focused

**✅ Agentability** fills the gap with:

1. **Memory Intelligence** - Track vector, episodic, semantic memory performance
2. **Decision Provenance** - Complete "why" chain for every decision
3. **Conflict Analytics** - Game-theoretic multi-agent disagreement analysis
4. **Causal Graphs** - Temporal causality mapping (not just traces)
5. **Zerodha-Class UX** - Best-in-class charting, real-time updates
6. **Offline + Online** - Works without infrastructure (SQLite) or scales (TimescaleDB)

## ✨ Key Features

### 1. 🔍 Decision Provenance Engine

Track complete reasoning chains:
- ✅ Every thought step recorded
- ✅ Uncertainty tracking
- ✅ Assumption logging
- ✅ Constraint validation
- ✅ Information lineage

```python
with tracer.trace_decision(
    agent_id="risk_agent",
    decision_type=DecisionType.CLASSIFICATION
):
    result = agent.assess(loan_application)
    tracer.record_decision(
        output=result,
        confidence=0.85,
        reasoning=["Credit score meets minimum", "Income verified"],
        uncertainties=["Employment history only 2 years"],
        assumptions=["W2 income is current"]
    )
```

### 2. 🧠 Memory Intelligence System

**NOBODY ELSE TRACKS THIS!**

Monitor all memory subsystems:

- **Vector Memory** (RAG/embeddings)
  - Retrieval precision/recall
  - Similarity distributions
  - Freshness tracking
  
- **Episodic Memory** (sequential experiences)
  - Temporal coherence
  - Context window utilization
  - Episode replay patterns
  
- **Semantic Memory** (knowledge graphs)
  - Graph traversal metrics
  - Relationship density
  - Query complexity

```python
tracer.record_memory_operation(
    agent_id="agent_1",
    memory_type=MemoryType.VECTOR,
    operation=MemoryOperation.RETRIEVE,
    latency_ms=42.5,
    items_processed=10,
    avg_similarity=0.82,
    retrieval_precision=0.85
)
```

### 3. ⚔️ Multi-Agent Conflict Analytics

Game-theoretic conflict analysis:

```python
conflict_id = tracer.record_conflict(
    session_id="session_123",
    conflict_type=ConflictType.GOAL_CONFLICT,
    involved_agents=["agent_A", "agent_B"],
    agent_positions={
        "agent_A": {"goal": "minimize_cost", "utility": 0.8},
        "agent_B": {"goal": "maximize_quality", "utility": 0.9}
    },
    severity=0.75,
    nash_equilibrium={"strategy": "compromise"}
)
```

### 4. 💰 LLM Cost Optimization

Automatic cost tracking and optimization:

```python
llm_call_id = tracer.record_llm_call(
    agent_id="agent_1",
    provider="anthropic",
    model="claude-sonnet-4",
    prompt_tokens=1500,
    completion_tokens=800,
    latency_ms=1250.0,
    cost_usd=0.0435  # Auto-calculated
)
```

### 5. 📊 Zerodha-Class Dashboard

Production-grade UI with:
- Real-time WebSocket updates
- Interactive causal graphs (D3.js)
- Advanced charting (Lightweight Charts)
- Memory performance heatmaps
- Conflict visualization matrix

## 🚀 Quick Start

### Installation

```bash
# Python SDK
pip install agentability

# TypeScript SDK
npm install agentability

# Self-hosted platform (Docker)
docker-compose up -d
```

### 30-Second Example

```python
from agentability import Tracer, DecisionType

# Initialize tracer (offline mode uses SQLite)
tracer = Tracer(offline_mode=True)

# Track a decision
with tracer.trace_decision(
    agent_id="my_agent",
    decision_type=DecisionType.CLASSIFICATION,
    input_data={"query": "Is this email spam?"}
):
    # Your agent logic here
    result = classify_email(email)
    
    # Record with full provenance
    tracer.record_decision(
        output={"is_spam": True},
        confidence=0.92,
        reasoning=[
            "Suspicious sender domain",
            "Contains phishing keywords"
        ],
        data_sources=["email_headers", "content_analysis"]
    )

# Query decisions
decisions = tracer.query_decisions(agent_id="my_agent", limit=10)
print(f"Tracked {len(decisions)} decisions")
```

## 📚 Usage Examples

### Example 1: Basic Tracking

See [examples/basic_usage.py](sdk/python/examples/basic_usage.py) for complete example.

### Example 2: Multi-Agent System

See [examples/multi_agent_system.py](sdk/python/examples/multi_agent_system.py) for conflict tracking.

### Example 3: LangChain Integration

```python
from agentability.integrations import AgentabilityLangChainCallback

callback = AgentabilityLangChainCallback(agent_id="langchain_agent")
chain.run(input_data, callbacks=[callback])
```

### Example 4: Memory Tracking

```python
# Track vector retrieval
tracker = tracer.record_memory_operation(
    agent_id="rag_agent",
    memory_type=MemoryType.VECTOR,
    operation=MemoryOperation.RETRIEVE,
    latency_ms=45.2,
    items_processed=10,
    avg_similarity=0.82,
    retrieval_precision=0.85
)
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AGENTABILITY PLATFORM                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Python SDK  │  │TypeScript SDK│  │    Go SDK    │ │
│  │              │  │              │  │  (Coming Soon)│ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│          │                 │                 │          │
│          └─────────────────┴─────────────────┘          │
│                          │                               │
│                ┌─────────┴──────────┐                   │
│                │  OTEL Collector    │                   │
│                │  (Optional)        │                   │
│                └─────────┬──────────┘                   │
│                          │                               │
│         ┌────────────────┼────────────────┐            │
│         │                │                │            │
│    ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐     │
│    │ SQLite  │    │  DuckDB   │   │TimescaleDB│     │
│    │(Offline)│    │(Analytics)│   │(Production)│     │
│    └─────────┘    └───────────┘   └───────────┘     │
│                                                          │
│    ┌──────────────────────────────────────────────┐    │
│    │         API Server (FastAPI)                 │    │
│    │  • REST & GraphQL endpoints                  │    │
│    │  • WebSocket for real-time                   │    │
│    │  • Background workers                        │    │
│    └──────────────────────────────────────────────┘    │
│                          │                               │
│    ┌─────────────────────▼──────────────────────────┐  │
│    │     Dashboard (React + TypeScript + Vite)     │  │
│    │  • Real-time metrics                          │  │
│    │  • Causal graph visualization                 │  │
│    │  • Memory performance heatmaps                │  │
│    │  • Conflict analysis matrix                   │  │
│    └───────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 💰 Pricing

| Tier | Price | Decisions/Month | Features |
|------|-------|----------------|----------|
| **Open Source** | Free | Unlimited | SQLite, self-hosted, full features |
| **Cloud Starter** | Free | 1M | Managed hosting, basic support |
| **Cloud Pro** | $49/mo | 10M | Priority support, advanced analytics |
| **Cloud Enterprise** | Custom | 100M+ | SLA, dedicated support, custom deployment |

**Cost per 1M decisions**: $2 (vs AgentOps $8, Langfuse $12)

## 📖 Documentation

Full documentation available at [docs.agentability.io](https://docs.agentability.io)

- [Getting Started Guide](docs/getting-started/quickstart.md)
- [Python SDK Reference](docs/api-reference/python-sdk.md)
- [TypeScript SDK Reference](docs/api-reference/typescript-sdk.md)
- [Memory Tracking Guide](docs/guides/memory-tracking.md)
- [Multi-Agent Systems Guide](docs/guides/multi-agent-systems.md)
- [Production Deployment](docs/guides/production-deployment.md)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/agentability/agentability.git
cd agentability

# Install Python SDK in development mode
cd sdk/python
pip install -e ".[dev]"

# Install TypeScript dependencies
cd ../typescript
npm install

# Run tests
pytest  # Python
npm test  # TypeScript

# Start local development stack
docker-compose -f infrastructure/docker/docker-compose.dev.yml up
```

## 📊 Comparison with Competitors

| Feature | Agentability | Langfuse | AgentOps | Arize Phoenix |
|---------|-------------|----------|----------|---------------|
| **Memory Tracking** | ✅ All types | ❌ | ❌ | ❌ |
| **Decision Provenance** | ✅ Complete | ⚠️ Partial | ⚠️ Partial | ❌ |
| **Multi-Agent Conflicts** | ✅ Game theory | ❌ | ❌ | ❌ |
| **Causal Graphs** | ✅ | ❌ | ❌ | ❌ |
| **Offline Mode** | ✅ SQLite | ❌ | ❌ | ⚠️ Limited |
| **Real-time Dashboard** | ✅ Zerodha-class | ⚠️ Basic | ✅ | ⚠️ Basic |
| **Cost per 1M** | $2 | $12 | $8 | $10 |
| **Open Source Core** | ✅ MIT | ⚠️ Partial | ❌ | ✅ |

## 🎯 Roadmap

### Q1 2026 (Current)
- [x] Python SDK core
- [x] SQLite storage
- [x] Basic dashboard
- [ ] TypeScript SDK
- [ ] LangChain integration
- [ ] CrewAI integration

### Q2 2026
- [ ] DuckDB analytics
- [ ] TimescaleDB production
- [ ] Advanced causal graphs
- [ ] AutoGen integration
- [ ] LlamaIndex integration

### Q3 2026
- [ ] Go SDK
- [ ] Kubernetes operators
- [ ] ML-powered insights
- [ ] Anomaly detection
- [ ] Cost optimization AI

### Q4 2026
- [ ] Enterprise features
- [ ] SOC 2 compliance
- [ ] Multi-region deployment
- [ ] Advanced RBAC

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with ❤️ by the Agentability team and contributors.

Special thanks to:
- The Anthropic team for Claude
- The LangChain community
- The OpenTelemetry project
- All our open-source contributors

## 📧 Contact

- Website: [agentability.io](https://agentability.io)
- Email: hello@agentability.io
- Twitter: [@agentability](https://twitter.com/agentability)
- Discord: [Join our community](https://discord.gg/agentability)

---

**Star ⭐ this repo if you find it useful!**

Made with 🚀 by developers, for developers building production AI agents.
