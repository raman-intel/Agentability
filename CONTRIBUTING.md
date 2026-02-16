# Contributing to Agentability

Thank you for your interest in contributing to Agentability! This document provides guidelines and instructions for contributing.

## 🤝 Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/agentability.git
cd agentability

# Install Python dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Install dashboard dependencies
cd dashboard
npm install
cd ..

# Start development environment
docker-compose -f docker-compose.dev.yml up
```

## 📝 How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** and **description**
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Code samples** or **screenshots** if applicable

### Suggesting Features

Feature suggestions are welcome! Please:

- **Check existing feature requests** first
- **Provide clear use case** and **benefits**
- **Describe proposed API** or **user interface**
- **Consider backwards compatibility**

### Pull Requests

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Write** or **update tests**
5. **Run** tests (`pytest` and `npm test`)
6. **Update** documentation
7. **Commit** with clear messages
8. **Push** to your fork
9. **Open** a pull request

#### PR Guidelines

- **One feature per PR** - Keep PRs focused
- **Follow code style** - Run linters (`ruff`, `black`, `eslint`)
- **Add tests** - Maintain >80% coverage
- **Update docs** - Keep documentation in sync
- **Sign commits** - Use verified commits
- **Reference issues** - Link related issues

## 🏗️ Project Structure

```
agentability/
├── sdk/
│   ├── python/          # Python SDK
│   │   ├── agentability/
│   │   ├── tests/
│   │   └── examples/
│   └── typescript/      # TypeScript SDK
├── platform/           # Backend services
│   ├── api/
│   ├── collectors/
│   └── workers/
├── dashboard/          # React dashboard
├── infrastructure/     # Deployment configs
└── docs/              # Documentation
```

## 💻 Development Guidelines

### Python Code Style

We follow Google Python Style Guide with these tools:

```bash
# Format code
black sdk/python/agentability
ruff check --fix sdk/python/agentability

# Type checking
mypy sdk/python/agentability

# Run tests
pytest sdk/python/tests/
```

**Key Principles:**
- Use **type hints** everywhere
- Write **docstrings** (Google style)
- Keep functions **<50 lines**
- Prefer **composition** over inheritance
- Write **pure functions** when possible

### TypeScript Code Style

We follow Airbnb TypeScript Style Guide:

```bash
# Format code
npm run format

# Lint
npm run lint

# Type check
npm run type-check

# Run tests
npm test
```

**Key Principles:**
- **Strict TypeScript** mode enabled
- **Functional components** with hooks
- **Immutable data** patterns
- **Explicit return types**
- **Comprehensive JSDoc**

### Testing Requirements

- **Unit tests**: >80% coverage
- **Integration tests**: Critical paths
- **E2E tests**: User workflows
- **Performance tests**: Benchmarks

```python
# Python testing example
import pytest
from agentability import Tracer

def test_decision_tracking():
    """Test basic decision tracking functionality."""
    tracer = Tracer(offline_mode=True)
    
    with tracer.trace_decision(agent_id="test_agent"):
        tracer.record_decision(
            output="test_output",
            confidence=0.95
        )
    
    decisions = tracer.get_decisions(agent_id="test_agent")
    assert len(decisions) == 1
    assert decisions[0].confidence == 0.95
```

### Documentation

- **Code comments**: Explain "why", not "what"
- **Docstrings**: All public APIs
- **README**: Keep up-to-date
- **Guides**: For complex features
- **API reference**: Auto-generated from docstrings

### Commit Messages

Follow Conventional Commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Build/tooling changes

**Example:**
```
feat(sdk): add episodic memory tracking

- Implement EpisodicMemoryTracker class
- Add temporal coherence metrics
- Update documentation

Closes #123
```

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_tracer.py

# With coverage
pytest --cov=agentability --cov-report=html

# JavaScript tests
npm test
```

### Writing Tests

```python
"""Test module for decision provenance."""

import pytest
from datetime import datetime
from agentability import Tracer
from agentability.models import DecisionType

class TestDecisionProvenance:
    """Test suite for decision provenance tracking."""
    
    @pytest.fixture
    def tracer(self):
        """Create tracer instance for testing."""
        return Tracer(offline_mode=True)
    
    def test_basic_decision_recording(self, tracer):
        """Test recording a basic decision."""
        with tracer.trace_decision(
            agent_id="test_agent",
            decision_type=DecisionType.CLASSIFICATION
        ):
            tracer.record_decision(
                output="approved",
                confidence=0.85
            )
        
        decisions = tracer.get_decisions()
        assert len(decisions) == 1
        assert decisions[0].output == "approved"
    
    def test_decision_with_reasoning(self, tracer):
        """Test decision with complete provenance."""
        with tracer.trace_decision(agent_id="risk_agent"):
            tracer.record_decision(
                output="approved",
                confidence=0.75,
                reasoning=[
                    "Credit score meets threshold",
                    "Income verified"
                ],
                uncertainties=["Short employment history"]
            )
        
        decision = tracer.get_decisions()[0]
        assert len(decision.reasoning) == 2
        assert len(decision.uncertainties) == 1
```

## 🔍 Code Review Process

1. **Automated checks** must pass (CI/CD)
2. **At least one approval** required
3. **Address all comments** before merge
4. **Squash commits** if messy history
5. **Update CHANGELOG.md**

### Review Checklist

- [ ] Code follows style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

## 📚 Resources

- [Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [TypeScript Style Guide](https://github.com/airbnb/javascript)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

## 🏆 Recognition

Contributors will be:
- Listed in [AUTHORS.md](AUTHORS.md)
- Mentioned in release notes
- Invited to contributor Discord channel

## 💬 Questions?

- **Discord**: [Join our community](https://discord.gg/agentability)
- **GitHub Discussions**: Ask questions
- **Email**: contributors@agentability.io

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making Agentability better! 🚀
