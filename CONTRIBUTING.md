# Contributing to SOFAM-Net

Thank you for your interest in contributing to SOFAM-Net! This project advances
federated learning research for cybersecurity, and community contributions are welcome.

## Getting Started

### Prerequisites

- Python 3.11+
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/cwccie/sofam-net.git
cd sofam-net

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
sofam --version
pytest
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code in `src/sofam_net/`
- Add tests in `tests/`
- Follow existing code style

### 3. Run Tests

```bash
# All tests
pytest -v

# Specific module
pytest tests/test_privacy.py -v

# With coverage
pytest --cov=sofam_net --cov-report=term-missing
```

### 4. Lint

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### 5. Submit a Pull Request

- Push your branch and open a PR against `main`
- Describe the change, motivation, and testing performed
- Link any related issues

## Code Organization

```
src/sofam_net/
├── coordinator/    # FL server, strategies (FedAvg, FedProx, SCAFFOLD)
├── client/         # Local training, gradient compression
├── privacy/        # DP mechanisms, RDP accounting
├── aggregation/    # Secure aggregation, Byzantine robustness
├── models/         # IDS models (MLP, autoencoder, GNN)
├── data/           # Data loaders, synthetic generation, splitters
├── dashboard/      # Flask web UI
├── simulation/     # Multi-org simulation harness
├── api/            # REST API
└── cli.py          # CLI entry point
```

## Guidelines

### Code Style

- Use type hints for all function signatures
- Write docstrings for public classes and methods
- Keep functions focused and testable
- Use NumPy for numerical operations (no hard framework dependencies)

### Testing

- Every new feature needs tests
- Aim for >90% coverage on new code
- Use fixtures from `tests/conftest.py`
- Tests should be deterministic (use seeds)

### Privacy and Security

- Never log or expose raw model parameters from individual organizations
- All DP implementations must have proper noise calibration
- Secure aggregation must ensure individual updates cannot be recovered
- Be extra careful with any code handling real network data

### Commit Messages

- Use clear, descriptive commit messages
- Reference issue numbers when applicable
- Format: `type: short description` (e.g., `feat: add Krum aggregation`)

## Research Context

SOFAM-Net extends the **SOFAM** (Secure Optimized Federated Autonomous Model)
research from the National University. When contributing to core algorithms
(strategies, privacy mechanisms, robust aggregation), please include references
to relevant academic papers in docstrings.

## Questions?

Open an issue on GitHub or reach out to the maintainer.

---

*By contributing, you agree that your contributions will be licensed under the MIT License.*
