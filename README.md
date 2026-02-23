<div align="center">

# SOFAM-Net

**Secure Optimized Federated Autonomous Model for Network Threat Detection**

[![CI](https://github.com/cwccie/sofam-net/actions/workflows/ci.yml/badge.svg)](https://github.com/cwccie/sofam-net/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A production federated learning platform enabling cross-organization collaborative
threat detection — without sharing sensitive network data.*

[Quickstart](#quickstart) · [Architecture](#architecture) · [Features](#features) · [API Reference](#api-reference) · [Privacy Guarantees](#privacy-guarantees) · [Research](#research-context)

</div>

---

## The Problem

Organizations face increasingly sophisticated network threats, yet each organization's
intrusion detection system (IDS) only learns from its own limited traffic. Sharing raw
network data between organizations is a non-starter — it exposes sensitive infrastructure
details, employee behavior, and potential compliance violations.

**The result**: each organization's threat detection model is trained in isolation,
blind to attack patterns seen by others.

## The Solution

SOFAM-Net enables multiple organizations to **collaboratively train a shared threat
detection model** while keeping all network data local. Using federated learning with
differential privacy and secure aggregation, organizations contribute model improvements
— never raw data — to a global IDS model that benefits everyone.

```
                          ┌─────────────────┐
                          │  Global Model    │
                          │  (Coordinator)   │
                          └───────┬─────────┘
                     ┌────────────┼────────────┐
                     │            │            │
              DP Noise +    DP Noise +    DP Noise +
              Secure Agg    Secure Agg    Secure Agg
                     │            │            │
              ┌──────┴──┐  ┌─────┴───┐  ┌─────┴───┐
              │  Org A   │  │  Org B   │  │  Org C   │
              │  Local   │  │  Local   │  │  Local   │
              │ Training │  │ Training │  │ Training │
              │          │  │          │  │          │
              │ ███ Data │  │ ██ Data  │  │ ████ Data│
              └──────────┘  └──────────┘  └──────────┘
```

## Features

### Federated Learning Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **FedAvg** | Weighted parameter averaging | IID or mildly heterogeneous data |
| **FedProx** | Proximal regularization to handle drift | Non-IID data across organizations |
| **SCAFFOLD** | Control variates for variance reduction | Highly heterogeneous distributions |

### Differential Privacy

- **Gaussian Mechanism**: (ε,δ)-DP with calibrated Gaussian noise
- **Laplace Mechanism**: ε-DP with Laplace noise
- **RDP Accounting**: Rényi DP for tight multi-round composition
- **Per-Org Budgets**: Each organization has an independent privacy budget
  with real-time tracking and automatic participation cutoff

### Secure Aggregation

- **Pairwise Masking**: Clients mask updates so the coordinator never sees individual contributions
- **Threshold Reconstruction**: Tolerates dropped clients through mask recovery
- **Commitment Verification**: SHA-256 commitments for mask integrity

### Byzantine Robustness

- **Krum / Multi-Krum**: Selects updates closest to neighbors, filtering outliers
- **Trimmed Mean**: Coordinate-wise trimming of extreme values

### IDS Models (NumPy, No Framework Required)

- **MLP Classifier**: Multi-layer perceptron for flow classification
- **Autoencoder**: Anomaly detector using reconstruction error thresholding
- **Flow GNN**: Graph neural network for network topology-aware detection

### Data & Simulation

- **Synthetic Generator**: CICIDS2017-like feature distributions with realistic attack signatures
- **Dirichlet Splitter**: Non-IID data partitioning to simulate real-world heterogeneity
- **Simulation Harness**: Full FL experiments with configurable orgs, data skew, network conditions

### Monitoring

- **Web Dashboard**: Real-time training progress, privacy budget consumption, client health
- **REST API**: Programmatic coordinator management and model retrieval

## Quickstart

### Option 1: Docker Compose (Recommended)

```bash
git clone https://github.com/cwccie/sofam-net.git
cd sofam-net

# Launch coordinator + 3 clients + dashboard
docker-compose up --build

# Dashboard: http://localhost:5000
# API:       http://localhost:8080
```

### Option 2: Local Simulation

```bash
# Install
pip install -e .

# Run a 3-organization simulation with 10 rounds
sofam simulate \
  --num-orgs 3 \
  --num-rounds 10 \
  --num-samples 10000 \
  --strategy fedavg \
  --alpha 0.5 \
  -v

# With differential privacy
sofam simulate \
  --num-orgs 5 \
  --num-rounds 20 \
  --enable-dp \
  --epsilon 1.0 \
  --strategy fedprox \
  -v -o results.json
```

### Option 3: Python API

```python
from sofam_net.simulation import SimulationRunner, SimulationConfig

config = SimulationConfig(
    num_orgs=3,
    num_rounds=10,
    num_samples=10000,
    num_features=20,
    num_classes=5,
    strategy="fedavg",
    dirichlet_alpha=0.5,       # Non-IID data
    enable_dp=True,            # Differential privacy
    epsilon_per_round=1.0,
    total_epsilon_budget=10.0,
)

result = SimulationRunner(config).run()
print(f"Accuracy: {result.final_accuracy:.4f}")
print(f"Privacy spent: {result.privacy_spent}")
```

## Architecture

```
sofam-net/
├── src/sofam_net/
│   ├── coordinator/     # FL server — strategy selection, round management, convergence
│   │   ├── server.py    #   CoordinatorServer: registration, aggregation, model distribution
│   │   └── strategy.py  #   FedAvg, FedProx, SCAFFOLD implementations
│   ├── client/          # FL client SDK — local training, serialization, heartbeat
│   │   └── trainer.py   #   FederatedClient: SGD, gradient compression, proximal term
│   ├── privacy/         # Differential privacy engine
│   │   ├── mechanisms.py #  Gaussian & Laplace noise mechanisms
│   │   └── accountant.py #  RDP accounting, per-org budget tracking
│   ├── aggregation/     # Secure & robust aggregation
│   │   ├── secure.py    #   Pairwise masking, threshold reconstruction
│   │   └── byzantine.py #   Krum, trimmed mean
│   ├── models/          # Pre-built IDS models (pure NumPy)
│   │   ├── mlp.py       #   Multi-layer perceptron classifier
│   │   ├── autoencoder.py #  Anomaly detector
│   │   └── gnn.py       #   Graph neural network for flow analysis
│   ├── data/            # Data loading & generation
│   │   ├── synthetic.py #   CICIDS2017-like generator (76 features, 11 classes)
│   │   ├── splitter.py  #   Dirichlet non-IID partitioning
│   │   └── loader.py    #   Unified CSV/NumPy/synthetic loader
│   ├── dashboard/       # Flask web dashboard
│   ├── simulation/      # Multi-org simulation harness
│   ├── api/             # REST API for coordinator management
│   └── cli.py           # Click CLI: simulate, start-coordinator, dashboard, evaluate
├── tests/               # 50+ comprehensive tests
├── docker-compose.yml   # Multi-container deployment
├── pyproject.toml       # Hatchling build configuration
└── .github/workflows/   # CI: ruff + pytest on Python 3.11/3.12
```

## CLI Reference

```bash
sofam --help                    # Show all commands
sofam simulate --help           # Simulation options
sofam start-coordinator --help  # Coordinator server options
sofam start-client --help       # Client options
sofam dashboard --help          # Dashboard options
sofam evaluate --help           # Model evaluation options
```

## API Reference

### Coordinator REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/register` | Register a new FL client |
| `POST` | `/api/v1/unregister` | Remove a client |
| `POST` | `/api/v1/heartbeat` | Client heartbeat |
| `GET`  | `/api/v1/model` | Get global model parameters |
| `POST` | `/api/v1/model/init` | Initialize global model |
| `POST` | `/api/v1/update` | Submit client model update |
| `POST` | `/api/v1/round` | Trigger aggregation round |
| `GET`  | `/api/v1/status` | Coordinator status |
| `GET`  | `/api/v1/privacy` | Privacy budget status |
| `GET`  | `/health` | Health check |

### Example: Register a Client

```bash
curl -X POST http://localhost:8080/api/v1/register \
  -H "Content-Type: application/json" \
  -d '{"client_id": "org-alpha-01", "org_name": "Alpha Corp", "num_samples": 5000}'
```

## Privacy Guarantees

SOFAM-Net provides formal differential privacy guarantees:

1. **Local Clipping**: Each client's model update is clipped to bound sensitivity
2. **Calibrated Noise**: Gaussian noise scaled to the clipping norm and target ε
3. **RDP Composition**: Privacy loss is tracked across rounds using Rényi DP,
   providing tighter bounds than basic composition
4. **Per-Org Budgets**: Each organization has an independent privacy budget;
   once exhausted, the organization stops contributing

| Component | Guarantee |
|-----------|-----------|
| Single round | (ε, δ)-DP per Gaussian mechanism |
| Multi-round | Composed via RDP with optimal order selection |
| Secure aggregation | Coordinator sees only aggregate (not individual updates) |
| Budget exhaustion | Automatic cutoff when ε-budget is spent |

### Privacy-Utility Tradeoff

```
Higher ε (less privacy) → Lower noise → Better model accuracy
Lower ε (more privacy)  → Higher noise → Reduced model accuracy
```

The simulation harness allows systematic exploration of this tradeoff.

## Comparison with Centralized Approaches

| Aspect | Centralized IDS | SOFAM-Net (Federated) |
|--------|----------------|----------------------|
| Data sharing | All data sent to central server | No data leaves the organization |
| Privacy | None (raw data exposed) | Formal DP guarantees per round |
| Compliance | GDPR/HIPAA/CCPA risk | Compliant by design |
| Attack diversity | Limited to one org's traffic | Learns from all participating orgs |
| Single point of failure | Central server | Distributed (coordinator only aggregates) |
| Communication cost | Transfer all raw data | Transfer only model updates |
| Byzantine resilience | N/A | Krum, trimmed mean filtering |

## Research Context

SOFAM-Net is the production implementation of **SOFAM** (Secure Optimized Federated
Autonomous Model), a research framework developed at the **University of Oklahoma**
as part of PhD research in AI and cybersecurity.

### Key Research Contributions

1. **Federated IDS**: Enabling collaborative intrusion detection without data sharing
2. **Privacy-Preserving Aggregation**: Combining differential privacy with secure
   aggregation for defense-in-depth privacy
3. **Non-IID Robustness**: Addressing the statistical heterogeneity inherent in
   network traffic across different organizations
4. **Byzantine Resilience**: Protecting the federated model against adversarial
   participants submitting poisoned updates

### Related Work

- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017) — FedAvg
- Li et al., "Federated Optimization in Heterogeneous Networks" (2020) — FedProx
- Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (2020)
- Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (2017)
- Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (2017) — Krum
- Mironov, "Rényi Differential Privacy" (2017) — RDP accounting

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run with coverage
pytest --cov=sofam_net --cov-report=term-missing

# Lint
ruff check src/ tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Corey A. Wade**
PhD Candidate, University of Oklahoma
CISSP | Retired CCIE

---

<div align="center">
<i>Enabling collaborative defense without compromising data sovereignty.</i>
</div>
