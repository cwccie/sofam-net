"""
SOFAM-Net CLI — Command-line interface for managing the federated learning platform.

Usage:
    sofam start-coordinator   Start the FL coordinator server
    sofam start-client        Start an FL client
    sofam simulate            Run a multi-org simulation
    sofam dashboard           Launch the web dashboard
    sofam evaluate            Evaluate a trained model
"""

from __future__ import annotations

import json
import logging

import click
import numpy as np


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="sofam-net")
def main():
    """SOFAM-Net: Federated Learning for Cross-Organization Threat Detection."""
    pass


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=8080, type=int, help="Port number")
@click.option("--strategy", default="fedavg", type=click.Choice(["fedavg", "fedprox", "scaffold"]))
@click.option("--num-rounds", default=10, type=int, help="Number of FL rounds")
@click.option("--min-clients", default=2, type=int, help="Minimum clients per round")
@click.option("--verbose", "-v", is_flag=True)
def start_coordinator(host, port, strategy, num_rounds, min_clients, verbose):
    """Start the FL coordinator server."""
    setup_logging(verbose)
    from sofam_net.api.server import create_api
    from sofam_net.coordinator.server import CoordinatorServer

    coordinator = CoordinatorServer(
        strategy=strategy,
        num_rounds=num_rounds,
        min_clients=min_clients,
        min_fit_clients=min_clients,
    )

    app = create_api(coordinator=coordinator)
    click.echo(f"Starting SOFAM-Net coordinator on {host}:{port}")
    click.echo(f"Strategy: {strategy} | Rounds: {num_rounds} | Min clients: {min_clients}")
    app.run(host=host, port=port, debug=verbose)


@main.command()
@click.option("--coordinator-url", default="http://localhost:8080", help="Coordinator URL")
@click.option("--org-name", default="default", help="Organization name")
@click.option("--client-id", default=None, help="Client ID (auto-generated if not set)")
@click.option("--epochs", default=5, type=int, help="Local training epochs")
@click.option("--batch-size", default=32, type=int)
@click.option("--learning-rate", default=0.01, type=float)
@click.option("--verbose", "-v", is_flag=True)
def start_client(coordinator_url, org_name, client_id, epochs, batch_size, learning_rate, verbose):
    """Start an FL client that connects to the coordinator."""
    setup_logging(verbose)
    from sofam_net.client.trainer import FederatedClient, TrainingConfig

    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    client = FederatedClient(
        client_id=client_id,
        org_name=org_name,
        config=config,
    )
    click.echo(f"SOFAM-Net client started: {client.client_id} (org={org_name})")
    click.echo(f"Coordinator: {coordinator_url}")
    click.echo("Client is ready. Use the API or simulation mode to drive training.")


@main.command()
@click.option("--num-orgs", default=3, type=int, help="Number of organizations")
@click.option("--num-rounds", default=10, type=int, help="Number of FL rounds")
@click.option("--num-samples", default=10000, type=int, help="Total data samples")
@click.option("--num-features", default=20, type=int, help="Number of features")
@click.option("--num-classes", default=5, type=int, help="Number of classes")
@click.option("--strategy", default="fedavg", type=click.Choice(["fedavg", "fedprox", "scaffold"]))
@click.option("--alpha", default=0.5, type=float, help="Dirichlet alpha (lower=more non-IID)")
@click.option("--enable-dp", is_flag=True, help="Enable differential privacy")
@click.option("--epsilon", default=1.0, type=float, help="DP epsilon per round")
@click.option("--drop-prob", default=0.0, type=float, help="Client drop probability")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--output", "-o", default=None, help="Save results to JSON file")
def simulate(num_orgs, num_rounds, num_samples, num_features, num_classes, strategy,
             alpha, enable_dp, epsilon, drop_prob, seed, verbose, output):
    """Run a multi-organization federated learning simulation."""
    setup_logging(verbose)
    from sofam_net.simulation.runner import SimulationConfig, SimulationRunner

    config = SimulationConfig(
        num_samples=num_samples,
        num_features=num_features,
        num_classes=num_classes,
        num_orgs=num_orgs,
        num_rounds=num_rounds,
        strategy=strategy,
        dirichlet_alpha=alpha,
        enable_dp=enable_dp,
        epsilon_per_round=epsilon,
        drop_probability=drop_prob,
        seed=seed,
        verbose=verbose,
    )

    runner = SimulationRunner(config)
    result = runner.run()

    click.echo("\n" + "=" * 50)
    click.echo("SIMULATION RESULTS")
    click.echo("=" * 50)
    click.echo(f"Final Test Accuracy: {result.final_accuracy:.4f}")
    click.echo(f"Final Test Loss:     {result.final_loss:.4f}")
    click.echo(f"Total Time:          {result.total_time:.2f}s")
    click.echo(f"Converged:           {result.converged}")
    if result.converged:
        click.echo(f"Convergence Round:   {result.convergence_round}")
    click.echo("\nPer-Org Accuracies:")
    for org, acc in result.org_accuracies.items():
        click.echo(f"  {org}: {acc:.4f}")
    if result.privacy_spent:
        click.echo("\nPrivacy Budget Spent:")
        for org, eps in result.privacy_spent.items():
            click.echo(f"  {org}: epsilon={eps:.4f}")

    if output:
        out_data = {
            "final_accuracy": result.final_accuracy,
            "final_loss": result.final_loss,
            "total_time": result.total_time,
            "converged": result.converged,
            "convergence_round": result.convergence_round,
            "org_accuracies": result.org_accuracies,
            "privacy_spent": result.privacy_spent,
            "round_metrics": result.round_metrics,
        }
        with open(output, "w") as f:
            json.dump(out_data, f, indent=2)
        click.echo(f"\nResults saved to {output}")


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=5000, type=int, help="Port number")
@click.option("--verbose", "-v", is_flag=True)
def dashboard(host, port, verbose):
    """Launch the web monitoring dashboard."""
    setup_logging(verbose)
    from sofam_net.coordinator.server import CoordinatorServer
    from sofam_net.dashboard.app import create_app

    coordinator = CoordinatorServer()
    app = create_app(coordinator=coordinator)
    click.echo(f"Starting SOFAM-Net dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=verbose)


@main.command()
@click.option("--model-path", default=None, help="Path to saved model parameters (JSON)")
@click.option("--num-samples", default=5000, type=int, help="Test samples to generate")
@click.option("--num-features", default=20, type=int)
@click.option("--num-classes", default=5, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--verbose", "-v", is_flag=True)
def evaluate(model_path, num_samples, num_features, num_classes, seed, verbose):
    """Evaluate a trained model on synthetic test data."""
    setup_logging(verbose)
    from sofam_net.data.synthetic import SyntheticIDSGenerator
    from sofam_net.models.mlp import MLPClassifier

    model = MLPClassifier(
        input_dim=num_features,
        hidden_dim=64,
        num_classes=num_classes,
        seed=seed,
    )

    if model_path:
        with open(model_path) as f:
            params_data = json.load(f)
        params = [np.array(p) for p in params_data]
        model.set_parameters(params)
        click.echo(f"Loaded model from {model_path}")
    else:
        click.echo("No model path provided — evaluating random initialization")

    gen = SyntheticIDSGenerator(
        num_features=num_features,
        num_classes=num_classes,
        seed=seed + 1,  # Different seed for test data
    )
    dataset = gen.generate(num_samples=num_samples)

    metrics = model.evaluate(dataset.X, dataset.y)
    click.echo(f"\nEvaluation Results ({num_samples} samples):")
    click.echo(f"  Loss:     {metrics['loss']:.4f}")
    click.echo(f"  Accuracy: {metrics['accuracy']:.4f}")
    for key, val in metrics.items():
        if key.startswith("precision") or key.startswith("recall"):
            click.echo(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
