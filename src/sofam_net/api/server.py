"""
REST API for SOFAM-Net Coordinator.

Provides HTTP endpoints for:
- Client registration and heartbeat
- Model distribution and update submission
- Coordinator management
- Privacy budget queries
"""

from __future__ import annotations

import numpy as np
from flask import Flask, jsonify, request

from sofam_net.coordinator.server import CoordinatorServer
from sofam_net.coordinator.strategy import ClientUpdate
from sofam_net.privacy.accountant import RDPAccountant


def create_api(
    coordinator: CoordinatorServer | None = None,
    accountant: RDPAccountant | None = None,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> Flask:
    """Create the REST API Flask application."""
    app = Flask(__name__)
    app.config["coordinator"] = coordinator or CoordinatorServer()
    app.config["accountant"] = accountant

    # --- Client Management ---

    @app.route("/api/v1/register", methods=["POST"])
    def register_client():
        """Register a new FL client."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        client_id = data.get("client_id")
        org_name = data.get("org_name", "unknown")
        num_samples = data.get("num_samples", 0)

        if not client_id:
            return jsonify({"error": "client_id is required"}), 400

        coord = app.config["coordinator"]
        result = coord.register_client(client_id, org_name, num_samples)
        return jsonify(result), 201

    @app.route("/api/v1/unregister", methods=["POST"])
    def unregister_client():
        """Unregister a client."""
        data = request.get_json()
        if not data or "client_id" not in data:
            return jsonify({"error": "client_id required"}), 400

        coord = app.config["coordinator"]
        success = coord.unregister_client(data["client_id"])
        return jsonify({"success": success})

    @app.route("/api/v1/heartbeat", methods=["POST"])
    def heartbeat():
        """Process client heartbeat."""
        data = request.get_json()
        if not data or "client_id" not in data:
            return jsonify({"error": "client_id required"}), 400

        coord = app.config["coordinator"]
        result = coord.heartbeat(data["client_id"])
        return jsonify(result)

    # --- Model Distribution ---

    @app.route("/api/v1/model", methods=["GET"])
    def get_model():
        """Get current global model parameters and config."""
        coord = app.config["coordinator"]
        return jsonify(coord.get_global_model())

    @app.route("/api/v1/model/init", methods=["POST"])
    def init_model():
        """Initialize global model parameters."""
        data = request.get_json()
        if not data or "parameters" not in data:
            return jsonify({"error": "parameters required"}), 400

        params = [np.array(p) for p in data["parameters"]]
        coord = app.config["coordinator"]
        coord.initialize_global_model(params)
        return jsonify({"status": "initialized"})

    # --- Update Submission ---

    @app.route("/api/v1/update", methods=["POST"])
    def submit_update():
        """Submit a client model update."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        required = ["client_id", "parameters", "num_samples"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"{field} is required"}), 400

        update = ClientUpdate(
            client_id=data["client_id"],
            parameters=[np.array(p) for p in data["parameters"]],
            num_samples=data["num_samples"],
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {}),
        )

        coord = app.config["coordinator"]
        result = coord.submit_update(update)
        return jsonify(result)

    # --- Round Management ---

    @app.route("/api/v1/round", methods=["POST"])
    def run_round():
        """Trigger aggregation for the current round."""
        coord = app.config["coordinator"]
        result = coord.run_round()
        if result is None:
            return jsonify({"error": "Not enough updates for aggregation"}), 409
        return jsonify({
            "round": result.round_num,
            "clients": result.num_clients,
            "samples": result.total_samples,
            "metrics": result.metrics,
            "duration": result.duration_seconds,
            "converged": result.converged,
        })

    # --- Status ---

    @app.route("/api/v1/status", methods=["GET"])
    def status():
        """Get coordinator status."""
        coord = app.config["coordinator"]
        return jsonify(coord.get_status())

    @app.route("/api/v1/privacy", methods=["GET"])
    def privacy():
        """Get privacy budget status."""
        acct = app.config.get("accountant")
        if acct is None:
            return jsonify({"budgets": {}})
        return jsonify({"budgets": acct.get_all_budgets()})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "service": "sofam-net-api"})

    return app
