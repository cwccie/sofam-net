"""
Flask Web Dashboard for SOFAM-Net.

Provides real-time monitoring of:
- Training progress across rounds
- Model performance per organization
- Privacy budget consumption
- Client health status
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template


def create_app(coordinator=None, accountant=None) -> Flask:
    """Create the Flask dashboard application."""
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

    # Store references
    app.config["coordinator"] = coordinator
    app.config["accountant"] = accountant

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/status")
    def api_status():
        coord = app.config.get("coordinator")
        if coord is None:
            return jsonify({"error": "No coordinator configured"}), 503
        return jsonify(coord.get_status())

    @app.route("/api/privacy")
    def api_privacy():
        acct = app.config.get("accountant")
        if acct is None:
            return jsonify({"budgets": {}})
        return jsonify({"budgets": acct.get_all_budgets()})

    @app.route("/api/history")
    def api_history():
        coord = app.config.get("coordinator")
        if coord is None:
            return jsonify({"rounds": []})
        return jsonify({
            "rounds": [
                {
                    "round": r.round_num,
                    "num_clients": r.num_clients,
                    "total_samples": r.total_samples,
                    "metrics": r.metrics,
                    "duration": r.duration_seconds,
                    "converged": r.converged,
                }
                for r in coord.round_results
            ]
        })

    @app.route("/api/clients")
    def api_clients():
        coord = app.config.get("coordinator")
        if coord is None:
            return jsonify({"clients": {}})
        return jsonify({
            "clients": {
                cid: {
                    "org": c.org_name,
                    "samples": c.num_samples,
                    "rounds": c.rounds_participated,
                    "last_heartbeat": c.last_heartbeat,
                }
                for cid, c in coord.clients.items()
            }
        })

    @app.route("/health")
    def health():
        return jsonify({"status": "healthy", "service": "sofam-net-dashboard"})

    return app
