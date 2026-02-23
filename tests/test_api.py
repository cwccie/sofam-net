"""Tests for the REST API."""


import numpy as np

from sofam_net.api.server import create_api
from sofam_net.coordinator.server import CoordinatorServer


class TestAPI:
    def setup_method(self):
        self.coordinator = CoordinatorServer(min_clients=2, min_fit_clients=2)
        self.app = create_api(coordinator=self.coordinator)
        self.client = self.app.test_client()

    def test_health(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "healthy"

    def test_register_client(self):
        resp = self.client.post("/api/v1/register", json={
            "client_id": "c1", "org_name": "OrgA", "num_samples": 500,
        })
        assert resp.status_code == 201
        assert resp.get_json()["status"] == "registered"

    def test_register_missing_id(self):
        resp = self.client.post("/api/v1/register", json={"org_name": "OrgA"})
        assert resp.status_code == 400

    def test_heartbeat(self):
        self.client.post("/api/v1/register", json={"client_id": "c1", "org_name": "OrgA"})
        resp = self.client.post("/api/v1/heartbeat", json={"client_id": "c1"})
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_get_model_empty(self):
        resp = self.client.get("/api/v1/model")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["parameters"] is None

    def test_init_model(self):
        params = [np.random.randn(10, 5).tolist(), np.zeros(5).tolist()]
        resp = self.client.post("/api/v1/model/init", json={"parameters": params})
        assert resp.status_code == 200

    def test_submit_update(self):
        self.coordinator.register_client("c1", "OrgA", 500)
        params = [np.random.randn(5).tolist()]
        resp = self.client.post("/api/v1/update", json={
            "client_id": "c1",
            "parameters": params,
            "num_samples": 500,
            "metrics": {"loss": 0.5},
        })
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "accepted"

    def test_status(self):
        resp = self.client.get("/api/v1/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "state" in data

    def test_unregister(self):
        self.client.post("/api/v1/register", json={"client_id": "c1", "org_name": "OrgA"})
        resp = self.client.post("/api/v1/unregister", json={"client_id": "c1"})
        assert resp.status_code == 200
        assert resp.get_json()["success"]

    def test_run_round_insufficient(self):
        resp = self.client.post("/api/v1/round")
        assert resp.status_code == 409
