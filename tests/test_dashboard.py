"""Tests for the web dashboard."""


from sofam_net.coordinator.server import CoordinatorServer
from sofam_net.dashboard.app import create_app


class TestDashboard:
    def setup_method(self):
        coordinator = CoordinatorServer()
        coordinator.register_client("c1", "OrgA", 500)
        self.app = create_app(coordinator=coordinator)
        self.client = self.app.test_client()

    def test_index(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert b"SOFAM-Net" in resp.data

    def test_health(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"

    def test_api_status(self):
        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "state" in data
        assert "registered_clients" in data

    def test_api_privacy_no_accountant(self):
        resp = self.client.get("/api/privacy")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "budgets" in data

    def test_api_history(self):
        resp = self.client.get("/api/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "rounds" in data

    def test_api_clients(self):
        resp = self.client.get("/api/clients")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "c1" in data["clients"]
