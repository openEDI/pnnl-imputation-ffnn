"""Unit tests for FastAPI server endpoints."""

import json

from fastapi.testclient import TestClient

from imputation_federate.server import app

client = TestClient(app)


def test_health_check_root():
    """Test GET / endpoint returns HeathCheck model."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "hostname" in data
    assert "host_ip" in data


def test_configure_endpoint(tmp_path, monkeypatch):
    """Test POST /configure endpoint validates and writes configuration files."""
    monkeypatch.chdir(tmp_path)

    payload = {
        "component": {
            "name": "test_imputation",
            "type": "ImputationComponent",
            "parameters": {
                "model_name": "ieee123",
                "deltat": 0.1,
                "end_time": 3600.0,
            },
        },
        "links": [
            {
                "source": "feeder",
                "source_port": "topology",
                "target": "test_imputation",
                "target_port": "topology",
            },
            {
                "source": "sensor_v",
                "source_port": "publication",
                "target": "test_imputation",
                "target_port": "voltages_magnitude",
            },
        ],
    }

    response = client.post("/configure", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "detail" in data

    assert (tmp_path / "static_inputs.json").exists()
    assert (tmp_path / "input_mapping.json").exists()

    with open(tmp_path / "static_inputs.json") as f:
        static = json.load(f)
        assert static["name"] == "test_imputation"
        assert static["model_name"] == "ieee123"

    with open(tmp_path / "input_mapping.json") as f:
        mapping = json.load(f)
        assert mapping["topology"] == "feeder/topology"
        assert mapping["voltages_magnitude"] == "sensor_v/publication"


def test_configure_invalid_payload():
    """Test POST /configure with invalid payload returns 400."""
    response = client.post("/configure", json={"invalid": "payload"})
    assert response.status_code == 422 or response.status_code == 400
