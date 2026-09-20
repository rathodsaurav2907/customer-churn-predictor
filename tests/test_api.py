import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.ml_model import ml_model

client = TestClient(app)

class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["service"] == "Customer Churn Predictor"

class TestPredictionEndpoint:
    def test_predict_churn_low_risk(self):
        request_data = {
            "age": 45,
            "tenure": 50,
            "monthly_charges": 50.0,
            "total_charges": 3000.0,
            "contracts": "Two year",
            "internet_type": "DSL"
        }
        response = client.post("/api/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_level" in data
        assert "confidence" in data
        assert 0 <= data["churn_probability"] <= 1
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_predict_churn_high_risk(self):
        request_data = {
            "age": 25,
            "tenure": 2,
            "monthly_charges": 100.0,
            "total_charges": 200.0,
            "contracts": "Month-to-month",
            "internet_type": "Fiber optic"
        }
        response = client.post("/api/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data

    def test_predict_invalid_contract_type(self):
        request_data = {
            "age": 45,
            "tenure": 50,
            "monthly_charges": 50.0,
            "total_charges": 3000.0,
            "contracts": "Invalid",
            "internet_type": "DSL"
        }
        response = client.post("/api/predict", json=request_data)
        # Should fail due to invalid contract type
        assert response.status_code in [400, 422]

    def test_predict_missing_fields(self):
        request_data = {
            "age": 45,
            "tenure": 50
        }
        response = client.post("/api/predict", json=request_data)
        assert response.status_code == 422  # Validation error

class TestModelInfoEndpoint:
    def test_get_model_info(self):
        response = client.get("/api/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_type" in data
        assert "accuracy" in data
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data
        assert "feature_importance" in data
        assert "training_samples" in data
        assert "features" in data
        
        # Verify metrics are in valid range
        assert 0 <= data["accuracy"] <= 1
        assert 0 <= data["precision"] <= 1
        assert 0 <= data["recall"] <= 1
        assert 0 <= data["f1_score"] <= 1
        assert data["training_samples"] > 0

class TestRetrainEndpoint:
    def test_retrain_model(self):
        response = client.post("/api/retrain")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

class TestMultiplePredictions:
    def test_batch_predictions(self):
        predictions = []
        for i in range(5):
            request_data = {
                "age": 20 + i * 10,
                "tenure": i * 5,
                "monthly_charges": 50.0 + i * 10,
                "total_charges": 1000.0 + i * 500,
                "contracts": "Month-to-month" if i % 2 == 0 else "Two year",
                "internet_type": "DSL" if i % 2 == 0 else "Fiber optic"
            }
            response = client.post("/api/predict", json=request_data)
            assert response.status_code == 200
            predictions.append(response.json())
        
        assert len(predictions) == 5
        for pred in predictions:
            assert "churn_probability" in pred

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
