# Customer Churn Predictor API

An explainable Machine Learning service and REST API for customer churn risk prediction, built with FastAPI, scikit-learn, and Pandas.

Part of the **[Microservices & ML Data Platform Ecosystem](../ECOSYSTEM.md)**.

---

## 🌟 Overview

The Customer Churn Predictor serves real-time probability estimates for customer attrition based on service utilization, tenure, monthly spend, and contract patterns. It consumes behavioral aggregates produced by the **[Service Analytics Dashboard](../service-analytics-dashboard/)** and alerts transactional platforms like **[PunctureWala](../puncturewala/)** to trigger customer retention interventions.

---

## 🛠️ Tech Stack

* **Language & Framework:** Python 3.12, FastAPI, Uvicorn
* **Machine Learning:** scikit-learn (Logistic Regression, StandardScaler, LabelEncoder), NumPy, Pandas
* **Persistence & Serialization:** Joblib (`model.pkl`, `scaler.pkl`, `encoders.pkl`)
* **Testing:** Pytest, HTTPX
* **Containerization:** Docker (Python 3.12 slim with curl healthcheck)

---

## 🚀 Quick Start

### 1. Using Docker (Recommended)

```bash
# Start API container on port 8003
docker compose up --build -d

# Verify health status
curl http://localhost:8003/health

# View live logs
docker compose logs -f api
```

API Documentation (Swagger UI): `http://localhost:8003/docs`

### 2. Running Locally

```bash
pip install -r requirements.txt

# Run model training & sample data generation
python -c "from app.ml_model import ml_model"

# Start FastAPI server
uvicorn app.main:app --reload --port 8000

# Run test suite
pytest tests/
```

---

## 🔌 API Endpoints

### Health & Model Status
* `GET /health` - API service health and runtime uptime
* `GET /model-info` - Active model metadata, training accuracy, precision, recall, and feature names

### Inference (`/predict`)
* `POST /predict` - Compute churn probability for an individual customer profile:
  ```json
  {
    "age": 42,
    "tenure": 12,
    "monthly_charges": 75.5,
    "total_charges": 906.0,
    "contracts": "Month-to-month",
    "internet_type": "Fiber optic"
  }
  ```
  **Response:**
  ```json
  {
    "churn_prediction": 1,
    "churn_probability": 0.784,
    "risk_level": "High",
    "model_version": "1.0.0",
    "timestamp": "2026-09-20T20:00:00Z"
  }
  ```

---

## 🧪 Testing

```bash
pytest tests/
```
Validates input bounds, categorical validation, probability limits, and mock request execution.
