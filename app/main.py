from fastapi import FastAPI

app = FastAPI(title="Customer Churn Predictor")

@app.get("/health")
def health():
    return {"status": "ok", "model": "not-trained"}
