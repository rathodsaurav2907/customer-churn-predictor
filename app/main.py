from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from app.models import ChurnPredictionRequest, ChurnPredictionResponse, ModelInfoResponse, HealthResponse
from app.ml_model import ml_model

app = FastAPI(
    title="Customer Churn Predictor API",
    description="ML API for predicting customer churn probability",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "service": "Customer Churn Predictor",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/predict", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest):
    try:
        result = ml_model.predict(
            age=request.age,
            tenure=request.tenure,
            monthly_charges=request.monthly_charges,
            total_charges=request.total_charges,
            contracts=request.contracts,
            internet_type=request.internet_type
        )
        
        return ChurnPredictionResponse(
            churn_probability=result['churn_probability'],
            churn_prediction=result['churn_prediction'],
            risk_level=result['risk_level'],
            confidence=result['confidence']
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/api/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    try:
        info = ml_model.get_model_info()
        return ModelInfoResponse(
            model_name=info['model_name'],
            model_type=info['model_type'],
            accuracy=info['accuracy'],
            precision=info['precision'],
            recall=info['recall'],
            f1_score=info['f1_score'],
            feature_importance=info['feature_importance'],
            training_samples=info['training_samples'],
            features=info['features']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/api/retrain")
async def retrain_model():
    try:
        ml_model.train_model()
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
