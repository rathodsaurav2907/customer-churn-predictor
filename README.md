# Customer Churn Predictor

FastAPI environment for an explainable churn-classification model. It starts at `http://localhost:8003`; the health endpoint is `/health`.

Run `docker compose up --build`. Put raw data in `data/`, training code in `app/`, and the exported model in `models/`.

## Suggested milestones

- reproducible preprocessing pipeline and train/evaluate commands
- precision, recall, F1, ROC-AUC and confusion-matrix report
- prediction endpoint with feature validation and model version
- EDA notebook with non-sensitive sample data
