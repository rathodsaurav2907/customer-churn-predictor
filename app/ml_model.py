import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

class MLModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_names = ['age', 'tenure', 'monthly_charges', 'total_charges', 'contracts_encoded', 'internet_type_encoded']
        self.model_path = Path(__file__).parent.parent / 'data' / 'model.pkl'
        self.scaler_path = Path(__file__).parent.parent / 'data' / 'scaler.pkl'
        self.encoders_path = Path(__file__).parent.parent / 'data' / 'encoders.pkl'
        
        self.model_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'training_samples': 0
        }
        
        self.load_or_train_model()

    def load_or_train_model(self):
        if (self.model_path.exists() and 
            self.scaler_path.exists() and 
            self.encoders_path.exists()):
            self.load_model()
        else:
            self.train_model()

    def train_model(self):
        data_path = Path(__file__).parent.parent / 'data' / 'sample_data.csv'
        
        if not data_path.exists():
            print(f"Creating sample data at {data_path}")
            self.create_sample_data(data_path)
        
        df = pd.read_csv(data_path)
        
        X = df.drop('churn', axis=1)
        y = df['churn'].astype(int)
        
        # Encode categorical variables
        self.encoders['contracts'] = LabelEncoder()
        self.encoders['internet_type'] = LabelEncoder()
        
        X['contracts'] = self.encoders['contracts'].fit_transform(X['contracts'])
        X['internet_type'] = self.encoders['internet_type'].fit_transform(X['internet_type'])
        
        X.columns = self.feature_names
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.model_metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        self.model_metrics['precision'] = float(precision_score(y_test, y_pred))
        self.model_metrics['recall'] = float(recall_score(y_test, y_pred))
        self.model_metrics['f1_score'] = float(f1_score(y_test, y_pred))
        self.model_metrics['training_samples'] = len(X_train)
        
        # Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.encoders, self.encoders_path)
        
        print(f"Model trained with accuracy: {self.model_metrics['accuracy']:.4f}")

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.encoders = joblib.load(self.encoders_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.train_model()

    def create_sample_data(self, data_path):
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'tenure': np.random.randint(0, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 120, n_samples),
            'total_charges': np.random.uniform(100, 5000, n_samples),
            'contracts': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'internet_type': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
        }
        
        df = pd.DataFrame(data)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Sample data created at {data_path}")

    def predict(self, age: int, tenure: int, monthly_charges: float, 
                total_charges: float, contracts: str, internet_type: str):
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            contracts_encoded = self.encoders['contracts'].transform([contracts])[0]
            internet_encoded = self.encoders['internet_type'].transform([internet_type])[0]
            
            X = np.array([[age, tenure, monthly_charges, total_charges, 
                          contracts_encoded, internet_encoded]])
            
            X_scaled = self.scaler.transform(X)
            
            churn_prob = float(self.model.predict_proba(X_scaled)[0][1])
            churn_prediction = bool(self.model.predict(X_scaled)[0])
            
            if churn_prob > 0.7:
                risk_level = 'HIGH'
            elif churn_prob > 0.4:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            confidence = max(self.model.predict_proba(X_scaled)[0])
            
            return {
                'churn_probability': churn_prob,
                'churn_prediction': churn_prediction,
                'risk_level': risk_level,
                'confidence': float(confidence)
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            raise

    def get_feature_importance(self):
        if self.model is None:
            return {}
        
        importance = {}
        for feature, coef in zip(self.feature_names, self.model.coef_[0]):
            importance[feature] = float(abs(coef))
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_model_info(self):
        return {
            'model_name': 'Customer Churn Predictor',
            'model_type': 'Logistic Regression',
            'accuracy': self.model_metrics['accuracy'],
            'precision': self.model_metrics['precision'],
            'recall': self.model_metrics['recall'],
            'f1_score': self.model_metrics['f1_score'],
            'feature_importance': self.get_feature_importance(),
            'training_samples': self.model_metrics['training_samples'],
            'features': self.feature_names
        }


# Global model instance
ml_model = MLModel()
