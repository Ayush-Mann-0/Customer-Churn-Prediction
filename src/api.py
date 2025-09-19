"""
FastAPI application for customer churn prediction.
This module creates a REST API endpoint for making predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import List
import warnings
warnings.filterwarnings('ignore')


# Pydantic model for input data
class CustomerInput(BaseModel):
    """
    Schema for customer input data.
    
    Documentation Reference:
    - FastAPI data validation: https://fastapi.tiangolo.com/tutorial/body/
    - Pydantic models: https://fastapi.tiangolo.com/tutorial/body/
    """
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# Pydantic model for batch input
class BatchCustomerInput(BaseModel):
    customers: List[CustomerInput]


# Pydantic model for response
class PredictionResponse(BaseModel):
    prediction: int
    probability: float


class BatchPredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]


# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using a trained machine learning model",
    version="1.0.0"
)

# Global variable to store the model
model = None


def load_model_once():
    """
    Load the model once when the application starts.
    
    Documentation Reference:
    - joblib.load: https://joblib.readthedocs.io/en/latest/generated/joblib.load.html
    """
    global model
    if model is None:
        # Try multiple possible model paths
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib"),
            os.path.join(os.getcwd(), "models", "model.joblib"),
            "models/model.joblib"
        ]
        
        model_loaded = False
        for model_path in possible_paths:
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    print(f"Model loaded successfully from {model_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load model from {model_path}: {e}")
        
        if not model_loaded:
            raise FileNotFoundError(
                f"Model file not found in any of these locations: {possible_paths}"
            )


@app.on_event("startup")
async def startup_event():
    """
    Load the model when the application starts.
    """
    try:
        load_model_once()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will start but predictions will not work until model is loaded")


@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: CustomerInput):
    """
    Predict churn for a single customer.
    
    Parameters:
    input_data (CustomerInput): Customer data for prediction
    
    Returns:
    PredictionResponse: Prediction result with churn probability
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]  # Probability of churn (class 1)
        
        return PredictionResponse(prediction=int(prediction), probability=float(probability))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchCustomerInput):
    """
    Predict churn for a batch of customers.
    
    Parameters:
    input_data (BatchCustomerInput): List of customers for prediction
    
    Returns:
    BatchPredictionResponse: Batch prediction results with churn probabilities
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([customer.dict() for customer in input_data.customers])
        
        # Make predictions
        predictions = model.predict(data).tolist()
        probabilities = model.predict_proba(data)[:, 1].tolist()  # Probabilities of churn (class 1)
        
        return BatchPredictionResponse(
            predictions=[int(pred) for pred in predictions],
            probabilities=[float(prob) for prob in probabilities]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# For testing purposes
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)