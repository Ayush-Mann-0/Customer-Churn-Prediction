# API Usage Guide

This guide explains how to use the Customer Churn Prediction API for making predictions.

## API Overview

The API is built with FastAPI and provides endpoints for:
- Health checks
- Single customer churn prediction
- Batch customer churn predictions

## Starting the API Server

Before using the API, make sure you have:
1. Trained the model (`python train_model.py`)
2. Started the API server:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### 1. Root Endpoint

**GET /**

Checks if the API is running.

Example:
```bash
curl -X 'GET' 'http://localhost:8000/' -H 'accept: application/json'
```

Response:
```json
{
  "message": "Customer Churn Prediction API is running"
}
```

### 2. Health Check

**GET /health**

Checks the health status of the API and if the model is loaded.

Example:
```bash
curl -X 'GET' 'http://localhost:8000/health' -H 'accept: application/json'
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Single Prediction

**POST /predict**

Predicts churn for a single customer.

Example:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}'
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.1234
}
```

Where:
- `prediction`: 0 = No churn, 1 = Churn
- `probability`: Probability of churn (0-1)

### 4. Batch Prediction

**POST /predict/batch**

Predicts churn for multiple customers in a single request.

Example:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict/batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "customers": [
    {
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 12,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "DSL",
      "OnlineSecurity": "Yes",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 29.85,
      "TotalCharges": 29.85
    },
    {
      "gender": "Male",
      "SeniorCitizen": 1,
      "Partner": "No",
      "Dependents": "No",
      "tenure": 5,
      "PhoneService": "Yes",
      "MultipleLines": "Yes",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "Yes",
      "StreamingMovies": "Yes",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 104.5,
      "TotalCharges": 500.25
    }
  ]
}'
```

Response:
```json
{
  "predictions": [0, 1],
  "probabilities": [0.1234, 0.8765]
}
```

## Field Descriptions

### Customer Information
- `gender`: Customer's gender ("Male" or "Female")
- `SeniorCitizen`: Whether the customer is a senior citizen (0 or 1)
- `Partner`: Whether the customer has a partner ("Yes" or "No")
- `Dependents`: Whether the customer has dependents ("Yes" or "No")

### Services
- `tenure`: Number of months the customer has been with the company (integer)
- `PhoneService`: Whether the customer has phone service ("Yes" or "No")
- `MultipleLines`: Whether the customer has multiple lines ("Yes", "No", or "No phone service")
- `InternetService`: Customer's internet service provider ("DSL", "Fiber optic", or "No")
- `OnlineSecurity`: Whether the customer has online security ("Yes", "No", or "No internet service")
- `OnlineBackup`: Whether the customer has online backup ("Yes", "No", or "No internet service")
- `DeviceProtection`: Whether the customer has device protection ("Yes", "No", or "No internet service")
- `TechSupport`: Whether the customer has tech support ("Yes", "No", or "No internet service")
- `StreamingTV`: Whether the customer has streaming TV ("Yes", "No", or "No internet service")
- `StreamingMovies`: Whether the customer has streaming movies ("Yes", "No", or "No internet service")

### Billing
- `Contract`: The customer's contract type ("Month-to-month", "One year", or "Two year")
- `PaperlessBilling`: Whether the customer has paperless billing ("Yes" or "No")
- `PaymentMethod`: The customer's payment method (e.g., "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)")
- `MonthlyCharges`: The amount charged to the customer monthly (float)
- `TotalCharges`: The total amount charged to the customer (float)

## Response Format

### Single Prediction
```json
{
  "prediction": 0,
  "probability": 0.1234
}
```

### Batch Prediction
```json
{
  "predictions": [0, 1, 0],
  "probabilities": [0.1234, 0.8765, 0.2345]
}
```

## Error Handling

The API will return appropriate HTTP status codes for different error conditions:

- `200`: Success
- `422`: Validation error (missing or incorrect fields)
- `500`: Internal server error (e.g., model not loaded)

Example error response:
```json
{
  "detail": "Prediction error: Model not loaded"
}
```

## Using Python Requests

You can also use the API with Python requests:

```python
import requests
import json

# Single prediction
url = "http://localhost:8000/predict"
data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

response = requests.post(url, json=data)
print(response.json())
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

These interfaces allow you to test the API directly from your browser and provide detailed information about each endpoint.