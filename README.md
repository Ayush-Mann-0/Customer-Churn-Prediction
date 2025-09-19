# Customer Churn Prediction System

A complete machine learning solution to predict customer churn for telecommunication companies. This project covers data preprocessing, feature engineering, model training with hyperparameter tuning, API deployment, and experiment tracking.

***

## Features

- End-to-end ML pipeline from raw data handling to production-ready API
- Data cleaning, transformation, and missing value processing
- Feature engineering including numerical scaling and categorical encoding
- Random Forest model training with cross-validation and hyperparameter tuning
- Detailed evaluation with F1-score, confusion matrix, and classification report
- REST API built with FastAPI for real-time and batch predictions
- Experiment tracking and model versioning using MLflow
- Docker support for seamless deployment
- Comprehensive unit tests for all components
- Well-documented code and API usage instructions

***

## Project Structure

```
.
├── dataset.csv                # Sample customer churn dataset
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── startup.sh                 # Setup script for environment and dependencies
├── train_model.py             # Script to train the model
├── check_requirements.py      # Verify package installation
├── README.md                  # Project overview and instructions
├── SETUP_INSTRUCTIONS.md      # Detailed setup guide
├── API_USAGE.md               # API usage examples
├── ARCHITECTURE.md            # Design and architecture details
├── LICENSE                    # MIT License file
├── models/                    # Directory containing saved models
│   └── model.joblib           # Trained model file
├── src/
│   ├── api.py                 # FastAPI application for serving predictions
│   ├── data_preprocessing.py  # Loading and cleaning data functions
│   ├── feature_engineering.py # Feature preprocessing pipelines
│   └── model_training.py      # Model training and evaluation scripts
└── tests/
    └── test_customer_churn.py # Unit tests
```

***

## Technologies Used

- **Python** (3.8+)
- **Pandas**, **NumPy** for data manipulation
- **Scikit-learn** for ML algorithms and preprocessing
- **FastAPI** for building the REST API
- **Joblib** for model serialization
- **MLflow** for experiment tracking and model registry
- **Docker** for containerization
- **Pydantic** for data validation
- **Seaborn**, **Matplotlib** for visualization
- **Gunicorn** & **Uvicorn** for API serving

***

## Setup Instructions

### Prerequisites

- Python 3.8 or later
- pip

### Quick Setup (Recommended)

```bash
chmod +x startup.sh
./startup.sh
```

This script sets up a virtual environment and installs dependencies automatically.

### Manual Setup

1. Create a virtual environment:
   ```bash
   python -m venv churn_env
   source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Docker Setup

Build and run the Docker container:

```bash
docker build -t customer-churn .
docker run -p 8000:8000 customer-churn
```

***

## Usage

### 1. Train the Model

Run the training script:

```bash
python train_model.py
```

This will:

- Load and preprocess the dataset
- Split the data into training and test sets
- Create pipelines for numerical and categorical features
- Train a Random Forest classifier with cross-validation
- Perform hyperparameter tuning using GridSearchCV
- Evaluate model performance with multiple metrics
- Save the trained model to `models/model.joblib`
- Log the experiment in MLflow

### 2. Start the API Server

For development:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

For production (recommended):

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 src.api:app
```

### 3. API Documentation

- Visit [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI
- Visit [http://localhost:8000/redoc](http://localhost:8000/redoc) for ReDoc

### 4. View MLflow Experiments

Run:

```bash
mlflow ui
```

Access the UI at [http://localhost:5000](http://localhost:5000)

### 5. Running Tests

Run unit tests for all components:

```bash
python -m pytest tests/
```

***

## API Endpoints

| Endpoint            | Method | Description                          |
|---------------------|--------|----------------------------------|
| `/`                 | GET    | Check if API is running            |
| `/health`           | GET    | Health check endpoint              |
| `/predict`          | POST   | Predict churn for a single customer|
| `/predict/batch`    | POST   | Predict churn for multiple customers|

***

## Example API Usage

### Single Prediction

```bash
curl -X POST 'http://localhost:8000/predict' \
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

### Batch Prediction

```bash
curl -X POST 'http://localhost:8000/predict/batch' \
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
    }
  ]
}'
```

***

## Dataset Description

The dataset consists of telecommunication customers with the following features:

### Customer Info

- `customerID`: Unique customer identifier
- `gender`: Male/Female
- `SeniorCitizen`: 1 if a senior citizen, else 0
- `Partner`: Yes/No
- `Dependents`: Yes/No

### Services

- `tenure`: Number of months with the company
- `PhoneService`: Yes/No
- `MultipleLines`: Yes/No/No phone service
- `InternetService`: DSL/Fiber optic/No
- `OnlineSecurity`: Yes/No/No internet service
- `OnlineBackup`: Yes/No/No internet service
- `DeviceProtection`: Yes/No/No internet service
- `TechSupport`: Yes/No/No internet service
- `StreamingTV`: Yes/No/No internet service
- `StreamingMovies`: Yes/No/No internet service

### Billing

- `Contract`: Month-to-month/One year/Two year
- `PaperlessBilling`: Yes/No
- `PaymentMethod`: Payment method
- `MonthlyCharges`: Monthly amount
- `TotalCharges`: Total charged amount

### Target

- `Churn`: Yes/No (whether the customer churned)

***

## Model Performance

- Balanced Precision and Recall
- Confusion matrix and classification report included in evaluation

***

## Contribution

Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

***

## Acknowledgments

- Dataset adapted from IBM Watson Telco Customer Churn dataset
- Built using scikit-learn, FastAPI, and other open-source libraries
