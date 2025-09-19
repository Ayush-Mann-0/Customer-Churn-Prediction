# Customer Churn Prediction System

A complete machine learning solution for predicting customer churn in telecommunication companies. This project includes data preprocessing, feature engineering, model training with hyperparameter tuning, API deployment, and experiment tracking.

## Features

- **Complete ML Pipeline**: From raw data to production-ready API
- **Data Preprocessing**: Cleaning, transformation, and handling missing values
- **Feature Engineering**: Numerical scaling and categorical encoding
- **Model Training**: Random Forest classifier with cross-validation and hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics including F1-score, confusion matrix, and classification report
- **API Service**: FastAPI-based REST API for real-time and batch predictions
- **Experiment Tracking**: MLflow integration for logging experiments and model versions
- **Containerization**: Docker support for easy deployment
- **Testing**: Unit tests for all components
- **Documentation**: Comprehensive code documentation and API docs

## Project Structure

```
.
├── dataset.csv                 # Sample customer churn dataset
├── Dockerfile                  # Docker configuration for containerization
├── requirements.txt            # Python dependencies
├── startup.sh                  # Setup script for quick environment creation
├── train_model.py              # Main training script
├── check_requirements.py       # Script to verify package installation
├── README.md                   # Project documentation
├── SETUP_INSTRUCTIONS.md       # Detailed setup guide
├── API_USAGE.md                # API usage examples
├── ARCHITECTURE.md             # Architecture and design decisions
├── LICENSE                     # MIT License
├── models/
│   └── model.joblib            # Trained model (generated after training)
├── src/
│   ├── api.py                  # FastAPI application for serving predictions
│   ├── data_preprocessing.py   # Data loading and cleaning functions
│   ├── feature_engineering.py  # Feature preprocessing pipeline
│   └── model_training.py       # Model training and evaluation functions
└── tests/
    └── test_customer_churn.py  # Unit tests for all components
```

## Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **FastAPI**: High-performance web framework for API development
- **Joblib**: Model serialization
- **MLflow**: Experiment tracking and model registry
- **Docker**: Containerization for deployment
- **Pydantic**: Data validation and settings management
- **Seaborn & Matplotlib**: Data visualization
- **Gunicorn**: Production WSGI server
- **Uvicorn**: ASGI server for FastAPI

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Quick Setup (Recommended)

Run the startup script to automatically set up a virtual environment and install dependencies:

```bash
chmod +x startup.sh
./startup.sh
```

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

## Usage

### 1. Model Training

Train the customer churn prediction model:

```bash
python train_model.py
```

This script will:
- Load and preprocess the data
- Split data into training and test sets
- Create preprocessing pipelines for numerical and categorical features
- Train a Random Forest classifier with cross-validation
- Tune hyperparameters using GridSearchCV
- Evaluate model performance
- Save the trained model to `models/model.joblib`
- Log experiment with MLflow

### 2. Start the API Server

After training the model, start the API server:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Or using Gunicorn for production:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 src.api:app
```

### 3. Access API Documentation

Once the server is running, visit:
- http://localhost:8000/docs - Interactive API documentation (Swagger UI)
- http://localhost:8000/redoc - Alternative API documentation (ReDoc)

### 4. View MLflow Experiments

To view experiment tracking:

```bash
mlflow ui
```

Then visit http://localhost:5000

### 5. Run Tests

Execute unit tests to verify functionality:

```bash
python -m pytest tests/
```

## API Endpoints

- `GET /` - Root endpoint to check if the API is running
- `GET /health` - Health check endpoint
- `POST /predict` - Predict churn for a single customer
- `POST /predict/batch` - Predict churn for multiple customers

### Example API Usage

#### Single Prediction

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

#### Batch Prediction

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
    }
  ]
}'
```

## Dataset

The project uses a telecommunications customer churn dataset with the following features:

### Customer Information
- `customerID`: Unique identifier for each customer
- `gender`: Customer's gender (Male/Female)
- `SeniorCitizen`: Whether the customer is a senior citizen (1/0)
- `Partner`: Whether the customer has a partner (Yes/No)
- `Dependents`: Whether the customer has dependents (Yes/No)

### Services
- `tenure`: Number of months the customer has been with the company
- `PhoneService`: Whether the customer has phone service (Yes/No)
- `MultipleLines`: Whether the customer has multiple lines (Yes/No/No phone service)
- `InternetService`: Customer's internet service provider (DSL/Fiber optic/No)
- `OnlineSecurity`: Whether the customer has online security (Yes/No/No internet service)
- `OnlineBackup`: Whether the customer has online backup (Yes/No/No internet service)
- `DeviceProtection`: Whether the customer has device protection (Yes/No/No internet service)
- `TechSupport`: Whether the customer has tech support (Yes/No/No internet service)
- `StreamingTV`: Whether the customer has streaming TV (Yes/No/No internet service)
- `StreamingMovies`: Whether the customer has streaming movies (Yes/No/No internet service)

### Billing
- `Contract`: The customer's contract type (Month-to-month/One year/Two year)
- `PaperlessBilling`: Whether the customer has paperless billing (Yes/No)
- `PaymentMethod`: The customer's payment method
- `MonthlyCharges`: The amount charged to the customer monthly
- `TotalCharges`: The total amount charged to the customer

### Target Variable
- `Churn`: Whether the customer churned (Yes/No)

## Model Performance

The model achieves the following performance metrics on the test set:
- F1 Score: ~0.85
- Precision and Recall balanced for both classes
- Confusion matrix visualization for detailed performance analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset adapted from the IBM Watson Telco Customer Churn dataset
- Built with scikit-learn, FastAPI, and other open-source libraries