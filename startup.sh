#!/bin/bash
# Enhanced startup script for Customer Churn Prediction project

echo "Customer Churn Prediction - Quick Start"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "churn_env" ]; then
    echo "Creating virtual environment..."
    python -m venv churn_env
    echo "Virtual environment created."
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source churn_env/bin/activate
echo "Virtual environment activated."
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install system dependencies message
echo "Checking for system dependencies..."
echo "If you encounter compilation errors, you may need to install system dependencies:"
echo "On Fedora/RHEL/CentOS:"
echo "  sudo dnf install gcc gcc-c++ python3-devel"
echo "On Ubuntu/Debian:"
echo "  sudo apt-get install build-essential python3-dev"
echo "On Arch Linux:"
echo "  sudo pacman -S base-devel python-pip"
echo ""

# Try installing requirements with pre-compiled wheels first
echo "Installing requirements (attempt 1 - pre-compiled wheels)..."
if [ -f "requirements.txt" ]; then
    pip install --only-binary=all -r requirements.txt
else
    echo "requirements.txt not found. Installing core packages..."
    pip install --only-binary=all pandas numpy scikit-learn fastapi uvicorn pydantic joblib
    pip install --only-binary=all jupyter notebook
    pip install --only-binary=all mlflow xgboost imbalanced-learn
    pip install --only-binary=all matplotlib seaborn
    pip install --only-binary=all pytest
fi

# If the above fails, try installing packages individually
if [ $? -ne 0 ]; then
    echo ""
    echo "First attempt failed. Trying individual package installation..."
    
    # Install core packages first
    echo "Installing core packages..."
    pip install --only-binary=all numpy pandas
    pip install --only-binary=all scikit-learn
    pip install --only-binary=all fastapi uvicorn pydantic
    pip install --only-binary=all joblib
    pip install --only-binary=all jupyter notebook
    pip install --only-binary=all matplotlib seaborn
    pip install --only-binary=all mlflow
    pip install --only-binary=all xgboost
    pip install --only-binary=all imbalanced-learn
    pip install --only-binary=all pytest
fi

echo ""
echo "Checking installed packages..."
python -c "
try:
    import pandas as pd
    print('✓ pandas installed')
except ImportError:
    print('✗ pandas NOT installed')

try:
    import numpy as np
    print('✓ numpy installed')
except ImportError:
    print('✗ numpy NOT installed')

try:
    import sklearn
    print('✓ scikit-learn installed')
except ImportError:
    print('✗ scikit-learn NOT installed')

try:
    import fastapi
    print('✓ fastapi installed')
except ImportError:
    print('✗ fastapi NOT installed')
"

echo ""

# Check if dataset exists
if [ ! -f "dataset.csv" ]; then
    echo "WARNING: dataset.csv not found!"
    echo "Please ensure your dataset file is in the project directory."
    echo ""
fi

# Check if model exists
if [ ! -f "models/model.joblib" ]; then
    echo "Model not found. To train the model later:"
    echo "1. Make sure all packages are installed correctly"
    echo "2. Run: python train_model.py"
    echo ""
fi

echo "Setup process completed!"
echo ""
echo "Next steps:"
echo "1. Start Jupyter Notebook: jupyter notebook"
echo "2. Start API server: uvicorn src.api:app --host 0.0.0.0 --port 8000"
echo "3. Check requirements: python check_requirements.py"
echo "4. Run tests: python -m pytest tests/"
echo ""
echo "If you had compilation errors, please install system dependencies:"
echo "Fedora/RHEL/CentOS: sudo dnf install gcc gcc-c++ python3-devel"
echo "Ubuntu/Debian: sudo apt-get install build-essential python3-dev"
echo ""
echo "For detailed instructions, see README.md, SETUP_INSTRUCTIONS.md, and other documentation files"