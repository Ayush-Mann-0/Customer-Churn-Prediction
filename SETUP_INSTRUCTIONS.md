# Setup Instructions

This document provides detailed instructions for setting up the Customer Churn Prediction project.

## System Requirements

- Python 3.8 or higher
- pip (Python package installer)
- Docker (optional, for containerized deployment)
- Git (for version control)

## Detailed Setup Options

### Option 1: Virtual Environment Setup (Recommended)

1. Clone the repository (if you haven't already):
   ```bash
   git clone <repository-url>
   cd CustomerChurn
   ```

2. Create a virtual environment:
   ```bash
   python -m venv churn_env
   ```

3. Activate the virtual environment:
   - On Linux/macOS:
     ```bash
     source churn_env/bin/activate
     ```
   - On Windows:
     ```bash
     churn_env\Scripts\activate
     ```

4. Upgrade pip to the latest version:
   ```bash
   pip install --upgrade pip
   ```

5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using the Startup Script

The project includes a `startup.sh` script that automates the setup process:

1. Make the script executable:
   ```bash
   chmod +x startup.sh
   ```

2. Run the script:
   ```bash
   ./startup.sh
   ```

The script will:
- Create a virtual environment
- Activate it
- Install all required packages
- Verify the installation
- Provide next steps

### Option 3: System-wide Installation

If you prefer to install the packages system-wide:

1. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

Note: This approach may affect other Python projects on your system.

## Handling Installation Issues

### Compilation Errors

If you encounter compilation errors during installation, you may need to install system-level dependencies:

#### Fedora/RHEL/CentOS:
```bash
sudo dnf install gcc gcc-c++ python3-devel
```

#### Ubuntu/Debian:
```bash
sudo apt-get install build-essential python3-dev
```

#### Arch Linux:
```bash
sudo pacman -S base-devel python-pip
```

### Using Pre-compiled Wheels

To avoid compilation issues, you can try installing packages as pre-compiled wheels:

```bash
pip install --only-binary=all -r requirements.txt
```

## Docker Deployment

### Building the Docker Image

1. Build the Docker image:
   ```bash
   docker build -t customer-churn .
   ```

### Running the Container

1. Run the container:
   ```bash
   docker run -p 8000:8000 customer-churn
   ```

2. Access the API at `http://localhost:8000`

### Docker Compose (Optional)

If you have docker-compose installed, you can create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  customer-churn:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
```

Then run:
```bash
docker-compose up
```

## Training the Model

Before using the API, you need to train the model:

```bash
python train_model.py
```

This will:
1. Load and preprocess the dataset
2. Train the machine learning model
3. Save the trained model to `models/model.joblib`
4. Log experiment data with MLflow

## Starting the API Server

### Development Mode

For development, you can use uvicorn directly:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload when code changes.

### Production Mode

For production, use gunicorn with uvicorn workers:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 src.api:app
```

## Testing the Installation

Run the unit tests to verify that everything is working correctly:

```bash
python -m pytest tests/
```

## Verifying the Setup

After installation, you can verify that the required packages are installed:

```bash
python -c "
import pandas as pd
import numpy as np
import sklearn
import fastapi
print('All required packages are installed successfully!')
"
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure you've activated your virtual environment and installed all requirements.

2. **Permission denied**: On Linux/macOS, make sure the startup script is executable (`chmod +x startup.sh`).

3. **Port already in use**: If port 8000 is already in use, change it in the startup command:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8001
   ```

4. **Model not found**: Make sure you've run `python train_model.py` to train and save the model.

### Getting Help

If you encounter issues not covered in this document:
1. Check the project's GitHub issues
2. Verify your Python version meets the requirements
3. Ensure all system dependencies are installed
4. Try creating a fresh virtual environment