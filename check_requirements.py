#!/usr/bin/env python3
"""
Script to check if all required packages are installed correctly.
"""

import sys
import importlib

def check_package(package_name, version=None):
    """Check if a package is installed and optionally check its version."""
    try:
        package = importlib.import_module(package_name)
        if version:
            installed_version = getattr(package, '__version__', 'Unknown')
            print(f"✓ {package_name} ({installed_version})")
        else:
            print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} (NOT INSTALLED)")
        return False

def main():
    """Main function to check all required packages."""
    print("Checking required packages...\n")
    
    # Core packages
    required_packages = [
        ("pandas", True),
        ("numpy", True),
        ("sklearn", True),
        ("fastapi", True),
        ("uvicorn", True),
        ("pydantic", True),
        ("joblib", True),
    ]
    
    # Optional packages
    optional_packages = [
        ("mlflow", True),
        ("matplotlib", True),
        ("seaborn", True),
        ("gunicorn", True),
        ("pytest", True),
    ]
    
    # Check required packages
    print("Required packages:")
    all_required_installed = True
    for package, check_version in required_packages:
        if not check_package(package, check_version):
            all_required_installed = False
    print()
    
    # Check optional packages
    print("Optional packages:")
    for package, check_version in optional_packages:
        check_package(package, check_version)
    print()
    
    # Overall status
    if all_required_installed:
        print("✓ All required packages are installed!")
        print("\nNext steps:")
        print("1. Train the model: python train_model.py")
        print("2. Start the API: uvicorn src.api:app --host 0.0.0.0 --port 8000")
    else:
        print("✗ Some required packages are missing!")
        print("Please install the missing packages using:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
