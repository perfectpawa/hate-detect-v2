#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Environment setup complete! To activate the environment, run: source venv/bin/activate" 