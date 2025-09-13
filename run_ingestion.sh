#!/bin/bash

# ==============================================================================
# Shell script to run the crypto data ingestion engine within its virtual env.
# ==============================================================================

# --- IMPORTANT ---
# Set the absolute path to your project directory.
# Replace "/home/your_username/crypto_bot" with your actual path.
PROJECT_DIR="/home/alexgol/Projects/RoboGol_trader_2"

# --- Venv Activation ---
# Set the name of your virtual environment folder. 'venv' is a common name.
VENV_NAME="/home/alexgol/python/venv"

# Navigate to the project directory. The 'cd' command is crucial.
cd "$PROJECT_DIR" || { echo "Error: Could not change to project directory '$PROJECT_DIR'"; exit 1; }

# Activate the virtual environment
source "$VENV_NAME/bin/activate" || { echo "Error: Could not activate virtual environment."; exit 1; }

echo "--- Venv activated. Running ingestion script at $(date) ---"

# --- Run the Python Script ---
# Run the main data ingestion engine script using the venv's python.
python3 data_ingestion_engine.py

echo "--- Ingestion script finished at $(date) ---"

# Deactivate the virtual environment (optional, but good practice)
deactivate
