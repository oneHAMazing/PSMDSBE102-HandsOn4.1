import os
import tempfile

# ==================================
# Data and Artifact Paths
# ==================================
# NOTE: Assuming your data is named 'credit_risk_data.csv' and is in the same directory.
DATA_FILE = "loan_data.csv"
MODEL_FILE = "final_model.pkl"
SCALER_FILE = "scaler.pkl"

# ==================================
# Ray Configuration
# ==================================
# Ray settings from the notebook
NUM_WORKERS = 8
CPUS_PER_WORKER = 1
# Using the specific temporary directory path from the notebook
RAY_TEMP_DIR = "C:/ray_temp"

# Ensure the temp directory exists
os.makedirs(RAY_TEMP_DIR, exist_ok=True)

# ==================================
# MLflow Configuration
# ==================================
MLFLOW_TRACKING_URI = "./mlruns"
EXPERIMENT_NAME = "loan_tune"

# ==================================
# Hyperparameter Tuning Settings
# ==================================
N_TRIALS = 10
MAX_CONCURRENT_TRIALS = 2

print("Config complete!")