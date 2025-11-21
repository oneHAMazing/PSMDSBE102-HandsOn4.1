import pandas as pd
import numpy as np
import psutil
import warnings
import os
import joblib
from functools import partial

# Ray/Tune/MLflow Imports
import ray
import ray.train
from ray.util.joblib import register_ray
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import mlflow
import mlflow.sklearn

# Sklearn Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Local Config
from config import (
    DATA_FILE, MODEL_FILE, SCALER_FILE, NUM_WORKERS, CPUS_PER_WORKER,
    RAY_TEMP_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, N_TRIALS, MAX_CONCURRENT_TRIALS
)

# Suppress minor warnings
warnings.filterwarnings('ignore')

# =================================================================
# 1. Data Processing Functions
# =================================================================

def load_and_preprocess_data(data_path):
    """Loads, cleans, and prepares data for modeling."""
    print("🧹 Starting data cleaning and feature engineering...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {data_path}. Please check DATA_FILE in config.py.")
        return None, None, None

    # Handle 'unknown' gender and other cleaning
    df = df[df['person_gender'] != 'OTHER']
    df['person_gender'] = df['person_gender'].replace({'unknown': 'male', 'female': 'female'})
    
    # Simple Imputation
    df['loan_int_rate'].fillna(df['loan_int_rate'].mean(), inplace=True)
    
    # Feature Engineering
    df['loan_to_income'] = df['loan_amnt'] / df['person_income']
    df['income_to_age'] = df['person_income'] / df['person_age']
    df['credit_age_ratio'] = df['credit_score'] / df['person_age']
    
    # One-Hot Encoding
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if 'previous_loan_defaults_on_file' in categorical_cols:
        categorical_cols.remove('previous_loan_defaults_on_file') 
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Convert binary 'Yes/No' to 1/0
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    
    # Final feature set
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    print("✅ Data prepared successfully.")
    return X, y, list(X.columns)

def split_and_scale_data(X, y):
    """Splits data and applies standard scaling."""
    print("📊 Data Splitting and Scaling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Data split and scaled.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# =================================================================
# 2. Ray Tune Model Training Function
# =================================================================

def train_loan_model(config, X_train, X_test, y_train, y_test):
    """
    Train a Random Forest model. 
    Note: We do NOT explicitly call mlflow.start_run here. 
    Ray Tune's MLflowLoggerCallback handles the run creation and metric logging automatically.
    """
    try:
        # Define model
        rf = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            bootstrap=config["bootstrap"],
            random_state=42,
            n_jobs=CPUS_PER_WORKER
        )

        # Cross-validation for robust metrics
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        auc_scores = []
        
        for train_index, val_index in kf.split(X_train, y_train):
            X_kf_train, X_kf_val = X_train[train_index], X_train[val_index]
            y_kf_train, y_kf_val = y_train.iloc[train_index], y_train.iloc[val_index]

            rf.fit(X_kf_train, y_kf_train)
            y_val_proba = rf.predict_proba(X_kf_val)[:, 1]
            auc = roc_auc_score(y_kf_val, y_val_proba)
            auc_scores.append(auc)

        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        
        # Calculate training AUC for overfitting check
        rf.fit(X_train, y_train)
        train_proba = rf.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)

        # Report metrics to Ray Tune
        # The MLflowLoggerCallback will automatically pick these up and log them to MLflow
        ray.train.report({
            "mean_auc": mean_auc, 
            "std_auc": std_auc, 
            "train_auc": train_auc
        })

    except Exception as e:
        print(f"❌ Trial failed: {e}")
        # Report failure
        ray.train.report({"mean_auc": 0.0, "std_auc": 0.0, "train_auc": 0.0})

# =================================================================
# 3. Main Execution Block
# =================================================================

def main():
    # --- A. Ray Initialization ---
    print("🚀 Initializing Ray...")
    total_cpus = psutil.cpu_count(logical=False)
    
    ray_init_config = {
        "ignore_reinit_error": True,
        "num_cpus": NUM_WORKERS * CPUS_PER_WORKER,
        "include_dashboard": False,
        "logging_level": "INFO",
        "_temp_dir": RAY_TEMP_DIR,
    }
    
    if ray.is_initialized():
        ray.shutdown()
    ray.init(**ray_init_config)
    register_ray()
    print(f"✅ Ray initialized with {NUM_WORKERS} workers!")

    # --- B. MLflow Configuration ---
    print("\n📊 Configuring MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"✅ Created new experiment: {EXPERIMENT_NAME}")
    except:
        print(f"📁 Experiment {EXPERIMENT_NAME} already exists")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- C. Data Loading and Preparation ---
    X, y, feature_columns = load_and_preprocess_data(DATA_FILE)
    if X is None:
        return 
    
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Save scaler immediately
    joblib.dump(scaler, SCALER_FILE)
    print(f"✅ Scaler saved to {SCALER_FILE}")
    
    # --- D. Hyperparameter Tuning Setup ---
    print("\n🔬 Setting up Hyperparameter Tuning...")

    search_space = {
        "n_estimators": tune.choice([50, 100, 200]),
        "max_depth": tune.choice([5, 10, 15]),
        "min_samples_split": tune.choice([2, 5, 10]),
        "min_samples_leaf": tune.choice([1, 2, 4]),
        "max_features": tune.choice(["sqrt", "log2"]),
        "bootstrap": tune.choice([True, False]),
    }
    
    hyperopt_search = HyperOptSearch(metric="mean_auc", mode="max")
    hyperopt_search = tune.search.ConcurrencyLimiter(hyperopt_search, max_concurrent=MAX_CONCURRENT_TRIALS)

    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="mean_auc", mode="max", 
        max_t=50, grace_period=2, reduction_factor=3
    )

    # This callback handles all MLflow logging for the trials
    run_config = RunConfig(
        name="tune",
        storage_path=RAY_TEMP_DIR,
        verbose=1,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=MLFLOW_TRACKING_URI,
                experiment_name=EXPERIMENT_NAME,
                save_artifact=True
            )
        ],
    )

    # --- E. Run Tuning Experiment ---
    print("\n🚀 Starting Hyperparameter Tuning with Ray Tune")
    
    # Use partial to pass the data arguments to the training function
    trainable_with_data = partial(train_loan_model, 
                                    X_train=X_train_scaled, 
                                    X_test=X_test_scaled, 
                                    y_train=y_train, 
                                    y_test=y_test)

    tuner = Tuner(
        trainable_with_data,
        tune_config=TuneConfig(
            search_alg=hyperopt_search,
            scheduler=scheduler,
            num_samples=N_TRIALS,
            max_concurrent_trials=MAX_CONCURRENT_TRIALS,
        ),
        param_space=search_space,
        run_config=run_config,
    )
    
    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result(metric="mean_auc", mode="max")
    best_config = best_result.config
    best_metrics = best_result.metrics
    
    print("\n🏆 Best Hyperparameters Found:")
    print(f"Config: {best_config}")
    print(f"Best AUC: {best_metrics['mean_auc']:.4f}")

    # --- F. Train Final Model and Save ---
    print("\n🎯 Training Final Model with Best Configuration...")
    
    # We start a NEW run for the final model training
    with mlflow.start_run(run_name="final_best_model") as final_run:
        mlflow.set_tag("final_model", "True")
        mlflow.log_params(best_config)
        
        final_model = RandomForestClassifier(
            n_estimators=best_config['n_estimators'],
            max_depth=best_config['max_depth'],
            min_samples_split=best_config['min_samples_split'],
            min_samples_leaf=best_config['min_samples_leaf'],
            max_features=best_config['max_features'],
            bootstrap=best_config['bootstrap'],
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        mlflow.log_metric("test_auc", test_auc)
        print(f"📊 Final Model Test AUC-ROC: {test_auc:.4f}")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=final_model, 
            artifact_path="random_forest_model", 
            registered_model_name="LoanRiskRFModel"
        )
        print("✅ Final model logged to MLflow!")
        
        # Save local artifacts for deployment
        joblib.dump(final_model, MODEL_FILE)
        joblib.dump(feature_columns, 'feature_columns.pkl')
        print(f"✅ Final model saved to {MODEL_FILE}")

    # --- G. Cleanup ---
    print("\n🗑️ Shutting down Ray...")
    ray.shutdown()
    print("✅ Training complete.")

if __name__ == "__main__":
    main()