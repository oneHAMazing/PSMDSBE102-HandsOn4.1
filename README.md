# PSMDSBE102-HandsOn4.1
In compliance with Hands-on 4.1 requirements.
This project was completed for Special Topics course.
It covers model training, hyperparameter tuning, experiment tracking, and deployment preparation using Python, Scikit-Learn, Ray Tune, and MLflow.

This repository contains the full workflow for training and evaluating a credit-risk classification model.
Key Components
Data processing: Cleaning, feature engineering, scaling, and train-test split.
Model training: Random Forest classifier optimized using Ray Tune with cross-validation.
Hyperparameter tuning: Automated search using HyperOpt + AsyncHyperBand Scheduler.
Experiment tracking: All runs logged to MLflow, including metrics, parameters, and final model artifacts.
Artifacts saved
- final_model.pkl (trained model)
- scaler.pkl
- feature_columns.pkl
- MLflow run logs
- Ray Tune tuning results

PLEASE NOTE:
Download file from Kaggle: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
Save in the same folder as other files as "loan_data.csv"

HOW TO RUN THE WEB APP:
Run "streamlit run app.py" in your cmd line.

For questions or clarifications regarding this project, you may contact:
Email: qhaamozo@tip.edu.ph

