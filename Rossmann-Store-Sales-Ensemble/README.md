# Rossmann Store Sales — Regression Ensemble

## Overview
This project predicts daily sales for Rossmann stores using a combination of historical sales, promotions, and store metadata.
The notebook implements feature engineering, model training (ensemble/stacking), and evaluation.

## Contents
- `notebooks/` — cleaned notebook: `03_Rossmann_Regression_Ensemble_Clean.ipynb`
- `data/` — dataset files (place `train.csv`, `store.csv`, etc. here)
- `models/` — saved model artifacts (e.g., `rossmann_pipeline_v1.joblib`)

## Highlights
- Engineered date-based and lag features for time-series prediction.
- Ensemble modeling improved performance over single models.
- Evaluation metrics: RMSE and MAE are reported in the notebook.

## Next Steps
- Serialize the final pipeline to `models/rossmann_pipeline_v1.joblib` for deployment.
- Optionally build FastAPI + Streamlit demo to expose predictions.

## Tech
Python, pandas, scikit-learn, xgboost/lightgbm, joblib

---
