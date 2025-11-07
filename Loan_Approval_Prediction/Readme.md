# 🧠 Loan Classification Pipeline (ML + Streamlit App)

This project demonstrates an end-to-end **machine learning classification pipeline** for predicting **loan approval status** based on applicant features.  
It includes:
- A **scikit-learn classification pipeline** trained, optimized, and saved as `loan_pipeline_v1.joblib`
- A **Streamlit web app** (`app.py`) for real-time prediction
- Supporting **Jupyter notebooks** for model research, training, and deployment preparation

---

## 🚀 Project Overview

This project automates the process of predicting whether a loan application should be approved or rejected using historical data.  
It covers all stages of a modern ML workflow — from exploration to deployment.

### 🔧 Key Components
| File | Description |
|------|--------------|
| `02_Classification_Research_Clean.ipynb` | Exploratory data analysis (EDA), feature engineering, and model experimentation |
| `02_Classification_Pipeline_Final.ipynb` | Finalized preprocessing and model training pipeline |
| `loan_pipeline_v1.joblib` | Serialized and trained scikit-learn Pipeline |
| `app.py` | Streamlit app for making predictions using the trained pipeline |
| `streamlit.ipynb` | Optional notebook for testing or demonstrating the Streamlit workflow |

---

## ⚙️ Tech Stack
- **Python 3.10+**
- **Scikit-learn** for preprocessing and model training
- **Pandas & NumPy** for data manipulation
- **Joblib** for model serialization
- **Streamlit** for interactive deployment
- **Jupyter Notebook** for research and documentation

---

## 📊 Machine Learning Workflow

1. **Data Preprocessing**
   - Missing value handling  
   - Categorical encoding  
   - Feature scaling  
   - Outlier detection and treatment  

2. **Model Training**
   - Multiple classification algorithms tested (e.g., Logistic Regression, Random Forest, Gradient Boosting)  
   - Evaluation using metrics like **accuracy**, **precision**, **recall**, and **F1-score**  

3. **Pipeline Creation**
   - Combined preprocessing and model into one unified pipeline  
   - Exported using `joblib.dump()`  

4. **Deployment**
   - Streamlit-based web UI  
   - Dynamic feature detection and CSV upload support  

---

## 🧩 Streamlit App Features

✅ Automatically detects input feature names from the trained pipeline  
✅ Accepts manual input or CSV uploads for prediction  
✅ Displays probability distribution (if model supports `predict_proba`)  
✅ Works directly with the serialized pipeline — no retraining needed  

### Run the app locally:
```bash
# Clone this repository
git clone https://github.com/<your-username>/loan-classifier-pipeline.git
cd loan-classifier-pipeline

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
