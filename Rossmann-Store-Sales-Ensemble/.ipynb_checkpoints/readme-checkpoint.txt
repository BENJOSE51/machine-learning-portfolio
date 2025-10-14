# 🏪 Rossmann Store Sales Prediction — Ensemble Regression Project

## 🧠 Problem Statement
Rossmann operates over 3,000 drug stores across Europe.  
The company needs a model to **predict daily sales** for each store to improve decisions around inventory, staffing, and promotions.

---

## 🎯 Business Questions
1. What factors drive daily sales at Rossmann stores?
2. Can we forecast sales accurately to support inventory and staffing?
3. How do promotions and competition influence store performance?

---

## 📊 Dataset Overview
**Source:** [Kaggle – Rossmann Store Sales Competition](https://www.kaggle.com/c/rossmann-store-sales)

**Files Used:**
- `train.csv` — historical sales and store details  
- `store.csv` — store metadata (type, assortment, competition info)

**Target Variable:** `Sales`

---

## 🧩 Approach
### 1️⃣ Data Preprocessing
- Converted datatypes (dates → datetime, categoricals → category)
- Checked for null values and applied simple, logical imputations:
  - `CompetitionDistance` → median  
  - Promotion fields → 0 / “None”
- Handled outliers using **IQR capping**
- Created meaningful features:
  - `Year`, `Month`, `WeekOfYear`, `DayOfWeek`
  - `CompetitionAgeMonths`, `IsPromoMonth`, `Promo2`
- Log-transformed `Sales` to reduce skew and stabilize variance

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized sales distributions before/after log transformation  
- Analyzed average sales by day of week and rolling 7-day trends  
- Correlation heatmaps for numeric variables  
- Insights documented as Markdown within the notebook

### 3️⃣ Modeling
Trained and compared multiple regression models:
| Model | Type | Notes |
|-------|------|-------|
| **Linear Regression** | Baseline | Benchmark for comparison |
| **Random Forest** | Ensemble (Bagging) | Handles non-linearities |
| **Gradient Boosting** | Ensemble (Boosting) | Strong predictive baseline |
| **XGBoost** | Gradient Boosting (Optimized) | Industry standard |
| **Stacking Regressor** | Ensemble of RF & GB | Combines multiple learners |

### 4️⃣ Evaluation Metrics
All models were evaluated on **inverse log-transformed predictions** using:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² (Coefficient of Determination)**

---

## 🧪 Results Summary
| Model | RMSE ↓ | MAE ↓ | R² ↑ |
|--------|--------|--------|-------|
| Linear Regression | 1450 | 980 | 0.80 |
| Random Forest | 1285 | 850 | 0.87 |
| Gradient Boosting | 1243 | 821 | 0.88 |
| XGBoost | 1208 | 800 | 0.89 |
| Stacking Regressor | **1185** | **785** | **0.90** |

✅ **XGBoost** and **Stacking** achieved the best performance.

---

## 💡 Key Insights
- **Promotions (`Promo`, `IsPromoMonth`)** significantly increase sales — suggesting well-timed campaigns drive revenue.  
- **DayOfWeek** reveals weekly seasonality; weekends show predictable peaks.  
- **CompetitionDistance** and **CompetitionAgeMonths** negatively correlate with sales — closer or newer competitors reduce store revenue.  
- Seasonal patterns (Month/WeekOfYear) imply forecasting could benefit from calendar effects.

---

## 🚀 Next Steps
1. Use **time-based cross-validation** (train on past, test on future) for realistic forecasting.  
2. Perform **hyperparameter tuning** with `RandomizedSearchCV` for ensembles.  
3. Integrate **external data** (holidays, weather, events) for better predictions.  
4. Add **SHAP** or **LIME** for model interpretability.  
5. Deploy model via Flask or FastAPI for real-time prediction.

---

## 🧰 Tech Stack
- **Language:** Python 3.10  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost  
- **Environment:** Jupyter Notebook / Anaconda  

