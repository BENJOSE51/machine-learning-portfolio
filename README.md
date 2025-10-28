# 🧩 Machine Learning Portfolio — Ben Jose

Welcome to my end-to-end **Machine Learning Portfolio**, showcasing projects that cover the complete data science and ML workflow — from **EDA and model development** to **deployment-ready pipelines**.

This repository demonstrates my skills in:
- Data cleaning and visualization  
- Supervised and unsupervised machine learning  
- Natural language processing (NLP)  
- Deep learning (CNN)  
- Model deployment using FastAPI, Streamlit, and Docker  

---

## 🗂️ Repository Structure

```
machine-learning-portfolio/
├── data/
│   ├── airbnb.csv
│   └── loan_approval.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Regression_Research_Clean.ipynb
│   ├── 02_Regression_Pipeline_Final.ipynb
│   ├── 03_Classification.ipynb
│   ├── 04_NLP_TFIDF.ipynb
│   ├── 05_CNN_Image_Classification.ipynb
│   └── 06_Unsupervised_Learning.ipynb
├── requirements.txt
└── README.md
```

---

## 🧮 Projects Overview

### **1️⃣ Airbnb EDA**
- Exploratory data analysis of Airbnb listings.
- Visualizes patterns in pricing, reviews, and location.
- Key tools: `pandas`, `matplotlib`, `seaborn`.

### **2️⃣ Loan Approval Prediction (Regression)**
- Research notebook explores feature correlations and baseline models.  
- Pipeline notebook builds a **reproducible ML pipeline** and saves a deployable model:  
  `loan_pipeline_v1.joblib`
- Deployment: joblib + Streamlit

### **3️⃣ Classification Project**
- Focused on categorical predictions using logistic regression and tree-based models.

### **4️⃣ NLP (TF-IDF)**
- Text preprocessing, vectorization, and sentiment classification.

### **5️⃣ CNN Image Classification**
- Simple CNN built using TensorFlow/Keras for image recognition tasks.

### **6️⃣ Unsupervised Learning**
- Clustering and dimensionality reduction to discover hidden data patterns.

---

## 🚀 Deployment Roadmap

Deployment repositories:

| Project | Repository | Description |
|----------|-------------|-------------|
| Loan Approval Pipeline | [ml-regression-api](#) | joblib + Streamlit + Docker model deployment |

---

## ⚙️ Tech Stack

- **Languages:** Python  
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib, tensorflow,pytorch, joblib  
- **Deployment:** Streamlit, Docker  
- **Version Control:** VSCode,Git & GitHub  

---

## 📚 How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/machine-learning-portfolio.git
   cd machine-learning-portfolio
   ```

2. Create a virtual environment & install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Windows: venv\Scripts\activate)
   pip install -r requirements.txt
   ```

3. Open notebooks:
   ```bash
   jupyter notebook
   ```


## 🧑‍💻 About Me

**Ben Jose**  
📍 Data Science & ML Enthusiast | Transitioning from mechanical automation to data science 
🎯 Focus: Data Science and Data Analysis, Machine Learning Engineering  
📧 Contact: benjose51@gmail.com #+919645259675

---

_This repository is actively updated as new ML projects and deployments are added._
