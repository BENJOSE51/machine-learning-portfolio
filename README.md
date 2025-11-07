# 🏠 Airbnb Market Insights — Exploratory Data Analysis (EDA)

This project explores an **Airbnb dataset** to uncover trends, pricing patterns, and key factors that influence listing popularity and pricing.  
It serves as the foundation for business insights and future predictive modeling.

---

## 📘 Project Overview

The goal of this project is to:
- Analyze Airbnb listings to identify **pricing and demand trends**
- Understand **how location, room type, and availability** impact prices
- Detect **outliers and data quality issues**
- Generate insights useful for **hosts, guests, and business strategy**

---

## 🧠 Key Objectives

1. Perform **data cleaning and preprocessing**  
   - Handle missing values and duplicates  
   - Convert price data to numeric format  
   - Fix incorrect datatypes  

2. Conduct **exploratory data analysis (EDA)**  
   - Study feature distributions (price, reviews, availability, etc.)  
   - Visualize geographical and categorical patterns  
   - Detect and handle outliers  

3. Create **new derived features** for deeper insights  
   - `price_per_review` — price normalized by number of reviews  
   - `availability_ratio` — yearly availability ratio  
   - `host_active_years` — years since host joined the platform  

4. Visualize relationships using:
   - Correlation heatmaps  
   - Boxplots, histograms, and scatter plots  
   - Geographic plots via Plotly  

---

## 🧰 Tools and Libraries Used

| Category | Libraries |
|-----------|------------|
| **Core** | pandas, numpy |
| **Visualization** | matplotlib, seaborn, plotly |
| **ML/Preprocessing (optional)** | scikit-learn |
| **Environment** | Jupyter Notebook |

---

## 🗂️ Project Structure

