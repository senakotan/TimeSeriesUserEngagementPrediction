# 📊 Tabii Time-Series User Engagement Prediction

**A data-driven forecasting project** developed during my internship at **TRT (Turkish Radio and Television Corporation)**.  
The aim of this study is to **analyze user engagement patterns** and **predict daily average watch durations** on the **Tabii** digital streaming platform using **time-series analysis** and **machine learning models**.

---

## 🧠 Overview

This project focuses on understanding user behavior dynamics over time and predicting future engagement metrics to support data-driven decision-making at TRT.  
We used **Meta Prophet** to model trends, seasonality, and the impact of holidays on user activity.

The forecasting pipeline covers:
- Data extraction from **BigQuery**
- Data preprocessing & feature engineering
- Model training and evaluation
- Deployment through an interactive **Streamlit Dashboard**

---

## 🔍 Data Preparation

- Raw engagement data contained **495,504,618 rows**, representing user sessions, devices, and content types.  
- Data was aggregated **daily** using BigQuery SQL queries, resulting in **381 records** for analysis.
- Service Account credentials were used for secure data access.
- Extracted datasets were visualized via **Matplotlib** for initial exploration.

### Key Preprocessing Steps
- Converted timestamps to `datetime` format.  
- Added time-based columns (`year`, `month`, `weekday`, `is_month_start`, `is_month_end`).  
- Checked data distribution with `df.info()` and `df.describe()`.  
- Detected and flagged outliers using **IQR** and **Z-score** methods.

---

## ⚙️ Feature Engineering

Feature engineering aimed to transform raw data into meaningful predictors that improve model performance.

### 🧩 Categories of Features
- **Lag Features:** `lag1`, `lag7`, `lag14`, `lag28`  
- **Rolling Features:** Moving averages and standard deviations for 7, 14, and 28 days  
- **Diff Features:** Daily and weekly change rates (`diff1`, `diff7`)  
- **Seasonal Features:** `month_sin`, `month_cos`, `dow_sin`, `dow_cos`  
- **Outlier Flags:** IQR and Z-score based anomaly detection  

Over **50 new features** were created.  
Feature importance analysis with **XGBoost** revealed that short-term lags and 7-day rolling averages were the most influential in predicting user engagement.

---

## 🔮 Forecasting Model

We used **Meta Prophet**, an open-source forecasting model developed by Facebook, which decomposes a time series into:

> **Trend + Seasonality + Holiday effects**

### Prophet Configuration
- Custom regressors added with `.add_regressor()`
- Official Turkish holidays and weekends included (`prophet_holidays_tr.csv`)
- Weekly and monthly seasonality encoded via sine/cosine transformations
- 80% training, 20% testing split

### Selected Regressors
- `ortalama_sure_lag1`
- `ortalama_sure_roll7_mean`
- `ortalama_sure_roll14_mean`
- `ortalama_sure_roll28_mean`
- `toplam_izleme_suresi_dk_roll14_std`
- `month_sin`

---

## 📏 Evaluation Metrics

| Metric | Description | Insight |
|:--|:--|:--|
| **MAE** | Mean Absolute Error | Measures average deviation of predictions |
| **MAPE** | Mean Absolute Percentage Error | Measures accuracy as a percentage |
| **Accuracy** | 100 − MAPE | Shows overall model performance |

✅ The model achieved an **average forecasting accuracy above 95%**, demonstrating strong predictive capability and stability across test periods.  
Weekly forecasts yielded the most consistent and reliable results.

---

## 🖥️ Streamlit Dashboard

The project includes a **Streamlit-based user interface** that allows:
- Interactive visualization of training, test, and forecast results  
- Parameter tuning for Prophet  
- Comparison of actual vs predicted engagement metrics  
- Downloadable charts and metrics (`.csv`, `.png`)


---

## 📈 Results

- Prophet successfully captured **weekly patterns** and **holiday effects** in user engagement.  
- Forecasting accuracy exceeded **95%**, confirming the model’s robustness for real-world usage.  
- The model’s interpretability allowed clear visualization of trend, seasonality, and external factors influencing user activity.

### Forecast Visualization
<img width="1918" height="931" alt="prophet_forecast" src="https://github.com/user-attachments/assets/c33fa635-6d75-4a3b-bdf3-0db0fcbc4808" />


---


## 👩‍💻 Author

**Sena Kotan**  
_Data Science & Machine Learning Intern at TRT_  
📍 Fatih Sultan Mehmet Vakıf University — Electrical Elecronics & Computer Engineering  



