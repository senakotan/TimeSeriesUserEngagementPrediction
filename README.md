# 📊 TRT Tabii | Time-Series User Engagement Prediction

A forecasting project developed during my internship at **TRT (Turkish Radio and Television Corporation)**.  
The goal is to analyze user behavior and predict **daily average watch duration** on the **Tabii** digital streaming platform using **time-series analysis**.

---

## 🧠 Project Summary

This study focuses on modeling engagement trends and capturing seasonal patterns.  
We used **Meta Prophet** as the main forecasting model, supported by custom regressors and extensive feature engineering.

The pipeline includes:
- Data extraction from **BigQuery**
- Preprocessing & feature engineering
- Prophet modeling
- A fully interactive **Streamlit Dashboard**

---

## 📂 Data & Preprocessing

- Raw dataset: **495M+ rows**
- Aggregated to **381 daily records** using BigQuery
- Cleaned and enhanced with:
  - Time features (`year`, `month`, `weekday`, etc.)
  - Outlier detection (IQR & Z-score)
  - Initial exploratory visualizations

---

## ⚙️ Feature Engineering

Over **50 predictive features** were created:

- **Lag features:** 1, 7, 14, 28 day  
- **Rolling stats:** 7, 14, 28 day means & std  
- **Diff features:** daily and weekly change rates  
- **Seasonality:** sine/cosine encodings  
- **Outlier flags**

XGBoost importance highlighted **short-term lags** and **7-day rolling averages** as strongest contributors.

---

## 🔮 Forecasting Model (Meta Prophet)

Prophet decomposes the series into **trend + seasonality + holidays**.

### Key Model Components
- Train/test split: **80/20**
- Regressors added with `.add_regressor()`
- Weekly & monthly seasonality
- Turkish official holidays included

---

## 📏 Evaluation

| Metric | Purpose |
|-------|---------|
| **MAE** | Measures average error |
| **MAPE** | Percentage-based error |
| **Accuracy** | 100 − MAPE |

➡️ The model achieved **95%+ accuracy**, effectively capturing weekly and holiday-related trends.

---

## 📈 Results

- Weekly engagement patterns modeled accurately  
- Stable short-term forecasts  
- Clear interpretability for trend & seasonal components  

---

## 🖥️ Streamlit Dashboard

The project includes an interactive dashboard that allows:
- Actual vs predicted visualization  
- Parameter tuning  
- Forecast exploration  
- Export options (`.csv`, `.png`)

### Dashboard Interface  
<img width="1912" height="821" alt="tabii_proje" src="https://github.com/user-attachments/assets/4b8a003b-4dc7-46e0-b31c-689dc363df38" />

---

## 👩‍💻 Author

**Sena Kotan**  
_Data Science & Machine Learning Intern at TRT_  
📍 FSMVU — Electrical Electronics & Computer Engineering  
