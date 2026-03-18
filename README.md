# 📡 Telco Customer Churn Prediction & Analysis

> End-to-end churn analytics project combining **Machine Learning** (Python) and **Business Intelligence** (Power BI) — predicting which telecom customers are likely to leave and visualizing churn patterns to support retention strategy.

---

## 📌 Project Overview

Customer churn is one of the most critical metrics for telecom companies. This project analyzes **7,043 customer records** to identify churn drivers, build predictive models, and deliver an interactive Power BI dashboard for stakeholders.

| | |
|---|---|
| **Target variable** | `Churn` (Yes / No) |
| **Churn rate** | 26.5% (1,869 churned out of 7,043) |
| **Approach** | EDA → Preprocessing → Classification → Dashboard |

---

## 🗂️ Dataset

**Source:** IBM Sample Dataset — Telco Customer Churn (Kaggle)

| Category | Features |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account info | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Charges | `MonthlyCharges`, `TotalCharges` |
| **Target** | `Churn` |

---

## 🔄 Project Pipeline

```
Raw Data (7,043 customers × 21 features)
        │
        ▼
1. EDA & Data Cleaning     ← Handle 11 whitespace TotalCharges, outliers, distributions
        │
        ▼
2. Feature Engineering     ← Encoding (Ordinal + OneHot), MinMax scaling, VIF check
        │
        ▼
3. Class Imbalance         ← SMOTE oversampling (for SVM pipeline)
        │
        ▼
4. Model Training          ← Random Forest / XGBoost / SVM
        │
        ▼
5. Model Evaluation        ← Accuracy, Precision, Recall, F1, ROC-AUC, Avg Precision
        │
        ▼
6. Power BI Dashboard      ← 5-page interactive report for business stakeholders
```

---

## 🔍 Exploratory Data Analysis

- **Distribution analysis** — histograms with KDE for `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`
- **Categorical breakdown** — countplots for all categorical features (<10 unique values)
- **Correlation heatmap** — multicollinearity check across numeric features
- **VIF analysis** — Variance Inflation Factor computed for `MonthlyCharges`, `TotalCharges`, and `tenure` to detect redundant predictors

**Key finding:** `TotalCharges` is highly correlated with both `tenure` and `MonthlyCharges`, indicating multicollinearity that was accounted for in preprocessing.

---

## ⚙️ Preprocessing

| Step | Detail |
|---|---|
| Missing values | 11 whitespace entries in `TotalCharges` → dropped |
| Target encoding | `Churn`: Yes → 1, No → 0 |
| Ordinal encoding | `Contract`: Month-to-month (0) → One year (1) → Two year (2) |
| OneHot encoding | All other nominal categorical columns (`drop='first'`) |
| Scaling | `MinMaxScaler` on `tenure`, `MonthlyCharges`, `TotalCharges` |
| Train/Test split | 80% / 20%, `random_state=42` |
| SMOTE | Applied to training set for SVM to handle class imbalance (26.5% churn) |

---

## 🤖 Models & Evaluation

Three classifiers were trained and compared:

| Model | Handling Imbalance | Notes |
|---|---|---|
| **Random Forest** | `class_weight='balanced'` | Ensemble, robust to noise |
| **XGBoost** | `scale_pos_weight` (inverse churn ratio) | Gradient boosting, handles non-linearity |
| **SVM** | SMOTE + `class_weight='balanced'` | Trained on resampled data with StandardScaler |

**Evaluation metrics per model:**
- Accuracy · Precision · Recall · F1-score (for both classes)
- **ROC-AUC** and **Average Precision** — critical for imbalanced classification

> ⚠️ For churn prediction, **Recall for class 1 (churned)** is prioritized over overall accuracy — missing a churner is more costly than a false alarm.

---

## 📊 Power BI Dashboard

A 5-page interactive report with a **navigation home page** and a `Churn?` filter slicer across all analytical pages.

### 🏠 HOME
Navigation hub with buttons linking to all 4 analytical pages.

### 📈 OVERVIEW
High-level churn KPIs and trend analysis.
- KPI cards: Total Customers · Churn Customers · Current Subscribers · Churn Rate %
- Churn Rate % by Payment Method
- Churn Customers by Tenure (binned)
- Decomposition tree — drill into churn drivers

### 👥 CUSTOMER DEMOGRAPHICS
- Gender distribution (donut chart)
- Age distribution — Senior vs. non-Senior citizens
- Partner & Dependents breakdown
- Pivot table — demographic cross-tabulation

### 📶 SERVICE & CONTRACT
- Phone Service, Internet Service, Multiple Lines (pie / ribbon charts)
- Add-on services: Online Security, Online Backup, Device Protection, Tech Support (treemaps)
- Streaming TV & Movies usage
- Gauge chart — contract type distribution

### 💰 CHURN & REVENUE
- Paperless Billing vs. churn
- Internet Service type vs. churn
- Revenue impact analysis by churn segment

**DAX Measures used:** `Churn Rate %` · `Total Customers` · `churn customers` · `Current Subscriber` · `StreamingTV%` · `StreamingMovies_yes%` · `tenure (bins)`

---

## 💡 Key Insights

1. **Month-to-month contracts** have the highest churn rate — customers on longer contracts are significantly more loyal.
2. **Fiber optic internet users** churn more than DSL users, suggesting a price-to-value perception issue.
3. **Customers without add-on services** (no OnlineSecurity, no TechSupport) are far more likely to churn.
4. **Electronic check** is the payment method most associated with churn.
5. **Tenure is a strong churn predictor** — the first 12 months are the highest-risk window, making early engagement critical.
6. **Senior citizens** (16.2% of base) show elevated churn, warranting targeted retention programs.

---

## 📁 File Structure

```
├── Telco_Customer_Churn.ipynb              ← EDA, preprocessing, model training & evaluation
├── Telco_Customer_Churn.pbix               ← Power BI dashboard (5 pages)
└── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Raw dataset (7,043 rows × 21 columns)
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data & EDA | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| Preprocessing | `scikit-learn` (OrdinalEncoder, OneHotEncoder, MinMaxScaler, ColumnTransformer) |
| Imbalance handling | `imbalanced-learn` (SMOTE) |
| Modeling | `scikit-learn` (RandomForest, SVM), `xgboost` (XGBClassifier) |
| Statistics | `statsmodels` (VIF) |
| BI Dashboard | Power BI Desktop (DAX measures, 5-page report) |

---

## ⚙️ How to Run

### Python (Notebook)
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn statsmodels
jupyter notebook Telco_Customer_Churn.ipynb
```

### Power BI
1. Download `Telco_Customer_Churn.pbix`
2. Open with **Power BI Desktop**
3. Use the `Churn?` slicer on each page to compare churned vs. retained customers

---

## 👤 Author

**Vo Quang Khai**
Data Analyst | Finance & Data Science Background
[LinkedIn](https://www.linkedin.com/in/voquangkhaikg2003/) · [GitHub](https://github.com/voquangkhai2003)
