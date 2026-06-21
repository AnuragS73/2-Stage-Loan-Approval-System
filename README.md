# 🏦 Loan Approval & Loan Amount Prediction System

An end-to-end Machine Learning project that predicts:

- ✅ Whether a loan application will be approved or rejected  
- ✅ The eligible loan amount for approved applicants  
- ✅ Estimated EMI based on loan tenure and interest rate  

The project follows a complete Data Science lifecycle including:

- Data Cleaning  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Classification Modeling  
- Regression Modeling  
- Hyperparameter Tuning  
- Model Deployment using Streamlit  

---

# 🌐 Live Demo

🚀 Try the deployed application here  

**🔗 Live App**  
👉 https://2-stage-loan-approval-system-anurags73.streamlit.app/

---

# 🚀 Project Overview

Financial institutions evaluate multiple factors before approving a loan application.

This project simulates a real-world loan underwriting system by building:

## 1. Loan Approval Classification Model

Predicts:

```text
Approved
Rejected
```

## 2. Loan Amount Regression Model

Predicts:

```text
Eligible Loan Amount
```

## 3. EMI Calculator

Calculates:

```text
Monthly EMI
Total Interest
Total Payment
```

for approved applicants.

---

# 📂 Project Structure

```text
2-Stage-Loan-Approval-System/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── app/
│   └── app.py
│
├── data/
│   ├── loan_approval_dataset.csv
│   ├── loan_approval_dataset_cleaned.csv
│   ├── classification_data.csv
│   └── regression_dataset.csv
│
├── model/
│   ├── loan_approval_model.pkl
│   └── loan_amount_model.pkl
│
└── notebook/
    ├── exploratory_data_analysis.ipynb
    ├── loan_status_prediction_classification.ipynb
    └── loan_amount_prediction.ipynb
```

---

# 📊 Dataset Description

The dataset contains applicant information, financial details, asset information, and loan status.

## Features

| Feature | Description |
|----------|-------------|
| no_of_dependents | Number of dependents |
| education | Applicant education level |
| self_employed | Employment status |
| income_annum | Annual income |
| loan_amount | Requested loan amount |
| loan_term | Loan tenure |
| cibil_score | Credit score |
| residential_assets_value | Residential assets |
| commercial_assets_value | Commercial assets |
| luxury_assets_value | Luxury assets |
| bank_asset_value | Bank assets |
| loan_status | Approved / Rejected |

---

# 🔍 Exploratory Data Analysis

The EDA notebook covers:

## Data Quality Checks

- Missing values  
- Duplicate records  
- Data types  

## Univariate Analysis

- Distribution plots  
- Histograms  
- Boxplots  

## Bivariate Analysis

- Loan status vs income  
- Loan status vs CIBIL score  
- Loan status vs loan amount  

## Correlation Analysis

Heatmap analysis revealed:

### Strongest Positive Indicators

- CIBIL Score  
- Asset Coverage Ratio  
- Wealth CIBIL Score  

### Key Observations

- Higher CIBIL scores significantly increase approval probability  
- Applicants with higher asset coverage are more likely to receive approvals  
- Loan amount alone has limited predictive power compared to creditworthiness  

---

# ⚙️ Feature Engineering

Several business-driven features were created.

## 1. Total Assets

```python
total_assets = (
    residential_assets_value +
    commercial_assets_value +
    luxury_assets_value +
    bank_asset_value
)
```

Purpose:

- Represents overall wealth

---

## 2. Loan Income Ratio

```python
loan_income_ratio = loan_amount / income_annum
```

Purpose:

- Measures borrowing relative to income

---

## 3. Asset Coverage Ratio

```python
asset_coverage_ratio = total_assets / loan_amount
```

Purpose:

- Measures ability to cover the loan using assets

---

## 4. Income Per Dependent

```python
income_per_dependent = income_annum / (no_of_dependents + 1)
```

Purpose:

- Captures financial burden

---

## 5. Wealth CIBIL Score

```python
wealth_cibil_score = total_assets * cibil_score
```

Purpose:

- Combines creditworthiness and net worth

---

# 🎯 Problem 1: Loan Approval Classification

## Target Variable

```text
loan_status
```

### Encoding

| Value | Meaning |
|---------|---------|
| 1 | Approved |
| 0 | Rejected |

---

## Features Used

```python
[
    'no_of_dependents',
    'education',
    'self_employed',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'total_assets',
    'loan_income_ratio',
    'asset_coverage_ratio',
    'income_per_dependent',
    'wealth_cibil_score'
]
```

---

## Models Evaluated

- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

## Classification Results

| Model | ROC-AUC |
|---------|---------|
| Logistic Regression | 0.9058 |
| Random Forest | 0.9954 |
| XGBoost | 0.9960 |

### Winner

🏆 **XGBoost Classifier**

Reason:

- Highest ROC-AUC  
- Highest Recall  
- Excellent generalization  
- Robust to outliers  
- Handles non-linear relationships effectively  

---

# 💰 Problem 2: Loan Amount Prediction

Only approved applications are considered.

## Features Used

```python
[
    'no_of_dependents',
    'education',
    'self_employed',
    'income_annum',
    'loan_term',
    'cibil_score',
    'total_assets',
    'income_per_dependent',
    'asset_coverage_ratio',
    'wealth_cibil_score'
]
```

---

## Models Evaluated

- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  

## Regression Results

| Model | R² Score |
|---------|---------|
| Linear Regression | 0.9216 |
| Random Forest | 0.9953 |
| XGBoost | 0.9959 |
| XGBoost Tuned | 0.9984 |

### Winner

🏆 **Tuned XGBoost Regressor**

Performance:

```text
R² Score = 99.84%
MAE = 248,653
RMSE = 356,333
```

---

# 🔥 Feature Importance

Top drivers of loan approval:

| Feature | Importance |
|----------|-----------|
| cibil_score | 71.70% |
| loan_income_ratio | 13.81% |
| loan_term | 10.67% |
| asset_coverage_ratio | 1.70% |

### Business Insight

Creditworthiness (CIBIL Score) is the strongest factor influencing loan approval decisions.

---

# 🌐 Streamlit Application

The application provides:

## Applicant Inputs

- Dependents  
- Education  
- Employment Status  
- Income  
- Loan Details  
- Asset Information  

## Predictions

- Loan Approval Probability  
- Approval Status  
- Eligible Loan Amount  
- EMI Details  

## Generated Features

- Total Assets  
- Loan Income Ratio  
- Asset Coverage Ratio  
- Income Per Dependent  
- Wealth CIBIL Score  

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/AnuragS73/2-Stage-Loan-Approval-System.git
```

Navigate to project:

```bash
cd 2-Stage-Loan-Approval-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Launch the Streamlit application:

```bash
streamlit run app/app.py
```

Application will run on:

```text
http://localhost:8501
```

Open the browser and test predictions locally.

---

# 📦 Requirements

Major libraries used:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
streamlit
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# 📈 Business Value

This project demonstrates:

- End-to-End Machine Learning Pipeline  
- Credit Risk Assessment  
- Loan Underwriting Automation  
- Feature Engineering for Financial Analytics  
- Model Deployment using Streamlit  
- Classification and Regression Modeling  

---

# 👨‍💻 Author

**Anurag Sarkar**

Data & Business Analyst

### Skills

- SQL  
- Python  
- Power BI  
- Azure Data Explorer (Kusto)  
- Machine Learning  
- Streamlit  
- Data Warehousing  

---

# ⭐ Future Enhancements

- Probability Calibration  
- Applicant Risk Segmentation  
- Model Monitoring Dashboard  
- Cloud Deployment Improvements  
- Automated Retraining Pipeline  
