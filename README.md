datasetlink
https://drive.google.com/drive/folders/1ezx1TwpCc4RZwF7v5U0WNM6AQxpw7Dux?usp=sharing
## Dataset

Dataset Download:
https://drive.google.com/...

After downloading, place files in:

data/raw/
data/processed/
# Mule Account Detection using Machine Learning and Explainable AI

## Overview

Money mule accounts are bank accounts used to receive, transfer, or move illegally obtained funds. Such accounts are commonly involved in money laundering, fraud schemes, and financial crimes.

This project uses Machine Learning and Explainable AI (XAI) techniques to identify potential mule accounts from anonymized banking data. The system performs data preprocessing, feature discovery, feature selection, model training, prediction, and automated explanation generation.

The goal is not only to classify suspicious accounts but also to explain why a particular account has been flagged.

---

# Key Features

* Automated mule account detection
* Data preprocessing pipeline
* Missing value analysis
* Feature discovery and selection
* Multiple machine learning models

  * Random Forest
  * XGBoost
  * CatBoost
* Cross-validation evaluation
* Explainable AI reporting
* Real account prediction pipeline
* LLM-generated investigation reports

---

# Project Workflow

```text
Raw Banking Dataset
        │
        ▼
Data Audit
        │
        ▼
Missing Value Analysis
        │
        ▼
Feature Discovery
        │
        ▼
Feature Selection
        │
        ▼
Model Training
        │
        ▼
Cross Validation
        │
        ▼
Best Model Selection
        │
        ▼
Prediction on New Accounts
        │
        ▼
Explainable AI Analysis
        │
        ▼
LLM Generated Investigation Report
```

---

# Dataset

The dataset contains anonymized banking features.

Example:

| Feature | Description                |
| ------- | -------------------------- |
| F1      | Anonymized Banking Feature |
| F115    | Anonymized Banking Feature |
| F520    | Anonymized Banking Feature |
| ...     | ...                        |
| F3924   | Target Variable            |

The actual meaning of feature names is hidden for privacy reasons.

---

# Target Variable

| Value | Meaning        |
| ----- | -------------- |
| 0     | Normal Account |
| 1     | Mule Account   |

---

# Repository Structure

MULE-ACCOUNT-DETECTION
│
├── data
│   ├── raw
│   │   └── DataSet.csv
│   │
│   └── processed
│       ├── cleaned_data.csv
│       ├── encoded_data.csv
│       ├── preprocessed_data.csv
│       └── final_selected_features.csv
│
├── models
│   └── saved_models
│       ├── random_forest.pkl
│       ├── xgboost.pkl
│       ├── final_xgboost.pkl
│       └── catboost.pkl
│
├── notebooks
│   ├── 01_data_audit.ipynb
│   ├── 02_missing_value_analysis.ipynb
│   ├── 03_feature_discovery.ipynb
│   ├── 04_feature_selection.ipynb
│   ├── 05_actual_mule_accounts.ipynb
│   └── all_discovered_features.txt
│
├── reports
│
├── src
│   ├── explainability
│   │
│   ├── llm
│   │   └── generate_report.py
│   │
│   ├── models
│   │   ├── train_rf.py
│   │   ├── train_xgb.py
│   │   ├── train_catboost.py
│   │   ├── final_xgb.py
│   │   ├── cross_validate_rf.py
│   │   ├── cross_validate_xgb.py
│   │   └── cross_validate_catboost.py
│   │
│   ├── pipeline
│   │   └── predict_real_account.py
│   │
│   ├── utils
│
├── requirements.txt
└── README.md

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* CatBoost
* Matplotlib
* SHAP
* Jupyter Notebook

---

# Installation Guide

## Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/MULE-ACCOUNT-DETECTION.git
```

```bash
cd MULE-ACCOUNT-DETECTION
```

---

## Step 2: Create Virtual Environment

### Windows

```bash
python -m venv env
env\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv env
source env/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Option 1: Explore the Complete Analysis

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run notebooks in order:

```text
01_data_audit.ipynb
↓
02_missing_value_analysis.ipynb
↓
03_feature_discovery.ipynb
↓
04_feature_selection.ipynb
↓
05_actual_mule_accounts.ipynb
```

---

# Training Models

## Random Forest

```bash
python src/models/train_rf.py
```

---

## XGBoost

```bash
python src/models/train_xgb.py
```

---

## CatBoost

```bash
python src/models/train_catboost.py
```

---

# Cross Validation

## Random Forest

```bash
python src/models/cross_validate_rf.py
```

## XGBoost

```bash
python src/models/cross_validate_xgb.py
```

## CatBoost

```bash
python src/models/cross_validate_catboost.py
```

---

# Train Final Production Model

```bash
python src/models/final_xgb.py
```

The trained model will be saved inside:

```text
models/saved_models/
```

---

# Predict Real Accounts

To evaluate a real account using the trained model:

```bash
python src/pipeline/predict_real_account.py
```

The pipeline will:

1. Load the trained model
2. Ask you for the account ROW 
3. enter 9001 which is MULE account
4. Process account features
5. Generate prediction
6. Identify mule account probability
7. Produce explainable results

---

# Generate Investigation Report

```bash
python src/llm/generate_report.py
```

The report includes:

* Predicted class
* Confidence score
* Important contributing features
* Human-readable explanation
* Investigation summary

---

# Machine Learning Models

The project compares multiple classifiers:

### Random Forest

* Ensemble learning algorithm
* Good interpretability
* Handles high-dimensional data

### XGBoost

* Gradient boosting framework
* High predictive performance
* Selected as the final production model

### CatBoost

* Gradient boosting algorithm
* Robust against overfitting
* Strong performance on structured data

---

# Explainable AI

The project includes explainability mechanisms that help answer:

* Why was an account classified as suspicious?
* Which features influenced the prediction?
* What factors increased risk?
* What factors reduced risk?

This makes the system more transparent and suitable for financial investigations.

---

# Example Output

```text
Prediction Result

Account Type:
Mule Account

Confidence:
94.7%

Top Contributing Features:

F115
F820
F1320
F2015

Risk Level:
High
```

---

# Applications

* Banking Fraud Detection
* Anti-Money Laundering (AML)
* Financial Crime Investigation
* Risk Monitoring
* Suspicious Account Detection

---

# Future Enhancements

* Real-time transaction monitoring
* Streamlit dashboard
* REST API deployment
* Continuous model retraining
* Advanced explainability dashboard
* Integration with banking systems

---

# Disclaimer

This project is intended for educational, research, and demonstration purposes. Predictions generated by the model should be reviewed by qualified financial crime analysts before operational use.

---

# Author

**Aviraj Abasaheb Murkute**

Machine Learning • Data Analytics • Explainable AI
