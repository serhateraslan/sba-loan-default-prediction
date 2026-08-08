# sba-loan-default-prediction
Predicting SBA loan default risk using Decision Tree and LightGBM.

# SBA Loan Repayment Prediction

A binary classification project comparing a traditional single Decision Tree against a LightGBM gradient boosting model to predict SBA loan repayment outcomes on a large tabular dataset.

This is a learning-oriented machine learning portfolio project. Its main purpose was to build a focused, practical understanding of the differences between a single Decision Tree and a gradient boosting ensemble (LightGBM) when applied to the same real-world tabular dataset and target variable.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objective](#objective)
- [Dataset](#dataset)
- [Target Variable](#target-variable)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling Approach](#modeling-approach)
- [Evaluation](#evaluation)
- [Tech Stack](#tech-stack)
- [Project Scope and Limitations](#project-scope-and-limitations)

---

## Project Overview

This project uses a large dataset of SBA-backed small business loans to predict loan repayment outcomes. Two tree-based models — a single Decision Tree and LightGBM — are trained and compared on the same preprocessed dataset.

The comparison was intentionally kept narrow: only Decision Tree and LightGBM were trained. Other algorithms such as Random Forest or Logistic Regression were deliberately excluded, since the goal of the project was to study the practical difference between a single decision tree and a gradient boosting approach on the same tabular data, rather than to benchmark a wide range of algorithms.

---

## Objective

The goal of this project is to predict the `MIS_Status` outcome of SBA-backed small business loans, framed as a binary classification problem.

The project demonstrates the following steps:

- Data cleaning
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Categorical encoding
- Target leakage prevention
- Decision Tree classification
- LightGBM classification
- Confusion matrix analysis
- Precision, Recall, F1 Score, and Accuracy evaluation
- ROC-AUC evaluation
- Feature importance analysis
- Model comparison between Decision Tree and LightGBM

---

## Dataset

The dataset contains approximately **895,641** loan records after cleaning. The original dataset is **not included** in this repository due to its large file size.

### Key Columns

| Column | Description |
|---|---|
| `State` | State of the borrowing business |
| `Bank` | Bank associated with the loan |
| `BankState` | State where the bank is located |
| `NAICS` | Industry classification code |
| `ApprovalYear` | Year the loan was approved |
| `Term` | Loan commitment period, in months |
| `NoEmp` | Number of employees |
| `NewExist` | New or existing business indicator |
| `CreateJob` | Number of jobs created |
| `RetainedJob` | Number of jobs retained |
| `FranchiseCode` | Franchise information |
| `UrbanRural` | Urban/rural classification |
| `LowDoc` | Low-documentation loan indicator |
| `DisbursementGross` | Gross amount disbursed |
| `GrAppv` | Gross amount approved by the bank |
| `SBA_Appv` | Amount approved/guaranteed by the SBA |
| `MIS_Status` | Target variable |
| `LoanStatus` | Loan status |

---

## Target Variable

The target variable is `MIS_Status`, which represents the final outcome of each loan and is used as a binary classification target.

This project treats `MIS_Status` strictly as an encoded binary label for modeling purposes. No claims are made here about the precise real-world business definition of each encoded value beyond what is captured in the dataset itself.

---

## Data Preprocessing

The following preprocessing steps were applied to the cleaned dataset before modeling.

### Target Leakage Prevention

`LoanStatus` was removed from the feature set prior to modeling, since it directly reflects the loan's status/outcome and could introduce target leakage with respect to `MIS_Status`.

### Bank Column Removal

`Bank` was removed from the final model inputs because it contains a very high number of unique categorical values, which makes it impractical to encode directly.

### LowDoc Encoding

The `LowDoc` feature was converted into a binary numeric format:

- `N` → `0`
- `Y` → `1`

### One-Hot Encoding

`State` and `BankState` were one-hot encoded using pandas:

```python
pd.get_dummies()
```

---

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed on the cleaned dataset as part of the workflow, prior to encoding and model training, to support the preprocessing and modeling decisions described above.

---

## Modeling Approach

Two tree-based classification models were trained and compared on the same preprocessed dataset:

### Decision Tree

A single Decision Tree classifier, used as a baseline, interpretable tree-based model.

### LightGBM

A LightGBM gradient boosting classifier, used to compare a boosted ensemble approach against the single Decision Tree baseline.

### Why Only These Two Models

Random Forest and Logistic Regression were intentionally not included in this project. The scope was deliberately limited to a Decision Tree vs. LightGBM comparison in order to keep the analysis focused on the practical differences between a single tree and a gradient boosting model on the same large tabular dataset.

---

## Evaluation

Both models were evaluated using the following methods and metrics:

- Confusion matrix analysis
- Precision
- Recall
- F1 Score
- Accuracy
- ROC-AUC
- Feature importance analysis
- Direct comparison of Decision Tree vs. LightGBM results

---

## Tech Stack

- **Python**
- **pandas** — data cleaning, preprocessing, and one-hot encoding
- **scikit-learn** — Decision Tree classification and evaluation metrics
- **LightGBM** — gradient boosting classification

---

## Project Scope and Limitations

- This project was built for learning and portfolio purposes, with a deliberately narrow model comparison (Decision Tree vs. LightGBM only).
- The raw dataset is not included in this repository due to its size.
- No hyperparameter tuning results, performance metrics, or feature importance values are included in this document beyond what is explicitly documented in the project's code/notebooks.
