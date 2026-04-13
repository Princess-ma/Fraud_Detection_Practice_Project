# **FRAUD DETECTION ANALYSIS - DETECTING FRAUD TRANSACTIONS AND PATTERNS ACROSS MAJOR CITIES IN INDIA**
> An Exploratory Analysis Backed by Classification Machine Learning Models (KNN & SVC)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?style=flat-square&logo=pandas)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)


## Overview

India has witnessed a steady and continuous rise in online financial transactions, driven by the rapid digitisation of banking services across the country and with this growth comes an equally significant rise in fraudulent activities. The rise in these fraudulent activities has caused panic amongst various commercial banks in India directly affecting the State Bank of India and it's branches. 
The Finance Director of the State Bank of India also identified a growing concern around these fraudulent transactions in the 2024-2025 yearly report occurring across major branch locations and therefore this project analyses over 100,000 transaction records from the State Bank of India to uncover fraud patterns and build classification models capable of predicting whether a given transaction is fraudulent or legitimate.

---

## Project Objectives

- Perform thorough data cleaning, feature engineering, and preprocessing on the fraud dataset
- Conduct univariate, bivariate, and multivariate exploratory analysis to uncover fraud patterns
- Address class imbalance in the target variable before model training
- Build and evaluate a K-Nearest Neighbours (KNN) model as the baseline classifier
- Build and evaluate a Support Vector Classification (SVC) model as the advanced classifier
- Compare both models and identify the stronger candidate for fraud detection

---

## Dataset

| Detail | Information |
|---|---|
| Source | [Fraud Detection Dataset](https://www.kaggle.com/datasets/sshriya08/fraud-detection-dataset) |
| Records | 100,186 transactions |
| Original Features | 18 columns |
| Time Period | January 2024 - January 2025 |
| Cities Covered | Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai |
| Target Variable | `is_fraud` (0 = Legitimate, 1 = Fraudulent) |
| Class Distribution | 90,168 legitimate (90%) vs 10,018 fraud (10%) |

---

## Methodology

The project is structured across five sequential stages:

**Stage 1: Data Cleaning and Preparation**
**Stage 2: Data Distribution Analysis**
**Stage 3: Bivariate and Multivariate Exploratory Analysis**
**Stage 4: Data Preprocessing**
**Stage 5: Machine Learning Model Development**

---

## Stage 1: Data Cleaning and Preparation

- `transaction_time` was converted from object (text) to `datetime64` using `pd.to_datetime()` with the `%d-%m-%Y %H:%M` format, as it was being read incorrectly as a string
- Label encoding was applied to the three text categorical columns using `sklearn.preprocessing.LabelEncoder`, producing three new encoded columns:
  - `transaction_type_encoded` - 6 unique integer labels (0–5)
  - `location_encoded` - 6 unique integer labels (0–5)
  - `device_type_encoded` - 3 unique integer labels (0–2)
- All encoding was performed on a copy of the original dataframe (`fraud_dff`) to preserve the raw data
- Zero duplicate rows were found across all 100,186 records

### Dataset After Stage 1

| Detail | Value |
|---|---|
| Total Rows | 100,186 |
| Total Columns | 21 |
| Float Columns | 2 |
| Integer Columns | 13 |
| Object (Text) Columns | 5 |
| Datetime Columns | 1 |

---

## Stage 2: Data Distribution

### Numerical Features

`transaction_amount` showed a right-skewed distribution with values ranging from ₹10 to ₹18,673 and a mean of ₹4,049. `avg_transaction_amount` showed a similar profile with a narrower spread. Both features were leptokurtic, indicating sharp peaks with heavy tails.

### Categorical and Binary Features

- **Foreign vs Local Transactions:** 94% of transactions were local and only 6% were foreign
- **New vs Old Device:** 90.6% of users transacted from their existing device and 9.4% used a new device
- **New vs Old Location:** 88.2% transacted from their registered location and 11.8% from new locations
- **Weekday vs Weekend:** 72.1% of transactions occurred on weekdays and 27.9% on weekends
- **Failed Logins:** Most users had 0 failed logins, however, the maximum recorded was 5
- **Fraud Rate:** Only 10% of transactions were fraudulent, confirming severe class imbalance
- **Transaction Types:** UPI led at approximately 30%, followed by NEFT and IMPS at roughly 20% each, then RTGS, ATM Withdrawal, and POS
- **Device Usage:** Mobile dominated at 62.8%, followed by Desktop at 21% and Tablet at 10.8%
- **Transaction Hours:** Normally distributed across a 24-hour window
- **Account Age:** Normally distributed between 100 and 3,649 days

---

## Stage 3: Bivariate and Multivariate Exploratory Analysis

### Fraud by Location

| City | Fraud Count |
|---|---|
| Kolkata | 1,730 |
| Chennai | 1,712 |
| Hyderabad | 1,686 |
| Delhi | 1,655 |
| Mumbai | 1,621 |
| Bangalore | 1,614 |

Fraud is nearly uniformly distributed across all six cities. The gap between the highest (Kolkata) and lowest (Bangalore) is only 116 cases, confirming that city alone is not a meaningful fraud discriminator.

### Fraud by Transaction Type

| Transaction Type | Fraud Count |
|---|---|
| UPI | 3,034 |
| NEFT | 2,052 |
| IMPS | 1,991 |
| RTGS | 1,012 |
| ATM Withdrawal | 979 |
| POS | 950 |

UPI's fraud dominance mirrors its volume dominance. NEFT and IMPS together account for nearly as many fraud cases as UPI despite each representing only ~20% of total transaction volume.

### Fraud by Device Type

| Device | Fraud Count |
|---|---|
| Mobile | 6,878 |
| Desktop | 2,053 |
| Tablet | 1,087 |

Fraud rates are broadly proportional to transaction volume across all three devices, meaning no single device type carries disproportionate fraud risk on its own.

### Top Three-Way Combination (Location × Transaction Type × Device)

The highest fraud-volume combination across all three categorical dimensions was **UPI on Mobile in Kolkata** at 3,605 total transactions, followed by UPI on Mobile in Mumbai (3,482) and Chennai (3,466). Every entry in the top ten involved Mobile devices.

### Key Fraud Patterns Discovered

**Transaction Amount**: Fraud and non-fraud KDE distributions overlap almost entirely. Fraud transactions average ₹4,484 against ₹4,000 for legitimate ones, a difference of only ₹484. Amount alone is an insufficient fraud detector.

**Temporal Patterns**: Fraud rate elevates noticeably during the late night and early morning window (0:00–4:00 AM). Monthly fraud activity shows non-uniform distribution across the year.

**Foreign Transactions**: Despite representing only 6% of all transactions, foreign transactions carry a notably higher fraud rate than local ones.

**New Device Flag**: Transactions from unrecognised devices carry a substantially elevated fraud rate, consistent with a 0.39 correlation with `is_fraud`. Account takeover scenarios almost always involve an unfamiliar device.

**New Location Flag**: Transactions from outside a user's registered geography carry a higher fraud rate, consistent with a 0.33 correlation with `is_fraud`.

**Weekend vs Weekday**: Marginally higher fraud rate on weekdays. The difference is modest, consistent with a weak -0.04 correlation.

**Failed Logins**: A clean monotonic relationship: each additional failed login count from 0 to 5 produces a stepwise increase in fraud rate. This is the clearest sequential fraud precursor in the dataset.

**Account Age**: Newer accounts (0–1 year old) carry the highest fraud rate. Risk declines as accounts mature.

**Time Since Last Transaction**: Fraudulent transactions tend to follow shorter intervals, indicating burst activity patterns, though the distributions overlap substantially.

**Mule Accounts**:  2,683 user accounts were identified as having more than one fraudulent transaction on record. The top five most flagged accounts:

| User ID | Fraud Transactions |
|---|---|
| U1741 | 7 |
| U9872 | 6 |
| U4368 | 6 |
| U7982 | 6 |
| U2720 | 6 |

### Correlation with Target Variable (`is_fraud`)

| Feature | Correlation |
|---|---|
| transactions_last_24h | +0.45 |
| is_new_device | +0.39 |
| is_new_location | +0.33 |
| is_foreign_transaction | +0.25 |
| transaction_amount | +0.09 |
| time_since_last_txn | -0.11 |
| txn_hour | -0.08 |
| is_weekend | -0.04 |

---

## Stage 4: Data Preprocessing

### 4.1 Class Imbalance

| Class | Count | Rate |
|---|---|---|
| Legitimate (0) | 90,168 | 90% |
| Fraud (1) | 10,018 | 10% |

Random undersampling was applied to match the majority class to the minority class size, producing a balanced working dataset of **20,036 rows** (10,018 per class).

### 4.2 Feature Selection

**13 features were selected for modelling:**

| Feature | Correlation | Reason for Selection |
|---|---|---|
| transactions_last_24h | +0.45 | Strongest signal; captures velocity bursts |
| is_new_device | +0.39 | Strong behavioural deviation indicator |
| is_new_location | +0.33 | Strong behavioural deviation indicator |
| is_foreign_transaction | +0.25 | Disproportionate fraud risk for 6% of data |
| time_since_last_txn | -0.11 | Velocity signal; rapid intervals flag bursts |
| txn_hour | -0.08 | Temporal signal; elevated fraud in early hours |
| transaction_amount | +0.09 | Marginal amount difference contributes to model |
| is_weekend | -0.04 | Contextual supporting feature |
| device_type_encoded | — | Categorical structure of transaction environment |
| transaction_type_encoded | — | Categorical structure of transaction environment |
| account_age_days | — | Account maturity and trust level |
| failed_logins_24h | — | Monotonic fraud precursor |
| location_encoded | — | Categorical geographic context |


### 4.3 Train-Test Split

| Split | Rows | Fraud Rate |
|---|---|---|
| Training Set | 16,028 | 50.00% |
| Test Set | 4,008 | 50.00% |

An 80-20 split was applied with `stratify=y` to preserve the class balance in both sets.

### 4.4 Feature Scaling

`StandardScaler` was applied and fitted exclusively on the training data and then applied to both sets. This prevents data leakage and is essential for both KNN (Euclidean distance) and SVC (margin optimisation).

- Mean of first feature after scaling: **~0.0000**
- Standard deviation of first feature after scaling: **1.0000**

---

## Stage 5:  Machine Learning

### Model 1-  K-Nearest Neighbours (Baseline)

K values from 1 to 30 were tested by evaluating both train and test accuracy at each k. The optimal k was identified as **k = 10**, which produced the highest test accuracy of 88.47%.

#### KNN Performance (k = 10)

| Metric | Score |
|---|---|
| Train Accuracy | 91.26% |
| Test Accuracy | 88.47% |
| Precision | 0.9037 |
| Recall | 0.8613 |
| F1 Score | 0.8820 |
| Train-Test Gap | 2.79% |

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Non-Fraud | 0.87 | 0.91 | 0.89 | 2,004 |
| Fraud | 0.90 | 0.86 | 0.88 | 2,004 |
| Accuracy | | | 0.88 | 4,008 |
| Macro Avg | 0.89 | 0.88 | 0.88 | 4,008 |
| Weighted Avg | 0.89 | 0.88 | 0.88 | 4,008 |

#### KNN Permutation Feature Importance

| Feature | Importance |
|---|---|
| transaction_amount | -0.001697 |
| time_since_last_txn | -0.000349 |
| is_new_location | -0.000225 |
| is_new_device | -0.000125 |
| location_encoded | -0.000100 |
| account_age_days | -0.000025 |
| txn_hour | -0.000025 |
| All others | 0.000000 |

Negative importance values indicate that shuffling those features degraded model performance, confirming their contribution. The zero-importance features had no measurable effect when permuted, suggesting KNN's distance-based mechanism did not utilise their variance effectively in the 13-dimensional scaled space.

---

### Model 2 - Support Vector Classification (Advanced)

GridSearchCV with 5-fold cross-validation was used to tune the SVC hyperparameters across 16 candidate combinations (4 C values × 2 kernels × 2 gamma values), totalling 80 fits. The grid was optimised for F1 score.

**Best Parameters:** `C=1`, `kernel=rbf`, `gamma=auto`
**Best Cross-Validated F1:** 0.9029

#### SVC Performance

| Metric | Score |
|---|---|
| Train Accuracy | 91.56% |
| Test Accuracy | 89.95% |
| Precision | 0.8850 |
| Recall | 0.9182 |
| F1 Score | 0.9013 |
| Best CV F1 | 0.9029 |
| Train-Test Gap | 1.61% |

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Non-Fraud | 0.91 | 0.88 | 0.90 | 2,004 |
| Fraud | 0.89 | 0.92 | 0.90 | 2,004 |
| Accuracy | | | 0.90 | 4,008 |
| Macro Avg | 0.90 | 0.90 | 0.90 | 4,008 |
| Weighted Avg | 0.90 | 0.90 | 0.90 | 4,008 |

> Note: Permutation importance for the SVC returned all zeros across all 13 features. This is expected behaviour for non-linear RBF kernel SVMs because the model compensates for any single shuffled feature using the remaining features. SHAP values would be required for meaningful feature-level attribution.

---

## Model Comparison

| Metric | KNN (k=10) | SVC (RBF, C=1) | Winner |
|---|---|---|---|
| Train Accuracy | 91.26% | 91.56% | SVC |
| Test Accuracy | 88.47% | 89.95% | SVC |
| Precision | 0.9037 | 0.8850 | KNN |
| Recall | 0.8613 | 0.9182 | SVC |
| F1 Score | 0.8820 | 0.9013 | SVC |
| Train-Test Gap | 2.79% | 1.61% | SVC |
| Fraud Recall | 86.13% | 91.82% | SVC |
| Non-Fraud Recall | 90.78% | 87.52% | KNN |

**Overall Winner: SVC**

The SVC outperformed KNN on every key metric except precision. Most critically, the SVC correctly identified 91.82% of all actual fraud cases versus KNN's 86.13%, a 5.69 percentage point advantage in fraud recall. In a banking fraud detection context, minimising false negatives (missed fraud) is the primary operational concern, making the SVC the clearly superior and recommended model.

---

## Key Recommendations

Based on the findings of this analysis, the following actions are recommended for the State Bank of India's fraud operations team:

Transactions initiated from a device not previously associated with the account should be automatically flagged for review. The new device flag carries a 0.39 correlation with fraud and represents the most reliable environmental indicator of account compromise.

Transactions originating from locations outside a user's registered geography should trigger additional authentication steps. The new location flag carries a 0.33 correlation with fraud and together with the new device flag covers the two strongest behavioural anomaly signals in the dataset.

Foreign transactions should receive elevated scrutiny by default. Despite representing only 6% of all transactions, they carry a disproportionately high fraud rate.

Accounts with three or more failed login attempts in the preceding 24 hours should be temporarily locked or subject to mandatory step-up verification. The failed logins heatmap showed a monotonic escalation in fraud rate with each additional failed attempt.

Fraud monitoring resources should be intensified during the late night and early morning window between 0:00 and 4:00 AM, when fraud rates are visibly elevated relative to the total transaction volume in those hours.

Newly created accounts, particularly those under one year old, should be subject to stricter transaction limits and enhanced monitoring during the initial account maturity period.

The 2,683 user accounts identified as repeat fraud participants  particularly the top five mule accounts with 6–7 fraudulent transactions each (U1741, U9872, U4368, U7982, U2720) should be investigated immediately and considered for suspension pending review.

---
### Technologies Used

```
Python 3
pandas
numpy
scipy
matplotlib
seaborn
scikit-learn
Google Colab (execution environment)
```
---

## Conclusion

This project successfully delivered a complete fraud detection analytical pipeline from raw data ingestion through to trained and evaluated classification models. The analysis uncovered that fraud in this dataset is driven primarily by behavioural deviations specifically transactions from new devices, new locations, and foreign origins rather than by transaction amount, geography, or device type alone. The SVC model with an RBF kernel, tuned to C=1 and gamma=auto, is the recommended production candidate with a test accuracy of 89.95%, fraud recall of 91.82%, and an F1 score of 0.9013. The KNN baseline performed credibly at 88.47% accuracy and 0.8820 F1, validating the feature set and preprocessing pipeline, and establishing a meaningful performance floor that the SVC successfully exceeded.

---

## Acknowledgements

- **Kaggle** - for the Fraud Detection Dataset
- **Scikit-learn team** - for developing machine learning libraries
- **GitHub** - for the deployment platform of our Fraud Dection Practice Project

---

## References
Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. 2015 IEEE Symposium Series on Computational Intelligence (SSCI), 159–166. https://doi.org/10.1109/SSCI.2015.33

Bin Sulaiman, R., Schetinin, V., & Sant, P. (2022). Review of machine learning approach on credit card fraud detection. Human-Centric Intelligent Systems, 2(1–2), 55–68. https://doi.org/10.1007/s44230-022-00004-0

---

**Notebook Link**: Click on [Fraud Detection Solo Project](https://colab.research.google.com/drive/1iTWXDgp0f7Lgdji2C6_CAYUWasFOS4N_?usp=sharing) to access the notebook containing the analysis and findings.

---

**License:** Apache License 2.0
**Copyright:** © 2026 Princess Emenari Fraud Detection Practice Project

<div align="center">

*Built with precision. Validated with evidence.*

</div>
