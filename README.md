# Breast Cancer Selective Classifier (Logistic Regression)

# Problem Statement
Built a reliable breast cancer classifier that prioritizes high recall (minimizing missed malignant cases) and supports a selective prediction / referral workflow. The model predicts when confident and defers to human review when uncertain.# Breast Cancer Selective Classifier (Logistic Regression)
### High-Recall Screening + “Defer to Human” Referral Workflow

## Summary 
- **Goal:** Predict malignant vs benign breast cancer with **high recall**, and **defer uncertain cases** for human review.
- **Data:** Wisconsin Breast Cancer (Diagnostic) dataset (Kaggle; **n=569**).
- **Model:** Logistic Regression in a leakage-safe `scikit-learn` Pipeline (impute + scale + LR).
- **Key idea:** Choose the **decision threshold from out-of-fold (OOF)** predictions (not 0.5), then apply a **selective prediction** policy using uncertainty = `min(p, 1-p)`.
- **Headline result:** Conservative policy achieves **0 missed malignant cases** with ~**55% auto-coverage** on test (remaining cases referred).

---

## Problem Statement
Built a reliable breast cancer classifier that prioritizes **high recall** (minimizing missed malignant cases) and supports a **selective prediction / referral workflow**: the model predicts when confident and defers to human review when uncertain.

---

## Dataset
- Source: Kaggle — Breast Cancer Wisconsin (Diagnostic)  
  https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

---

## Clinical Motivation
In screening/diagnostic workflows, **false negatives are more costly than false positives**. A missed malignancy can delay treatment, while false positives typically trigger follow-up imaging/biopsy.  
This project frames classification as **risk triage**, not just label prediction.

---

## Approach

### Model
- Logistic Regression (L2-regularized)

### Pipeline (leakage-safe)
- `SimpleImputer(strategy="median")`
- `StandardScaler()`
- `LogisticRegression(solver="liblinear", C=0.2, max_iter=3000)`

### Threshold + selective prediction
1. **Pick decision threshold** on Train/Val using **OOF predictions** to target **Recall ≥ 0.98**.
2. Define uncertainty: `u = min(p, 1-p)` (near 0 = confident).
3. Apply referral policy:  
   - **Auto-handle** if `u ≤ u_th`  
   - **Refer** otherwise

---

## Leakage Prevention
- Strict train/test split; test held out until final reporting.
- Decision threshold chosen using **OOF predictions** on training data only.
- Preprocessing + model wrapped in a single `scikit-learn` **Pipeline** so each CV fold fits preprocessing only on its training fold.

---

## Key Results

### Test set (selected referral policies)
| Policy | Auto-coverage | Auto accuracy | Missed malignant (auto-only) | Miss upper bound |
|---|---:|---:|---:|---:|
| High coverage | 0.930 | 0.981 | 1 | 0.126 |
| Balanced | 0.912 | 0.990 | 1 | 0.126 |
| **Conservative** | **0.553** | **1.000** | **0** | **0.084** |

**Interpretation:** You can trade off automation vs safety. The conservative policy is most clinically cautious (no missed malignant cases among auto-handled predictions on test).

### OOF-selected policies (Train/Val)
| Policy | Auto-coverage | Auto accuracy | Miss | Miss upper bound |
|---|---:|---:|---:|---:|
| High coverage | 0.949 | 0.988 | 0.0176 | 0.0507 |
| Balanced | 0.925 | 0.993 | 0.0176 | 0.0507 |
| Conservative | 0.558 | 1.000 | 0.0000 | 0.0215 |

---

## Probability Quality
- Reports **Brier score** and **calibration curve** on the held-out test set to verify predicted probabilities are meaningful (not just well-ranked).

---

## Model Interpretability
- Displays **top coefficients** (absolute value) from the logistic regression model to show which standardized features are most influential.

---

## Visuals
- Top 10 coefficients
- Calibration curve (test)
- Coverage vs Auto Accuracy (test sweep)
- Coverage vs Miss Rate (auto-only)

---

## Limitations
- Uncertainty signal is derived directly from predicted probability (`min(p, 1-p)`), which is a useful heuristic but not full epistemic uncertainty.
- Results depend on one dataset + one split; performance may change under dataset shift or alternative splits.

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Download dataset: data.csv from Kaggle and place it in the project root.
3. Run the notebook: jupyter notebook breast_cancer_selective_classifier.ipynb

# Dataset

- Source file : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

# Clinical Importance
- In breast cancer screening and diagnostic workflows, the cost of a false negative is higher than the cost of a false positive. A missed malignancy can delay treatment and worsen outcomes, while false positives generally lead to additional imaging or biopsy. Because of this asymmetry, models optimized for overall accuracy can be unsafe in practice. Clinical decision support tools should instead prioritize high sensitivity, produce reliable risk estimates, and handle uncertainty explicitly rather than forcing a prediction on every case.
- This project reframes breast cancer classification as a risk triage problem: the goal is not just to predict, but to support safe clinical workflows.

# Leakage prevention 
- I kept a strict train/test split. The test set is held out until the very end.
- Selected the decision threshold using out-of-fold (OOF) predictions from cross-validation on the training portion only (so the threshold isn’t tuned on the test set).
- Wrapped preprocessing (imputation and scaling) and the model in a single scikit-learnp Pipeline, so preprocessing is fit only on training folds and applied to validation folds without leaking information.
- Reported final performance using the held-out test set to estimate generalization.

# Approach

# Models

- Baseline model: Logistic Regression (L2-regularized)
- Pipeline:
  - `SimpleImputer(strategy="median")`
  - `StandardScaler()`
  - `LogisticRegression(solver="liblinear", C=0.2, max_iter=3000)`

# Metrics

Evaluated on both ranking performance and probability quality:
- AUROC, AUPRC
- Precision, Recall, Specificity, FNR (at the chosen decision threshold)
- Brier score and calibration curve (probability calibration)

# Key Results (tables)
Test referral policy results
| policy | u_th | coverage | auto_acc | miss | miss_ub | auto_n | referred_n | fn_auto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| High coverage | 0.294628 | 0.929825 | 0.981132 | 0.023810 | 0.125659 | 106 | 8 | 1 |
| Balanced | 0.224942 | 0.912281 | 0.990385 | 0.023810 | 0.125659 | 104 | 10 | 1 |
| Conservative | 0.012727 | 0.552632 | 1.000000 | 0.000000 | 0.084084 | 63 | 51 | 0 |

Selected policies (OOF)
| policy | u_th | coverage | auto_acc | miss | miss_ub | auto_n | referred_n |
|---|---:|---:|---:|---:|---:|---:|---:|
| High coverage | 0.294628 | 0.949451 | 0.988426 | 0.017647 | 0.050704 | 432.0 | 23.0 |
| Balanced | 0.224942 | 0.925275 | 0.992874 | 0.017647 | 0.050704 | 421.0 | 34.0 |
| Conservative | 0.012727 | 0.558242 | 1.000000 | 0.000000 | 0.021466 | 254.0 | 201.0 |

OOF policy sweep (first 12 rows)
| u_th | coverage | auto_acc | fn_auto | miss | miss_ub | auto_n | referred_n |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000013 | 0.050549 | 1.000000 | 0 | 0.000000 | 0.021466 | 23 | 432 |
| 0.000055 | 0.074725 | 1.000000 | 0 | 0.000000 | 0.021466 | 34 | 421 |
| 0.000103 | 0.096703 | 1.000000 | 0 | 0.000000 | 0.021466 | 44 | 411 |
| 0.000176 | 0.120879 | 1.000000 | 0 | 0.000000 | 0.021466 | 55 | 400 |
| 0.000267 | 0.142857 | 1.000000 | 0 | 0.000000 | 0.021466 | 65 | 390 |
| 0.000416 | 0.167033 | 1.000000 | 0 | 0.000000 | 0.021466 | 76 | 379 |
| 0.000605 | 0.189011 | 1.000000 | 0 | 0.000000 | 0.021466 | 86 | 369 |
| 0.000827 | 0.213187 | 1.000000 | 0 | 0.000000 | 0.021466 | 97 | 358 |
| 0.001031 | 0.235165 | 1.000000 | 0 | 0.000000 | 0.021466 | 107 | 348 |
| 0.001431 | 0.257143 | 1.000000 | 0 | 0.000000 | 0.021466 | 117 | 338 |
| 0.001818 | 0.281319 | 1.000000 | 0 | 0.000000 | 0.021466 | 128 | 327 |
| 0.002171 | 0.303297 | 1.000000 | 0 | 0.000000 | 0.021466 | 138 | 317 |

Top 10 coefficients (by absolute value)
| feature | coef | abs_coef |
|---|---:|---:|
| texture_worst | 0.772298 | 0.772298 |
| radius_se | 0.653652 | 0.653652 |
| radius_worst | 0.610333 | 0.610333 |
| area_worst | 0.590953 | 0.590953 |
| symmetry_worst | 0.590299 | 0.590299 |
| concave points_mean | 0.559750 | 0.559750 |
| perimeter_worst | 0.545051 | 0.545051 |
| area_se | 0.510446 | 0.510446 |
| concave points_worst | 0.502319 | 0.502319 |
| texture_mean | 0.483583 | 0.483583 |

# Takeaways:

# Visuals

- Top 10 features
- Calibration curve
- Coverage vs Auto Accuracy (test set sweep)
- Coverage vs Miss Rate 

# Limitations

- The uncertainty signal is derived directly from the model’s predicted probability, which is a useful heuristic but not a full measure of model uncertainty.
- Results are based on one dataset and one train/test split so performance and calibration could change under different splits or on truly external data.

# How to Run

1. Install dependencies pip install -r requirements.txt
2. Download dataset: data.csv from Kaggle and place it in the project root.
3. Run the notebook jupyter notebook breast_cancer_selective_classifier.ipynb

# Future work
- I want to use a more robust uncertainty/abstention method.
- Compare against a stronger non-linear baseline (e.g., gradient boosting) and apply calibration.
- Stress test generalization with repeated splits/seeds and an external dataset to measure stability under dataset shift.

