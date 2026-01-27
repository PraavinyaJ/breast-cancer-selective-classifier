# Breast Cancer Selective Classifier (Logistic Regression)

# Problem Statement
Built a reliable breast cancer classifier that prioritizes high recall (minimizing missed malignant cases) and supports a selective prediction / referral workflow. The model predicts when confident and defers to human review when uncertain.

# Dataset

- Source File : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

# Approach

# Models

- Baseline model:** Logistic Regression (L2-regularized)
- Pipeline:
  - 'SimpleImputer(strategy="median")`
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


# Visuals

- Top 10 features
- Calibration curve
- Coverage vs Auto Accuracy (test set sweep)
- Coverage vs Miss Rate (auto-only)** (test set sweep)

# Limitations

- The uncertainty signal is derived directly from the model’s predicted probability, which is a useful heuristic but not a full measure of model uncertainty.
- Results are based on one dataset and one train/test split so performance and calibration could change under different splits or on truly external data.

# How to Run

1. Install dependencies pip install -r requirements.txt
2. Download dataset Download data.csv from Kaggle and place it in the project root.
3. Run the notebook jupyter notebook breast_cancer_selective_classifier.ipynb

# Future work
- I want to use a more robust uncertainty/abstention method.
- Compare against a stronger non-linear baseline (e.g., gradient boosting) and apply calibration.
- Stress test generalization with repeated splits/seeds and an external dataset to measure stability under dataset shift.

