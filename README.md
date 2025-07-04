# Sepsis Prediction

This repository contains a unified pipeline for training machine learning models to predict sepsis onset within 6 hours in patients with prolonged/chronic critical illness (PCI/CCI), using structured ICU time-series data.

## Problem Definition

- **Target:** Binary classification (`sepsis_3`)
- **Prediction window:** 6 hours

## Features

- **Vitals (3-hour observation window):**
  - Mean, min, max, standard deviation, and 3-hour delta of heart rate, respiratory rate, temperature, systolic/diastolic/mean BP, SpO₂
- **Laboratory values (12-hour window):**
  - CRP, WBC, hemoglobin, lactate, albumin, pH
- **Other variables:**
  - Age, sex, comorbidities (e.g., diabetes, CKD, CAD, TBI)

## Feature Selection

- Mutual Information (`mutual_info_classif`)
- Correlation filtering (threshold ≥ 0.9)

## Models

Implemented using Scikit-learn, XGBoost, and LightGBM.

Supported classifiers:
- XGBoost (`xgb.XGBClassifier`)
- AdaBoost (`AdaBoostClassifier` + `DecisionTreeClassifier`)
- Random Forest (`RandomForestClassifier`)
- LightGBM (`lgb.LGBMClassifier`)

## Hyperparameter Optimization

- Grid search with time constraint (max 1 hour)
- Best model selected by AUROC on the **validation set**
- Performance also evaluated on **independent internal test set** and **external test set**

### XGBoost
| Parameter         | Values                       |
|------------------|------------------------------|
| `n_estimators`   | 100, 200, 300                |
| `max_depth`      | 3, 4, 6                      |
| `learning_rate`  | 0.04, 0.08, 0.14             |
| `subsample`      | 0.7, 0.8, 0.9                |
| `colsample_bytree` | 0.5, 0.7, 0.9             |
| `scale_pos_weight` | class imbalance ratio     |

### AdaBoost
| Parameter             | Values              |
|----------------------|---------------------|
| `n_estimators`       | 100, 200, 300       |
| `learning_rate`      | 0.04, 0.08, 0.10, 0.16 |
| `estimator__max_depth` | 2, 3              |

### Random Forest
| Parameter             | Values              |
|----------------------|---------------------|
| `n_estimators`       | 100, 200, 300       |
| `max_depth`          | 4, 6, 8             |
| `max_features`       | `'sqrt'`, `'log2'`  |
| `min_samples_split`  | 2, 3                |
| `min_samples_leaf`   | 1, 2                |

### LightGBM
| Parameter             | Values                         |
|----------------------|--------------------------------|
| `n_estimators`       | 100, 200, 300                  |
| `num_leaves`         | 15, 31                         |
| `max_depth`          | 4, 6, 8                        |
| `learning_rate`      | 0.04, 0.08                     |
| `subsample`          | 0.8, 1.0                       |
| `colsample_bytree`   | 0.8, 1.0                       |
| `scale_pos_weight`   | imbalance ratio, ×1.5          |

## Outputs

The following files are saved to the working directory:

| File                        | Description                                  |
|-----------------------------|----------------------------------------------|
| `var1_<MODEL>_0.9.model`    | Trained model (XGB / ADA / RF / LGB)        |
| `var1_<MODEL>_0.9_val.csv`  | Validation predictions                      |
| `var1_<MODEL>_0.9_train.csv`| Train set predictions                       |
| `var1_<MODEL>_0.9_test.csv` | Test set predictions                        |
| `var1_<MODEL>_0.9_external.csv` | External set predictions             |

All `.csv` files contain:
- `sepsis_score`: predicted probability
- `sepsis_predicted`: binary label (threshold = 0.5)
- `sepsis_3`: true label (if available)

## 📝 License

MIT License.
