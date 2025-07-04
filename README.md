# Sepsis Prediction

This repository contains a unified pipeline for training and evaluating machine learning models for early sepsis prediction in patients with prolonged/chronic critical illness (PCI/CCI), using structured ICU time-series data.

## Problem Definition

- **Task:** Binary classification (`sepsis_3`)
- **Prediction window:** 6 hours before sepsis onset

## Feature Set

### Vitals (3-hour observation window)
- Mean, min, max, standard deviation, and 3-hour delta of:
  - Heart rate
  - Respiratory rate
  - Temperature
  - Systolic, diastolic, and mean arterial pressure
  - SpO₂

### Laboratory values (12-hour window)
- CRP, WBC, hemoglobin, lactate, albumin, pH

### Demographics and comorbidities
- Age, sex  
- Diabetes, chronic kidney disease (CKD), coronary artery disease (CAD), traumatic brain injury (TBI), stroke, heart failure, COPD, hypertension

## Feature Selection

- **Mutual information** via `mutual_info_classif`
- **Correlation filtering** with a threshold of ≥ 0.9 to remove highly collinear predictors

## Models

Implemented using Scikit-learn, XGBoost, and LightGBM. Supported classifiers:
- `XGBClassifier` (XGBoost)
- `AdaBoostClassifier` + `DecisionTreeClassifier` (AdaBoost)
- `RandomForestClassifier` (Random Forest)
- `LGBMClassifier` (LightGBM)

## Hyperparameter Optimization

- Grid search over predefined parameter spaces
- Early stopping via runtime limit (1 hour)
- Best model selected by AUROC on the **validation set**
- Performance additionally evaluated on **internal test set** and **external validation set**

### XGBoost Grid
| Parameter         | Values                       |
|------------------|------------------------------|
| `n_estimators`   | 100, 200, 300                |
| `max_depth`      | 3, 4, 6                      |
| `learning_rate`  | 0.04, 0.08, 0.14             |
| `subsample`      | 0.7, 0.8, 0.9                |
| `colsample_bytree` | 0.5, 0.7, 0.9             |
| `scale_pos_weight` | imbalance ratio           |

### AdaBoost Grid
| Parameter             | Values              |
|----------------------|---------------------|
| `n_estimators`       | 100, 200, 300       |
| `learning_rate`      | 0.04, 0.08, 0.10, 0.16 |
| `estimator__max_depth` | 2, 3              |

### Random Forest Grid
| Parameter             | Values              |
|----------------------|---------------------|
| `n_estimators`       | 100, 200, 300       |
| `max_depth`          | 4, 6, 8             |
| `max_features`       | `'sqrt'`, `'log2'`  |
| `min_samples_split`  | 2, 3                |
| `min_samples_leaf`   | 1, 2                |

### LightGBM Grid
| Parameter             | Values                         |
|----------------------|--------------------------------|
| `n_estimators`       | 100, 200, 300                  |
| `num_leaves`         | 15, 31                         |
| `max_depth`          | 4, 6, 8                        |
| `learning_rate`      | 0.04, 0.08                     |
| `subsample`          | 0.8, 1.0                       |
| `colsample_bytree`   | 0.8, 1.0                       |
| `scale_pos_weight`   | imbalance ratio, ×1.5          |

## Output Files

After training, the following are saved:
| File                             | Description                               |
|----------------------------------|-------------------------------------------|
| `var1_<MODEL>_0.9.model`         | Trained model                             |
| `var1_<MODEL>_0.9_train.csv`     | Predictions on training set               |
| `var1_<MODEL>_0.9_val.csv`       | Predictions on validation set             |
| `var1_<MODEL>_0.9_test.csv`      | Predictions on internal test set          |
| `var1_<MODEL>_0.9_external.csv`  | Predictions on external test set          |

Each `.csv` includes:
- `sepsis_score`: predicted probability
- `sepsis_predicted`: predicted binary outcome (≥ 0.5)
- `sepsis_3`: real outcome 

---

## Model interpretation and calibration (`calibration_and_shap.py`)

This script is used for:

### Calibration curve
- Visualizes agreement between predicted probabilities and observed outcomes
- Uses `calibration_curve` from `sklearn.calibration`

### SHAP Summary plot *(Optional)*
- Visualizes global feature importance using SHAP values
- Requires compatible model and installation of `shap`

### Output
- `<model_name>_calibration_curve.png`
- `<model_name>_shap_summary.png` *(if enabled)*

---

## Directory Structure
├── python_code.py # Main training and prediction script
├── calibration_and_shap.py # Post-hoc interpretation and calibration
├── train.csv # Training data
├── val.csv # Validation data
├── test.csv # Internal test set
├── external.csv # External validation set
└── outputs/
├── var1_XGB_0.9.model
├── var1_XGB_0.9_val.csv
├── var1_XGB_0.9_test.csv
├── var1_XGB_0.9_external.csv
└── var1_XGB_0.9_calibration_curve.png

---

## 📝 License

MIT License
