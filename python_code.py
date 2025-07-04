import pandas as pd
import numpy as np
import os
import time
import joblib
import warnings
from itertools import product
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE_PATH =  "C:/Your/Path/Here"  # <-- Change to your real path
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
VAL_PATH = os.path.join(BASE_PATH, "val.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
RICD_PATH = os.path.join(BASE_PATH, "external.csv")

FEATURES = [
    'age', 'sex', 'wbc', 'plt', 'crp', 'hb', 'albumin', 'lactate', 'pH',
    'avg_heart_rate', 'min_heart_rate', 'max_heart_rate', 'sd_heart_rate', 'delta_3h_heart_rate',
    'avg_respiratory_rate', 'min_respiratory_rate', 'max_respiratory_rate', 'sd_respiratory_rate', 'delta_3h_respiratory_rate',
    'avg_temperature', 'min_temperature', 'max_temperature', 'sd_temperature', 'delta_3h_temperature',
    'avg_systolic_BP', 'min_systolic_BP', 'max_systolic_BP', 'sd_systolic_BP', 'delta_3h_systolic_BP',
    'avg_diastolic_BP', 'min_diastolic_BP', 'max_diastolic_BP', 'sd_diastolic_BP', 'delta_3h_diastolic_BP',
    'avg_mean_AP', 'min_mean_AP', 'max_mean_AP', 'sd_mean_AP', 'delta_3h_mean_AP',
    'avg_SpO2', 'min_SpO2', 'max_SpO2', 'sd_SpO2', 'delta_3h_SpO2',
    'diabetis_2_type', 'chronic_kidney_disease', 'COPD',
    'coronary_artery_disease', 'arterial_hypertension', 'heart_failure',
    'ischemic_stroke', 'hemorrhagic_stroke', 'traumatic_brain_injury'
]

def load_and_prepare_data(path):
    df = pd.read_csv(path, sep=';')
    if 'sepsis_3' in df.columns:
        df['sepsis_3'] = df['sepsis_3'].astype(int)
    return df

def select_features(X, y, corr_thresh=0.9):
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    selected = []
    for feat in mi_series.index:
        if all(abs(X[feat].corr(X[s])) < corr_thresh for s in selected):
            selected.append(feat)
    print(f"\n2. Feature selection: {len(selected)} selected")
    return selected

def get_param_grid(model_type, scale_pos_weight):
    if model_type == 'xgb':
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.04, 0.08, 0.14],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'scale_pos_weight': [scale_pos_weight]
        }
    elif model_type == 'ada':
        return {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.04, 0.08, 0.1, 0.16],
            'estimator__max_depth': [2, 3]
        }
    elif model_type == 'rf':
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2]
        }
    elif model_type == 'lgb':
        return {
            'num_leaves': [15, 31],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.04, 0.08],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'scale_pos_weight': [scale_pos_weight, scale_pos_weight*1.5]
        }

def train_model(model_type, X_train, y_train, X_val, y_val, selected_features, max_time=3600):
    grid = get_param_grid(model_type, (y_train == 0).sum() / (y_train == 1).sum())
    param_combinations = list(product(*grid.values()))
    print(f"\n3. Grid search: {len(param_combinations)} combinations")

    best_auc = 0
    best_model = None
    best_params = None
    start = time.time()

    for i, values in enumerate(param_combinations, 1):
        if time.time() - start > max_time:
            print("⏱ Max runtime exceeded")
            break
        params = dict(zip(grid.keys(), values))
        try:
            if model_type == 'xgb':
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1, **params)
            elif model_type == 'ada':
                model = AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=params.pop('estimator__max_depth')),
                    random_state=42, **params)
            elif model_type == 'rf':
                model = RandomForestClassifier(class_weight={0: 1, 1: grid['max_depth'][0]}, random_state=42, n_jobs=-1, **params)
            elif model_type == 'lgb':
                model = lgb.LGBMClassifier(objective='binary', random_state=42, **params)

            model.fit(X_train[selected_features], y_train)
            preds = model.predict_proba(X_val[selected_features])[:, 1]
            auc = roc_auc_score(y_val, preds)

            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_params = params
                print(f"\n New best AUROC: {auc:.4f}")
                print(f"Best parameters: {best_params}")
            if i % 5 == 0:
                print(f"Checked {i}/{len(param_combinations)}")
        except Exception as e:
            print(f"Error at {params}: {e}")
            continue

    return best_model, best_auc, best_params

def apply_model(model, selected_features, input_path, output_path):
    df = load_and_prepare_data(input_path)
    for f in selected_features:
        if f not in df.columns:
            df[f] = -999
    X = df[selected_features].fillna(-999).astype(float)
    df['sepsis_score'] = model.predict_proba(X)[:, 1]
    df['sepsis_predicted'] = (df['sepsis_score'] >= 0.5).astype(int)
    if 'sepsis_3' in df.columns:
        auc = roc_auc_score(df['sepsis_3'], df['sepsis_score'])
        print(f"{os.path.basename(input_path)} AUROC: {auc:.4f}")
    df.to_csv(output_path, sep=';', index=False)
    print(f"Saved: {output_path}")

def main(model_type):
    assert model_type in ['xgb', 'ada', 'rf', 'lgb'], "Model type must be one of: xgb, ada, rf, lgb"

    print("1. Loading data...")
    train_df = load_and_prepare_data(TRAIN_PATH)
    val_df = load_and_prepare_data(VAL_PATH)

    X_train = train_df[FEATURES].fillna(-999).astype(float)
    y_train = train_df['sepsis_3']
    X_val = val_df[FEATURES].fillna(-999).astype(float)
    y_val = val_df['sepsis_3']

    selected = select_features(X_train, y_train)

    print("\n Starting model training...")
    model, val_auc, best_params = train_model(model_type, X_train, y_train, X_val, y_val, selected)
    if model is None:
        print("No model was successfully trained.")
        return

    model_name = f"var1_{model_type.upper()}_0.9"
    model_file = os.path.join(BASE_PATH, f"{model_name}.model")
    joblib.dump(model, model_file)
    print(f"\n Model saved to: {model_file}")
    print(f"\n Validation AUROC: {val_auc:.4f}")

    val_df['sepsis_score'] = model.predict_proba(X_val[selected])[:, 1]
    val_df['sepsis_predicted'] = (val_df['sepsis_score'] >= 0.5).astype(int)
    val_df.to_csv(os.path.join(BASE_PATH, f"{model_name}_val.csv"), sep=';', index=False)

    apply_model(model, selected, TRAIN_PATH, os.path.join(BASE_PATH, f"{model_name}_train.csv"))
    apply_model(model, selected, TEST_PATH, os.path.join(BASE_PATH, f"{model_name}_test.csv"))
    apply_model(model, selected, RICD_PATH, os.path.join(BASE_PATH, f"{model_name}_ricd.csv"))

if __name__ == "__main__":
    main("ada")  # change to 'ada', 'rf', or 'lgb' as needed

