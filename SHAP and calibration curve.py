import os
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import warnings

warnings.filterwarnings("ignore")

base_path = r"C:\Your\Path\Here"  # <-- Replace with your actual path
model_path = os.path.join(base_path, "example.model")  # Saved model
data_path = os.path.join(base_path, "example.csv")  # Dataset to analyze
output_prefix = "example"

model = joblib.load(model_path)

df = pd.read_csv(data_path, sep=';')
df['sepsis_3'] = df['sepsis_3'].astype(int)

predictors = model.feature_names_in_.tolist()
df = df[predictors + ['sepsis_3']].fillna(-999)
X = df[predictors].astype(float)
y = df['sepsis_3']

calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrated_model.fit(X, y)

# Predictions and AUROC
df['sepsis_score'] = calibrated_model.predict_proba(X)[:, 1]
df['sepsis_predicted'] = (df['sepsis_score'] >= 0.5).astype(int)
auroc = roc_auc_score(y, df['sepsis_score'])
print(f"\n AUROC: {auroc:.4f}")

# SHAP summary plot
explainer = shap.Explainer(model)
shap_values = explainer(X)

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig(os.path.join(base_path, f"{output_prefix}_shap_summary.png"), bbox_inches='tight', dpi=600)
plt.close()

# Feature importance CSV
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'model_importance': model.feature_importances_,
    'shap_mean_abs': np.abs(shap_values.values).mean(axis=0)
}).sort_values('model_importance', ascending=False)

feature_importance.to_csv(os.path.join(base_path, f"{output_prefix}_feature_importance.csv"), sep=';', index=False)

df[['sepsis_3', 'sepsis_score', 'sepsis_predicted']].to_csv(
    os.path.join(base_path, f"{output_prefix}_predictions.csv"), sep=';', index=False
)

# Calibration curve
prob_true, prob_pred = calibration_curve(y, df['sepsis_score'], n_bins=100, strategy='uniform')
calibration_data = pd.DataFrame({
    'predicted_probability': prob_pred,
    'true_probability': prob_true
})
calibration_data.to_csv(os.path.join(base_path, f"{output_prefix}_calibration_curve.csv"), sep=';', index=False)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Calibrated model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.xlim(0, 0.2)
plt.ylim(0, 0.8)
plt.xlabel("Predicted probability")
plt.ylabel("Observed probability")
plt.title("Calibration curve (0–0.2)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_path, f"{output_prefix}_calibration_curve.png"))
plt.close()

print("✅ SHAP summary, calibration curve, predictions and feature importance saved.")
