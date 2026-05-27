import pandas as pd
import numpy as np
import json
import xgboost as xgb

csv_path = 'synthetic_workouts.csv'
model_path = 'workout_model.json'
config_path = 'model_config.json'

print("======================================================")
print("             DATASET & FEATURE ANALYSIS")
print("======================================================\n")

try:
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.\n")
except FileNotFoundError:
    print(f"Error: Could not find {csv_path}")
    exit(1)

try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    feature_cols = config['feature_columns']
    label_names = {int(k): v for k, v in config['label_names'].items()}
except FileNotFoundError:
    feature_cols = [c for c in df.columns if c not in ['workout_id', 'effectiveness_label']]
    label_names = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Maximum'}

# 1. Global Averages
print("1. GLOBAL AVERAGES (Across all classes)")
print("-" * 50)
summary = df[feature_cols].agg(['mean', 'std', 'min', 'max']).T
summary['mean'] = summary['mean'].round(1)
summary['std'] = summary['std'].round(1)
summary['min'] = summary['min'].round(1)
summary['max'] = summary['max'].round(1)
print(summary.to_string())
print("\n")

# 2. Per-Class Averages
print("2. PER-CLASS AVERAGES")
print("-" * 50)
df['label_name'] = df['effectiveness_label'].map(label_names)
class_means = df.groupby(['effectiveness_label', 'label_name'])[feature_cols].mean().round(1)
print(class_means.to_string())
print("\n")

# 3. Model Feature Importance
print("3. MODEL FEATURE IMPORTANCE (What the model relies on most)")
print("-" * 50)
try:
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # XGBoost default importance is 'gain'
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    
    for _, row in feat_imp.iterrows():
        print(f"{row['Feature']:>15} : {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")
        
    print("\nObservation:")
    top_feature = feat_imp.iloc[0]
    if top_feature['Importance'] > 0.35:
        print(f"  -> WARNING: The model relies heavily on '{top_feature['Feature']}' ({top_feature['Importance']*100:.1f}%).")
        print(f"  -> This means if '{top_feature['Feature']}' is slightly off, predictions may flip drastically.")
except Exception as e:
    print(f"Could not load model for feature importance: {e}")

print("\n======================================================")
print("Recommendations for Refining the Dataset:")
print("1. If a feature's average is too high/low compared to real-life, update the")
print("   (min, max) generation ranges in `generate_synthetic_data.py`.")
print("2. If the model relies too heavily on 'total_reps', you can reduce its variance")
print("   or drop it if you want the model to rely more on Heart Rate and EMG.")
print("3. You can apply feature scaling (StandardScaler) or adjust `scale_pos_weight`")
print("   in XGBoost, though XGBoost natively handles unscaled features well.")
print("======================================================")
