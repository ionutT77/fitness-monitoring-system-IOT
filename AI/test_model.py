import json
import pandas as pd
import xgboost as xgb
import shap

# 1. Load the model configuration
try:
    with open('model_config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: model_config.json not found. Did you train the model yet?")
    exit(1)

FEATURE_COLS = config['feature_columns']
LABEL_NAMES = {int(k): v for k, v in config['label_names'].items()}
FEATURE_DESCRIPTIONS = config['feature_descriptions']

# 2. Load the trained XGBoost model
model = xgb.XGBClassifier(enable_categorical=True)
model.load_model('workout_model.json')

# 3. Initialize the SHAP explainer (using our robust monkeypatch for XGBoost 2.1+)
import shap.explainers._tree
import ast
original_decode = shap.explainers._tree.decode_ubjson_buffer

def patched_decode(*args, **kwargs):
    jmodel = original_decode(*args, **kwargs)
    base_score = jmodel.get("learner", {}).get("learner_model_param", {}).get("base_score", "")
    if isinstance(base_score, str) and base_score.startswith('['):
        scores = ast.literal_eval(base_score.replace('E', 'e'))
        jmodel["learner"]["learner_model_param"]["base_score"] = str(scores[0])
    return jmodel
    
shap.explainers._tree.decode_ubjson_buffer = patched_decode
explainer = shap.TreeExplainer(model)


def _get_shap_for_class(shap_vals, pred_label):
    """Extract per-class SHAP values safely."""
    if isinstance(shap_vals, list):
        return shap_vals[pred_label][0]
    arr = shap_vals.values if hasattr(shap_vals, 'values') else shap_vals
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            return arr[0, :, pred_label]
        return arr[0, pred_label, :]
    return arr[0]


def explain_prediction(model_obj, explainer_obj, sample, top_k=3):
    """Generates the prediction and a natural language explanation."""
    sample_df = pd.DataFrame([sample], columns=FEATURE_COLS)

    categorical_cols = ['fitness_level', 'workout_type', 'athlete_type', 'limb_length']
    for col in categorical_cols:
        if col in sample_df.columns:
            sample_df[col] = sample_df[col].astype('category')

    pred_label = int(model_obj.predict(sample_df)[0])
    pred_proba = model_obj.predict_proba(sample_df)[0]
    confidence = float(pred_proba[pred_label])

    shap_vals = explainer_obj.shap_values(sample_df)
    shap_for_class = _get_shap_for_class(shap_vals, pred_label)

    feat_shap = list(zip(FEATURE_COLS, shap_for_class, sample_df.iloc[0]))
    feat_shap.sort(key=lambda x: abs(x[1]), reverse=True)

    def get_feature_desc(feat_name, feat_val):
        thresholds = {
            'duration_mins': (40, 'long session duration', 'short session duration'),
            'avg_hr': (125, 'elevated average heart rate', 'low average heart rate'),
            'max_hr': (145, 'high peak heart rate', 'low peak heart rate'),
            'hr_spikes': (4, 'frequent HR spikes', 'few HR spikes'),
            'pct_time_low': (25, 'significant time in low HR zone', 'minimal time in low HR zone'),
            'avg_emg': (400, 'strong muscle engagement', 'weak muscle engagement'),
            'emg_fatigue': (18, 'significant muscle fatigue', 'minimal muscle fatigue'),
            'total_reps': (90, 'high rep count', 'low rep count')
        }
        if feat_name in thresholds:
            thresh, high_text, low_text = thresholds[feat_name]
            text = high_text if feat_val > thresh else low_text
            if feat_name in ['pct_time_low', 'emg_fatigue']:
                return f"{text} ({feat_val:.1f}%)"
            elif feat_name == 'avg_emg':
                return f"{text} (EMG: {feat_val:.0f})"
            elif feat_name == 'duration_mins':
                return f"{text} ({feat_val:.0f} min)"
            elif feat_name in ['avg_hr', 'max_hr']:
                return f"{text} ({feat_val:.0f} BPM)"
            else:
                return f"{text} ({feat_val:.0f})"
        
        # Categorical handling
        if feat_name in ['fitness_level', 'workout_type', 'athlete_type', 'limb_length']:
            return f"{feat_name.replace('_', ' ')} ({feat_val})"
            
        try:
            return f"{feat_name} ({float(feat_val):.1f})"
        except ValueError:
            return f"{feat_name} ({feat_val})"

    top_factors = []
    for feat_name, shap_val, feat_val in feat_shap[:top_k]:
        top_factors.append(get_feature_desc(feat_name, feat_val))

    explanation = (
        f"Result: {LABEL_NAMES[pred_label]} effectiveness "
        f"(confidence: {confidence:.0%})\n"
        f"Why? Key factors: {'; '.join(top_factors)}."
    )
    
    return explanation


if __name__ == "__main__":
    print("\n============================================")
    print("   WORKOUT EFFECTIVENESS TESTER")
    print("============================================\n")
    print("Enter your workout stats below (or press Enter to use default test values).\n")

    def get_input(prompt, default_val):
        user_input = input(f"{prompt} [Default: {default_val}]: ").strip()
        return float(user_input) if user_input else default_val

    def get_str_input(prompt, default_val):
        user_input = input(f"{prompt} [Default: {default_val}]: ").strip()
        return user_input if user_input else default_val

    # Prompt the user for their custom data
    my_workout = {
        'age':           get_input("Age (years)", 30.0),
        'fitness_level': get_str_input("Fitness Level (low/medium/high)", "medium"),
        'workout_type':  get_str_input("Workout Type (HILV/LIHV/hypertrophy/endurance_lifting)", "hypertrophy"),
        'athlete_type':  get_str_input("Athlete Type (powerlifter/hybrid/gym_bro/non_athletic)", "gym_bro"),
        'body_fat_pct':  get_input("Body Fat %", 15.0),
        'limb_length':   get_str_input("Limb Length (short/medium/long)", "medium"),
        'duration_mins': get_input("Duration (minutes)", 45.0),
        'avg_hr':        get_input("Average Heart Rate (BPM)", 135.0),
        'max_hr':        get_input("Max Heart Rate (BPM)", 165.0),
        'hr_spikes':     get_input("HR Spikes (count)", 4.0),
        'pct_time_low':  get_input("% Time in Low HR Zone", 12.0),
        'avg_emg':       get_input("Average EMG (0-1000)", 480.0),
        'emg_fatigue':   get_input("EMG Fatigue %", 20.0),
        'total_reps':    get_input("Total Reps", 120.0)
    }

    print("\nAnalyzing workout...\n")
    
    explanation = explain_prediction(model, explainer, my_workout)
    
    print("-" * 60)
    print(explanation)
    print("-" * 60)
    print("\nDone!")
