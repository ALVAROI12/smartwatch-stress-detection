import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score

# Paths (adjust as needed)
MODEL_PATH = "models/thesis_final/stress_detection_model.pkl"
FEATURES_PATH = "data/processed/wesad_features.json"
SELECTED_FEATURES_PATH = "models/thesis_final/selected_features.txt"
OUTPUT_PATH = "results/advanced_figures/model_predictions.csv"

def load_features_and_labels():
    import json
    # Load all data
    with open(FEATURES_PATH, "r") as f:
        data = json.load(f)
    # Determine all feature keys (exclude metadata)
    exclude_keys = {"label", "subject_id", "condition", "window_start", "window_end", "segment_idx", "window_duration_sec", "purity"}
    all_feature_keys = [k for k in data[0].keys() if k not in exclude_keys]
    # Build feature matrix and label vector
    X = []
    y = []
    meta = []
    for entry in data:
        row = [entry.get(feat, float('nan')) for feat in all_feature_keys]
        X.append(row)
        y.append(entry["label"])
        meta.append({k: entry[k] for k in ("subject_id", "condition", "window_start", "window_end") if k in entry})
    X = np.array(X)
    y = np.array(y)
    return X, y, meta, all_feature_keys

def main():
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Load features and labels
    X, y, meta, selected_features = load_features_and_labels()

    # Impute missing values (NaN) with feature means
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Get predictions and probabilities
    y_pred = model.predict(X_imputed)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_imputed)
    else:
        # For models without predict_proba, use decision_function or fallback
        if hasattr(model, "decision_function"):
            from scipy.special import softmax
            y_proba = softmax(model.decision_function(X_imputed), axis=1)
        else:
            raise ValueError("Model does not support probability prediction.")

    # Try to get class names from model if available
    class_names = None
    if hasattr(model, "classes_"):
        class_names = [str(c) for c in model.classes_]

    # Save to CSV: true label, predicted label, probabilities, and meta info
    df = pd.DataFrame({
        "y_true": y,
        "y_pred": y_pred
    })
    for i in range(y_proba.shape[1]):
        col_name = f"proba_class_{i}"
        if class_names:
            col_name = f"proba_{class_names[i]}"
        df[col_name] = y_proba[:, i]
    # Add meta info
    for k in meta[0].keys():
        df[k] = [m[k] for m in meta]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Model predictions and probabilities saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()