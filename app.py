"""
app.py
Production-ready inference and training API for Heart Disease Prediction.

Usage:
  - To train and save a model: POST /train
  - To get prediction: POST /predict with JSON payload containing features

Note: This script expects `heart.csv` to be present in the repository root.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
MODEL_PATH = os.path.join("models", "model.joblib")
DATA_PATH = "heart.csv"
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def ensure_models_dir():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df


def preprocess(df):
    # Keep only expected features and target if present
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in dataset: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    # Basic type conversions and NA handling
    X = X.astype(float)
    X = X.fillna(X.median())

    y = None
    if "target" in df.columns:
        y = df["target"].astype(int)
    return X, y


def train_and_save(path=MODEL_PATH):
    logging.info("Loading data for training...")
    df = load_data()
    X, y = preprocess(df)
    if y is None:
        raise ValueError("Training requires 'target' column in the dataset.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds).tolist()

    ensure_models_dir()
    joblib.dump(clf, path)

    logging.info(f"Model trained and saved to {path} (accuracy={acc:.4f})")
    return {"accuracy": acc, "confusion_matrix": cm}


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/train", methods=["POST"])
def train():
    try:
        result = train_and_save()
        return jsonify({"status": "trained", "result": result}), 200
    except Exception as e:
        logging.exception("Training failed")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model = load_model()
        if model is None:
            return jsonify({"status": "error", "message": "Model not found. Train it first via POST /train."}), 400

        payload = request.get_json()
        if payload is None:
            return jsonify({"status": "error", "message": "Invalid JSON payload."}), 400

        # Accept single record or list of records
        records = payload if isinstance(payload, list) else [payload]

        X_list = []
        for rec in records:
            if not all(k in rec for k in FEATURE_COLUMNS):
                return jsonify({"status": "error", "message": f"Each record must contain the features: {FEATURE_COLUMNS}"}), 400
            ordered = [float(rec[c]) for c in FEATURE_COLUMNS]
            X_list.append(ordered)

        X_arr = np.array(X_list)
        preds = model.predict(X_arr).tolist()
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_arr).tolist()

        response = {"predictions": preds}
        if probs is not None:
            response["probabilities"] = probs

        return jsonify(response), 200

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # Allow running locally: python app.py
    app.run(host="0.0.0.0", port=5000)
