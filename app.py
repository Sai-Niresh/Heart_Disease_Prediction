"""
Production-ready Flask API for heart disease prediction.

Endpoints:
- GET /health
- POST /train
- POST /predict

Requirements:
- `heart.csv` must exist in the project root.
- `heart.csv` must contain a `target` column.
"""

import os
import logging
from typing import List

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

DATA_PATH = os.path.join("data", "heart.csv")
MODEL_PATH = os.path.join("models", "model.joblib")

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


def ensure_model_dir() -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def prepare_features_and_target(df: pd.DataFrame):
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")

    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column for training.")

    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].astype(int)

    X = X.fillna(X.median(numeric_only=True))
    return X, y


def train_model() -> dict:
    df = load_dataset()
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions).tolist()

    ensure_model_dir()
    joblib.dump(model, MODEL_PATH)

    logging.info("Model trained successfully at %s with accuracy %.4f", MODEL_PATH, accuracy)
    return {
        "status": "trained",
        "accuracy": round(float(accuracy), 4),
        "confusion_matrix": cm,
        "model_path": MODEL_PATH,
    }


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please train it first by calling POST /train.")
    return joblib.load(MODEL_PATH)


def validate_prediction_record(record: dict) -> dict:
    missing = [col for col in FEATURE_COLUMNS if col not in record]
    if missing:
        raise ValueError(f"Each record must include all required features. Missing: {missing}")

    normalized = {}
    for col in FEATURE_COLUMNS:
        try:
            normalized[col] = float(record[col])
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{col}' must be numeric.")
    return normalized


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/train", methods=["POST"])
def train():
    try:
        result = train_model()
        return jsonify(result), 200
    except Exception as exc:
        logging.exception("Training failed")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"status": "error", "message": "Request body must be valid JSON."}), 400

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = [payload]
        else:
            return jsonify({"status": "error", "message": "Payload must be a JSON object or an array of objects."}), 400

        model = load_model()

        predictions = []
        probabilities = []

        for record in records:
            try:
                normalized = validate_prediction_record(record)
            except ValueError as exc:
                return jsonify({"status": "error", "message": str(exc)}), 400

            row = np.array([[normalized[col] for col in FEATURE_COLUMNS]], dtype=float)
            pred = int(model.predict(row)[0])
            predictions.append(pred)

            if hasattr(model, "predict_proba"):
                probabilities.append(model.predict_proba(row)[0].tolist())

        response = {"predictions": predictions}
        if probabilities:
            response["probabilities"] = probabilities

        return jsonify(response), 200

    except FileNotFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        logging.exception("Prediction failed")
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
