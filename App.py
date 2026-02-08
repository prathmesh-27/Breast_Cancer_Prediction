from __future__ import annotations

import json

import pandas as pd
from flask import Flask, jsonify, render_template, request

from breast_cancer.config import FEATURE_NAMES, METRICS_PATH
from breast_cancer.model import predict_from_dataframe

app = Flask(__name__)


@app.route("/")
def home():
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return render_template("home.html", features=FEATURE_NAMES, metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = {feature: float(request.form[feature])
                   for feature in FEATURE_NAMES}
        result = predict_from_dataframe(pd.DataFrame([payload]))
        return render_template(
            "home.html",
            features=FEATURE_NAMES,
            prediction=result,
            submitted_values=payload,
            metrics=json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if METRICS_PATH.exists()
            else {},
        )
    except Exception as exc:  # broad for user-friendly errors
        return render_template(
            "home.html",
            features=FEATURE_NAMES,
            error=str(exc),
            metrics=json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if METRICS_PATH.exists()
            else {},
        ), 400


@app.route("/predict_api", methods=["POST"])
def predict_api():
    body = request.get_json(silent=True) or {}
    try:
        df = pd.DataFrame([body])
        result = predict_from_dataframe(df)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=8080)
