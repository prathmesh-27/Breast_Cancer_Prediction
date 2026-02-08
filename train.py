from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from breast_cancer.config import FEATURE_NAMES, METRICS_PATH, MODEL_PATH


def build_candidate_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1200, random_state=42),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_split=4,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()

    X = df[FEATURE_NAMES]
    y = raw.target

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = build_candidate_models()

    cv_scores = {
        name: cross_val_score(model, x_train, y_train,
                              cv=cv, scoring="roc_auc").mean()
        for name, model in candidates.items()
    }

    selected_name = max(cv_scores, key=cv_scores.get)
    selected_model = candidates[selected_name]
    selected_model.fit(x_train, y_train)

    y_pred = selected_model.predict(x_test)
    y_proba = selected_model.predict_proba(x_test)[:, 1]

    metrics = {
        "selected_model": selected_name,
        "cv_roc_auc": {k: round(float(v), 4) for k, v in cv_scores.items()},
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["malignant", "benign"],
            output_dict=True,
        ),
    }

    artifact = {
        "pipeline": selected_model,
        "model_name": selected_name,
        "features": FEATURE_NAMES,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
