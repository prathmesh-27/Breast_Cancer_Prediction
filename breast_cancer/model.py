from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from breast_cancer.config import FEATURE_NAMES, MODEL_PATH, TARGET_NAMES


@lru_cache(maxsize=1)
def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run `python train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_from_dataframe(input_df: pd.DataFrame) -> dict:
    model_bundle = load_model()
    pipeline = model_bundle["pipeline"]

    missing_features = [
        col for col in FEATURE_NAMES if col not in input_df.columns]
    if missing_features:
        raise ValueError(f"Missing input features: {missing_features}")

    ordered = input_df[FEATURE_NAMES]
    prediction_id = int(pipeline.predict(ordered)[0])
    probability = float(pipeline.predict_proba(ordered)[0][prediction_id])

    return {
        "prediction_id": prediction_id,
        "prediction_label": TARGET_NAMES[prediction_id],
        "confidence": round(probability, 4),
    }
