"""Breast cancer prediction project package."""

from .config import FEATURE_NAMES, TARGET_NAMES
from .model import load_model, predict_from_dataframe

__all__ = ["FEATURE_NAMES", "TARGET_NAMES",
           "load_model_bundle", "predict_from_dataframe"]
