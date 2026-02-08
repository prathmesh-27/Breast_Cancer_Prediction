'''

file containing the configuration details for model training
'''
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model" / "cancer_predictor.pkl"
METRICS_PATH = BASE_DIR / "models" / "model" / "metrics.json"

FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

TARGET_NAMES = {0: "malignant", 1: "benign"}

# If the cells in the tumor are normal, it's benign.
# If they're abnormal and grow uncontrollably, they're cancerous cells and the tumor is malignant.
