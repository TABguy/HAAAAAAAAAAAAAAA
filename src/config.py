"""Configuration constants for the corrosion model."""

from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Data files
ENV_TRAIN = INPUT_DIR / "environment_training.csv"
COR_TRAIN = INPUT_DIR / "corrosions_training.csv"
ENV_TEST = INPUT_DIR / "environment_test.csv"
SAMPLE_SUB = INPUT_DIR / "sample_submission.csv"

# Model parameters
RANDOM_STATE = 42
BLEND_GBDT = 0.6  # Ensemble weight on gradient-boosted model
N_SPLITS = 5      # GroupKFold splits

# Structural post-processing
STRUCT_HI = 0.90      # Prior P(corrosion) for later (observation) month
STRUCT_LO = 0.10      # Prior P(corrosion) for earlier (-24m) month
STRUCT_WEIGHT = 0.80  # Blend toward prior (0=model only, 1=prior only)

# Feature engineering
KEYS = ["aircraft_id", "year_month", "month_start_date"]
SEA_SALT_COLS = [
    "sea_salt_aerosol_003_05_mixing_ratio",
    "sea_salt_aerosol_05_5_mixing_ratio",
    "sea_salt_aerosol_5_20_mixing_ratio",
]

# HistGradientBoosting parameters
GBDT_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 400,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_STATE,
}

# Logistic Regression parameters
LOGISTIC_PARAMS = {
    "max_iter": 2000,
    "C": 1.0,
    "random_state": RANDOM_STATE,
}

# Made with Bob
