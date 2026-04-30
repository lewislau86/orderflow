from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Binance"
DATASET_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

REQUIRED_FILES = {
    "price": "data__cg_price_history.parquet",
    "oi": "data__cg_open_interest_history.parquet",
    "cvd": "data__cg_cvd_history.parquet",
}

DEFAULT_OPEN_FEE = 0.0005
DEFAULT_CLOSE_FEE = 0.0005
DEFAULT_BUFFER = 0.0005

LABEL_LONG = 1
LABEL_SHORT = -1
LABEL_NO_TRADE = 0

LABEL_TO_NAME = {
    LABEL_SHORT: "SHORT",
    LABEL_NO_TRADE: "NO_TRADE",
    LABEL_LONG: "LONG",
}

NAME_TO_LABEL = {value: key for key, value in LABEL_TO_NAME.items()}
