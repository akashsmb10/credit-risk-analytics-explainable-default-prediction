from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "default_credit_card_clients.csv"
EXPECTED_COLUMNS = ["ID"] + [f"X{i}" for i in range(1, 24)] + ["Y"]


def load_data():
    """Load the untouched UCI CSV and verify its documented table shape."""
    if not RAW_PATH.is_file():
        raise FileNotFoundError(
            "Missing raw dataset: data/raw/default_credit_card_clients.csv. "
            "Download UCI Default of Credit Card Clients (dataset 350) and place its "
            "CSV at that exact path before running analysis scripts."
        )
    data = pd.read_csv(RAW_PATH)
    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError("CSV columns do not match the expected UCI layout.")
    if data.shape != (30000, 25):
        raise ValueError(f"Unexpected dataset shape: {data.shape}")
    return data
