"""Loading and cleaning the BigMart dataset (one row per item x outlet, single year of sales)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Item_Outlet_Sales"
FAT_MAP = {"low fat": "Low Fat", "lf": "Low Fat", "regular": "Regular", "reg": "Regular"}
REFERENCE_YEAR = 2013  # the dataset is a 2013 snapshot; outlet age is measured against it


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the BigMart Sales training file (Train.csv) from the "
            "Analytics Vidhya / Kaggle BigMart Sales Prediction dataset and place it there."
        )
    return pd.read_csv(path)


def clean_fat_content(item_identifier: pd.Series, fat: pd.Series) -> pd.Series:
    """Five spellings collapse to two labels; non-consumables (NC prefix) cannot be 'Low Fat'."""
    cleaned = fat.astype(str).str.strip().str.lower().map(FAT_MAP).fillna("Regular")
    return cleaned.mask(item_identifier.str[:2] == "NC", "Non-Edible")
