"""Leakage-safe feature builder.

Everything learned from data (imputation lookups, outlet visibility means) is learned in `fit`
on training rows only and re-used in `transform`, so the same code serves cross-validation and the API.
No feature is derived from the target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from .data import REFERENCE_YEAR, TARGET, clean_fat_content

CATEGORICAL = [
    "Item_Category", "Item_Type", "Fat_Content", "Outlet_Type", "Outlet_Location_Type", "Outlet_Size",
]
OUTLET_ID = "Outlet_Identifier"
NUMERIC = ["Item_MRP", "Item_Weight", "Item_Visibility", "Visibility_Ratio", "Outlet_Age"]


class FeatureBuilder:
    def __init__(self, use_outlet_id: bool = True, use_item_index: bool = False, shrink: float = 2.0, seed: int = 0):
        self.use_outlet_id = use_outlet_id
        self.use_item_index = use_item_index
        self.shrink = shrink
        self.seed = seed
        self.categories_: dict[str, list[str]] = {}

    # ---- learn lookups from training rows only -------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        self.item_weight_ = df.dropna(subset=["Item_Weight"]).groupby("Item_Identifier").Item_Weight.median().to_dict()
        self.type_weight_ = df.groupby("Item_Type").Item_Weight.median().to_dict()
        self.global_weight_ = float(df.Item_Weight.median())
        pos = df[df.Item_Visibility > 0]
        self.item_vis_ = pos.groupby("Item_Identifier").Item_Visibility.mean().to_dict()
        self.outlet_vis_ = pos.groupby(OUTLET_ID).Item_Visibility.mean().to_dict()
        self.global_vis_ = float(pos.Item_Visibility.mean())
        # Outlet size: take the known size of the outlet; for outlets with none, use the most common
        # size among outlets of the same type, else "Unknown" (kept as its own honest category).
        known = df.dropna(subset=["Outlet_Size"]).groupby(OUTLET_ID).Outlet_Size.first().to_dict()
        self.outlet_size_ = known
        type_mode = (
            df.dropna(subset=["Outlet_Size"]).groupby("Outlet_Type").Outlet_Size.agg(lambda s: s.mode().iat[0]).to_dict()
        )
        self.type_size_ = {t: m for t, m in type_mode.items() if t == "Grocery Store"}  # only unambiguous case
        self.categories_ = {
            "Item_Category": ["FD", "DR", "NC"],
            "Item_Type": sorted(df.Item_Type.unique()),
            "Fat_Content": ["Low Fat", "Regular", "Non-Edible"],
            "Outlet_Type": sorted(df.Outlet_Type.unique()),
            "Outlet_Location_Type": sorted(df.Outlet_Location_Type.unique()),
            "Outlet_Size": ["Small", "Medium", "High", "Unknown"],
            OUTLET_ID: sorted(df[OUTLET_ID].unique()),
        }
        if self.use_item_index:
            self.item_stats_ = self._item_stats(df)
        return self

    # ---- item demand index (target-derived => must be out-of-fold for training rows) -------------
    def _item_stats(self, df: pd.DataFrame) -> dict:
        """Per item: sum and count of (sales / outlet mean sales). Outlet means come from the same rows."""
        om = df.groupby(OUTLET_ID)[TARGET].mean()
        rel = df[TARGET] / df[OUTLET_ID].map(om)
        g = rel.groupby(df["Item_Identifier"]).agg(["sum", "count"])
        return {"outlet_mean": om.to_dict(), "table": g}

    def _index_from(self, stats: dict, idents: pd.Series) -> tuple[pd.Series, pd.Series]:
        g = stats["table"].reindex(idents.values)
        s = g["sum"].fillna(0.0).to_numpy(); n = g["count"].fillna(0.0).to_numpy()
        idx = (s + self.shrink * 1.0) / (n + self.shrink)   # shrink toward 1.0 (an average item)
        return pd.Series(idx, index=idents.index), pd.Series(n, index=idents.index)

    def _oof_index(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Index for the rows we train on: each row is encoded using ONLY rows from other inner folds."""
        idx = pd.Series(np.nan, index=df.index); cnt = pd.Series(np.nan, index=df.index)
        for tr, va in KFold(5, shuffle=True, random_state=self.seed).split(df):
            stats = self._item_stats(df.iloc[tr])
            i, c = self._index_from(stats, df["Item_Identifier"].iloc[va])
            idx.iloc[va] = i.to_numpy(); cnt.iloc[va] = c.to_numpy()
        return idx, cnt

    # ---- apply -----------------------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, _oof: tuple | None = None) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        ident = df["Item_Identifier"].astype(str)
        out["Item_MRP"] = df["Item_MRP"].astype(float)

        w = df["Item_Weight"]
        w = w.fillna(ident.map(self.item_weight_)).fillna(df["Item_Type"].map(self.type_weight_)).fillna(self.global_weight_)
        out["Item_Weight"] = w.astype(float)

        vis = df["Item_Visibility"].astype(float)
        fallback = ident.map(self.item_vis_).fillna(df[OUTLET_ID].map(self.outlet_vis_)).fillna(self.global_vis_)
        vis = vis.where(vis > 0, fallback)  # zero visibility is impossible for a stocked item
        out["Item_Visibility"] = vis
        out["Visibility_Ratio"] = vis / df[OUTLET_ID].map(self.outlet_vis_).fillna(self.global_vis_)

        out["Outlet_Age"] = REFERENCE_YEAR - df["Outlet_Establishment_Year"].astype(int)
        out["Item_Category"] = ident.str[:2]
        out["Item_Type"] = df["Item_Type"]
        out["Fat_Content"] = clean_fat_content(ident, df["Fat_Content"] if "Fat_Content" in df else df["Item_Fat_Content"])
        out["Outlet_Type"] = df["Outlet_Type"]
        out["Outlet_Location_Type"] = df["Outlet_Location_Type"]
        size = df["Outlet_Size"].fillna(df[OUTLET_ID].map(self.outlet_size_)).fillna(df["Outlet_Type"].map(self.type_size_))
        out["Outlet_Size"] = size.fillna("Unknown")
        cols = NUMERIC + CATEGORICAL
        if self.use_item_index:
            if _oof is not None:
                out["Item_Demand_Index"], out["Item_Index_Support"] = _oof
            else:
                out["Item_Demand_Index"], out["Item_Index_Support"] = self._index_from(self.item_stats_, ident)
            cols = cols + ["Item_Demand_Index", "Item_Index_Support"]
        if self.use_outlet_id:
            out[OUTLET_ID] = df[OUTLET_ID]
            cols = cols + [OUTLET_ID]
        out = out[cols]
        for c in CATEGORICAL + ([OUTLET_ID] if self.use_outlet_id else []):
            out[c] = pd.Categorical(out[c], categories=self.categories_[c])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Training-time transform: target-derived features are out-of-fold so no row sees its own label."""
        self.fit(df)
        return self.transform(df, _oof=self._oof_index(df) if self.use_item_index else None)


def to_onehot(X: pd.DataFrame) -> pd.DataFrame:
    """Dense one-hot view for linear models (tree models use native categoricals)."""
    return pd.get_dummies(X, dtype=float)
