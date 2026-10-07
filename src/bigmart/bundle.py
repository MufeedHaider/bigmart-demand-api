"""The shipped model: point forecast + conformalized quantile (CQR) 80% interval + per-request explanation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import TARGET
from .features import FeatureBuilder
from .models import LGB

ALPHA = 0.20  # 80% prediction interval


def split_items(df: pd.DataFrame, frac: float, seed: int):
    """Group split by item so a product never appears on both sides."""
    rng = np.random.RandomState(seed)
    items = np.array(df["Item_Identifier"].unique(), dtype=object); rng.shuffle(items)
    assert len(set(items)) == len(items)
    held = set(items[: int(round(len(items) * frac))])
    mask = df["Item_Identifier"].isin(held).to_numpy()
    return np.where(~mask)[0], np.where(mask)[0]


@dataclass
class DemandModel:
    point_params: dict
    quantile_params: dict = field(default_factory=dict)
    seed: int = 0
    qhat: float = 0.0
    meta: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "DemandModel":
        y = df[TARGET]
        self.fb = FeatureBuilder(use_outlet_id=False).fit(df)   # lookups only; no target is used
        X = self.fb.transform(df)
        self.point = LGB(objective="gamma", **self.point_params).fit(X, y)
        # --- conformalized quantile regression: quantile models on proper-train, margin from calibration ---
        tr, cal = split_items(df, 0.2, self.seed)
        qp = {**self.point_params, **self.quantile_params}
        self.lo = LGB(objective="quantile", alpha=ALPHA / 2, **qp).fit(X.iloc[tr], y.iloc[tr])
        self.hi = LGB(objective="quantile", alpha=1 - ALPHA / 2, **qp).fit(X.iloc[tr], y.iloc[tr])
        lo_c, hi_c = self.lo.predict(X.iloc[cal]), self.hi.predict(X.iloc[cal])
        scores = np.maximum(lo_c - y.iloc[cal].to_numpy(), y.iloc[cal].to_numpy() - hi_c)
        n = len(scores); k = min(n - 1, math.ceil((n + 1) * (1 - ALPHA)) - 1)
        self.qhat = float(np.sort(scores)[k])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.fb.transform(df)
        p = self.point.predict(X)
        lo = np.minimum(self.lo.predict(X) - self.qhat, p)
        hi = np.maximum(self.hi.predict(X) + self.qhat, p)
        return pd.DataFrame({"prediction": p, "lower": np.clip(lo, 0, None), "upper": hi}, index=df.index)

    def explain(self, df: pd.DataFrame, top: int = 5) -> list[list[dict]]:
        """Gamma objective => log link, so contributions are additive in log space = multiplicative on sales.
        factor > 1 raises the estimate versus the average item-outlet row; factor < 1 lowers it."""
        X = self.fb.transform(df)
        contrib = self.point.m_.predict(X, pred_contrib=True)
        names = list(X.columns)
        out = []
        for row, (_, xr) in zip(contrib, X.iterrows()):
            c = pd.Series(row[:-1], index=names).sort_values(key=np.abs, ascending=False).head(top)
            out.append([{"feature": f, "value": _py(xr[f]), "factor": round(float(np.exp(v)), 3)} for f, v in c.items()])
        return out


def _py(v):
    if isinstance(v, (np.floating, float)): return round(float(v), 4)
    if isinstance(v, (np.integer, int)): return int(v)
    return str(v)
