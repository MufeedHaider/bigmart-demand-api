"""Cross-validation under three generalization regimes.

warm        random 5-fold              item AND outlet both seen in training (optimistic)
cold_item   GroupKFold by item         every validation item is new to the model (realistic for new products)
cold_outlet leave-one-outlet-out       every validation outlet is new (stress test: only 10 outlets exist)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut


def splitter(protocol: str, df: pd.DataFrame, seed: int = 0):
    if protocol == "warm":
        return KFold(5, shuffle=True, random_state=seed).split(df)
    if protocol == "cold_item":
        # shuffle items so folds are not ordered by identifier
        rng = np.random.RandomState(seed)
        items = np.array(df["Item_Identifier"].unique(), dtype=object); rng.shuffle(items)
        assert len(set(items)) == len(items) == df["Item_Identifier"].nunique()  # no item lost or duplicated
        fold_of = {it: i % 5 for i, it in enumerate(items)}
        f = df["Item_Identifier"].map(fold_of).to_numpy()
        assert not np.isnan(f.astype(float)).any()                               # every row lands in a fold
        return ((np.where(f != k)[0], np.where(f == k)[0]) for k in range(5))
    if protocol == "cold_outlet":
        return LeaveOneGroupOut().split(df, groups=df["Outlet_Identifier"])
    raise ValueError(protocol)


def metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate(make_model, make_features, df: pd.DataFrame, y: pd.Series, protocol: str, seeds=(0,)) -> dict:
    """Out-of-fold predictions; features are re-fitted inside every fold (no lookup leakage)."""
    per_seed = []
    for seed in seeds:
        oof = np.zeros(len(df))
        for tr, va in splitter(protocol, df, seed):
            fb = make_features()
            Xtr = fb.fit_transform(df.iloc[tr])           # target-derived features are out-of-fold here
            model = make_model()
            model.fit(Xtr, y.iloc[tr])
            oof[va] = model.predict(fb.transform(df.iloc[va]))
        per_seed.append(metrics(y, oof))
    keys = per_seed[0].keys()
    out = {k: float(np.mean([m[k] for m in per_seed])) for k in keys}
    out["r2_sd"] = float(np.std([m["r2"] for m in per_seed])) if len(seeds) > 1 else 0.0
    return out
