"""Model zoo: every model exposes fit(X, y) / predict(X) on the FeatureBuilder output."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .features import to_onehot


class GroupMeanBaseline:
    """Predict the mean sales of the outlet type (or outlet if it was seen) - the 'dumb but honest' baseline."""
    def __init__(self, by: str = "Outlet_Type"):
        self.by = by
    def fit(self, X, y):
        self.means_ = y.groupby(X[self.by].astype(str).values).mean().to_dict(); self.g_ = float(y.mean()); return self
    def predict(self, X):
        return X[self.by].astype(str).map(self.means_).fillna(self.g_).to_numpy()


class RidgeLog:
    def fit(self, X, y):
        self.cols_ = None
        Z = to_onehot(X); self.cols_ = Z.columns
        self.sc_ = StandardScaler().fit(Z)
        self.m_ = Ridge(alpha=3.0).fit(self.sc_.transform(Z), np.log1p(y)); return self
    def predict(self, X):
        Z = to_onehot(X).reindex(columns=self.cols_, fill_value=0.0)
        return np.expm1(self.m_.predict(self.sc_.transform(Z)))


class RF:
    def __init__(self, **kw): self.kw = dict(n_estimators=400, min_samples_leaf=5, max_features=0.6, n_jobs=-1, random_state=0) | kw
    def fit(self, X, y):
        self.cols_ = None; Z = to_onehot(X); self.cols_ = Z.columns
        self.m_ = RandomForestRegressor(**self.kw).fit(Z, y); return self
    def predict(self, X): return self.m_.predict(to_onehot(X).reindex(columns=self.cols_, fill_value=0.0))


class LGB:
    def __init__(self, **kw):
        self.kw = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=0, verbose=-1) | kw
    def fit(self, X, y): self.m_ = lgb.LGBMRegressor(**self.kw).fit(X, y); return self
    def predict(self, X): return self.m_.predict(X)


class XGB:
    def __init__(self, **kw):
        self.kw = dict(n_estimators=500, learning_rate=0.03, max_depth=4, min_child_weight=5, subsample=0.8,
                       colsample_bytree=0.8, reg_lambda=5.0, enable_categorical=True, tree_method="hist",
                       random_state=0, n_jobs=4) | kw
    def fit(self, X, y):
        import xgboost as xgb   # training-only dependency: imported lazily so the serving image does not need it
        self.m_ = xgb.XGBRegressor(**self.kw).fit(X, y); return self
    def predict(self, X): return self.m_.predict(X)
