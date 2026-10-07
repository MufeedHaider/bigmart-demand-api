"""BigMart demand API: predictions with calibrated 80% intervals, per-request explanations, SQLite audit log."""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

ItemType = Literal["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegetables",
                   "Hard Drinks", "Health and Hygiene", "Household", "Meat", "Others", "Seafood", "Snack Foods",
                   "Soft Drinks", "Starchy Foods"]
OutletType = Literal["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
Tier = Literal["Tier 1", "Tier 2", "Tier 3"]
Size = Literal["Small", "Medium", "High"]
FAT_ALIASES = {"low fat", "lf", "regular", "reg", "non-edible"}
MAX_BATCH = 500
COLD_OUTLET_NOTE = "New outlet: accuracy is lower for stores the model has not seen (leave-one-outlet-out R2 about {r2:.2f}) and the interval is not calibrated for it."


class ItemOutletRow(BaseModel):
    """One product in one outlet. Field names follow the BigMart dataset in snake_case."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "item_identifier": "FDA15", "item_type": "Dairy", "item_mrp": 249.81, "item_weight": 9.3,
        "item_fat_content": "Low Fat", "item_visibility": 0.016, "outlet_identifier": "OUT049",
        "outlet_establishment_year": 1999, "outlet_size": "Medium", "outlet_location_type": "Tier 1",
        "outlet_type": "Supermarket Type1"}})
    item_identifier: str = Field(pattern=r"^(FD|DR|NC)[A-Z0-9]{3}$", description="Prefix FD food, DR drink, NC non-consumable")
    item_type: ItemType
    item_mrp: float = Field(gt=0, le=500, description="Maximum retail price")
    item_weight: Optional[float] = Field(default=None, gt=0, le=50, description="Optional; imputed when missing")
    item_fat_content: str = "Regular"
    item_visibility: float = Field(default=0.0, ge=0, le=1, description="Share of display area; 0 means unknown and is imputed")
    outlet_identifier: Optional[str] = Field(default=None, pattern=r"^OUT\d{3}$")
    outlet_establishment_year: int = Field(ge=1950, le=2013)
    outlet_size: Optional[Size] = None
    outlet_location_type: Tier
    outlet_type: OutletType

    @field_validator("item_fat_content")
    @classmethod
    def _fat(cls, v: str) -> str:
        if v.strip().lower() not in FAT_ALIASES:
            raise ValueError(f"item_fat_content must be one of {sorted(FAT_ALIASES)} (case-insensitive)")
        return v


class Prediction(BaseModel):
    prediction: float
    lower: float
    upper: float
    interval_level: float
    outlet_known: bool
    warnings: list[str]


class BatchRequest(BaseModel):
    rows: list[ItemOutletRow] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class Factor(BaseModel):
    feature: str
    value: float | str | int
    factor: float


class Explanation(Prediction):
    model_version: str
    top_factors: list[Factor]
    reading: str = "factor > 1 raises the estimate relative to an average item-outlet row, factor < 1 lowers it (effects multiply)."


COLUMN_MAP = {"item_identifier": "Item_Identifier", "item_type": "Item_Type", "item_mrp": "Item_MRP", "item_weight": "Item_Weight",
              "item_fat_content": "Fat_Content", "item_visibility": "Item_Visibility", "outlet_identifier": "Outlet_Identifier",
              "outlet_establishment_year": "Outlet_Establishment_Year", "outlet_size": "Outlet_Size",
              "outlet_location_type": "Outlet_Location_Type", "outlet_type": "Outlet_Type"}


class AuditLog:
    """Every served prediction is written to SQLite so usage and drift can be analysed with plain SQL."""
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False); self.lock = threading.Lock()
        self.con.execute("""CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY, ts REAL, item_type TEXT, outlet_type TEXT, item_mrp REAL,
            prediction REAL, lower REAL, upper REAL, outlet_known INTEGER, model_version TEXT)""")
        self.con.commit()

    def write(self, rows: list[ItemOutletRow], preds: list[Prediction], version: str):
        with self.lock:
            self.con.executemany(
                "INSERT INTO predictions(ts,item_type,outlet_type,item_mrp,prediction,lower,upper,outlet_known,model_version) VALUES (?,?,?,?,?,?,?,?,?)",
                [(time.time(), r.item_type, r.outlet_type, r.item_mrp, p.prediction, p.lower, p.upper, int(p.outlet_known), version)
                 for r, p in zip(rows, preds)])
            self.con.commit()

    def stats(self) -> dict:
        with self.lock:
            total = self.con.execute("SELECT COUNT(*), AVG(prediction), AVG(upper-lower) FROM predictions").fetchone()
            by = self.con.execute("""SELECT outlet_type, COUNT(*) n, ROUND(AVG(prediction),1) avg_pred
                                     FROM predictions GROUP BY outlet_type ORDER BY n DESC""").fetchall()
        return {"predictions_served": total[0], "mean_prediction": round(total[1] or 0, 1), "mean_interval_width": round(total[2] or 0, 1),
                "by_outlet_type": [{"outlet_type": a, "n": b, "avg_prediction": c} for a, b, c in by]}


def _frame(rows: list[ItemOutletRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in rows]).rename(columns=COLUMN_MAP)
    df["Outlet_Identifier"] = df["Outlet_Identifier"].fillna("NEW")
    return df


def _warnings(app: FastAPI, rows: list[ItemOutletRow], known: list[bool]) -> list[list[str]]:
    lo, hi = app.state.model.meta["mrp_range"]; out = []
    for r, k in zip(rows, known):
        w = []
        if not k: w.append(COLD_OUTLET_NOTE.format(r2=app.state.model.meta["cold_outlet_r2"]))
        if r.item_weight is None: w.append("item_weight missing: imputed from the item or its type.")
        if r.item_visibility == 0: w.append("item_visibility was 0 (impossible for a stocked item): imputed.")
        if not lo <= r.item_mrp <= hi: w.append(f"item_mrp outside the training range [{lo:.0f}, {hi:.0f}]: extrapolation.")
        out.append(w)
    return out


def _predict(app: FastAPI, rows: list[ItemOutletRow]) -> list[Prediction]:
    m = app.state.model; df = _frame(rows)
    known = df["Outlet_Identifier"].isin(m.fb.categories_["Outlet_Identifier"]).tolist()
    res = m.predict(df); warns = _warnings(app, rows, known)
    preds = [Prediction(prediction=round(float(a), 2), lower=round(float(b), 2), upper=round(float(c), 2),
                        interval_level=m.meta["interval_level"], outlet_known=k, warnings=w)
             for a, b, c, k, w in zip(res.prediction, res.lower, res.upper, known, warns)]
    app.state.log.write(rows, preds, m.meta["version"])
    return preds


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("BIGMART_MODEL", "models/demand_model.joblib")
    if not Path(path).exists():
        raise RuntimeError(f"Model file {path} not found. Run `python -m bigmart.train --data data/raw/Train.csv` first.")
    app.state.model = joblib.load(path)
    app.state.log = AuditLog(os.environ.get("BIGMART_DB", "data/predictions.db"))
    yield


app = FastAPI(title="BigMart Demand API", version="1.0.0", lifespan=lifespan,
              description="Estimates yearly sales of a product in an outlet with a calibrated 80% prediction interval.")


@app.get("/health")
def health():
    m = app.state.model
    return {"status": "ok", "model_version": m.meta["version"], "locked_test_r2": round(m.meta["locked_test_r2"], 3)}


@app.get("/model")
def model_card():
    m = app.state.model
    return {"meta": m.meta, "known_outlets": m.fb.categories_["Outlet_Identifier"],
            "limitations": ["Cross-sectional: one year of sales per item-outlet pair, so this estimates demand level, it does not forecast over time.",
                            "Only 10 outlets exist in the data; new-outlet estimates are much less reliable.",
                            "Intervals are 80% on average (marginal), not guaranteed for every outlet type.",
                            "Associations only: do not read the explanation factors as causal effects of changing a price."]}


@app.post("/predict", response_model=Prediction)
def predict(row: ItemOutletRow):
    return _predict(app, [row])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    return BatchResponse(model_version=app.state.model.meta["version"], predictions=_predict(app, req.rows))


@app.post("/explain", response_model=Explanation)
def explain(row: ItemOutletRow):
    p = _predict(app, [row])[0]; m = app.state.model
    top = m.explain(_frame([row]), top=5)[0]
    return Explanation(**p.model_dump(), model_version=m.meta["version"], top_factors=[Factor(**f) for f in top])


@app.get("/stats")
def stats():
    return app.state.log.stats()
