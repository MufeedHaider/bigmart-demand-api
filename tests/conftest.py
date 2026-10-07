import numpy as np, pandas as pd, pytest, joblib
from fastapi.testclient import TestClient

TYPES = ["Dairy", "Soft Drinks", "Meat", "Household", "Snack Foods", "Breads"]
OUTLETS = {  # id: (type, tier, size, year)
    "OUT010": ("Grocery Store", "Tier 3", None, 1998), "OUT013": ("Supermarket Type1", "Tier 3", "High", 1987),
    "OUT017": ("Supermarket Type1", "Tier 2", None, 2007), "OUT018": ("Supermarket Type2", "Tier 3", "Medium", 2009),
    "OUT019": ("Grocery Store", "Tier 1", "Small", 1985), "OUT027": ("Supermarket Type3", "Tier 3", "Medium", 1985),
    "OUT035": ("Supermarket Type1", "Tier 2", "Small", 2004), "OUT045": ("Supermarket Type1", "Tier 2", None, 2002),
    "OUT046": ("Supermarket Type1", "Tier 1", "Small", 1997), "OUT049": ("Supermarket Type1", "Tier 3", "Medium", 1999)}
LEVEL = {"Grocery Store": 0.15, "Supermarket Type1": 1.0, "Supermarket Type2": 0.9, "Supermarket Type3": 1.6}


def make_df(n_items=80, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed); rows = []
    for i in range(n_items):
        ident = rng.choice(["FD", "DR", "NC"], p=[.6, .2, .2]) + f"{chr(65 + i % 26)}{i // 26}{i % 10}"
        typ = rng.choice(TYPES); mrp = rng.uniform(30, 260); wt = rng.uniform(4, 20); fat = rng.choice(["Low Fat", "LF", "Regular", "reg"])
        for oid, (ot, tier, size, yr) in OUTLETS.items():
            if rng.rand() < 0.15: continue
            rows.append(dict(Item_Identifier=ident, Item_Weight=np.nan if rng.rand() < .15 else wt, Item_Fat_Content=fat,
                             Item_Visibility=0.0 if rng.rand() < .06 else rng.uniform(.01, .2), Item_Type=typ, Item_MRP=mrp,
                             Outlet_Identifier=oid, Outlet_Establishment_Year=yr, Outlet_Size=size, Outlet_Location_Type=tier, Outlet_Type=ot,
                             Item_Outlet_Sales=mrp * 14 * LEVEL[ot] * rng.lognormal(0, .35)))
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def df(): return make_df().rename(columns={"Item_Fat_Content": "Fat_Content"})

@pytest.fixture(scope="session")
def raw_df(): return make_df()


@pytest.fixture(scope="session")
def tiny_model(df):
    from bigmart.bundle import DemandModel
    m = DemandModel(point_params=dict(n_estimators=60, num_leaves=4, min_child_samples=8, learning_rate=0.08, verbose=-1)).fit(df)
    m.meta = {"version": "test", "mrp_range": [30.0, 260.0], "locked_test_r2": 0.5, "cold_outlet_r2": 0.3, "interval_level": 0.8}
    return m


@pytest.fixture()
def client(tiny_model, tmp_path, monkeypatch):
    path = tmp_path / "m.joblib"; joblib.dump(tiny_model, path)
    monkeypatch.setenv("BIGMART_MODEL", str(path)); monkeypatch.setenv("BIGMART_DB", str(tmp_path / "log.db"))
    from app.main import app
    with TestClient(app) as c:
        yield c


GOOD_ROW = {"item_identifier": "FDA15", "item_type": "Dairy", "item_mrp": 120.5, "item_weight": 9.3, "item_fat_content": "Low Fat",
            "item_visibility": 0.03, "outlet_identifier": "OUT049", "outlet_establishment_year": 1999, "outlet_size": "Medium",
            "outlet_location_type": "Tier 3", "outlet_type": "Supermarket Type1"}
