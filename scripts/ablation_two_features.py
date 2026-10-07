"""How much of the model is just price + outlet type?  Re-runs cold-item CV with feature subsets."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
from bigmart.bundle import split_items
from bigmart.data import TARGET, load_raw
from bigmart.evaluate import cross_validate
from bigmart.features import FeatureBuilder
from bigmart import models as M

class Subset(FeatureBuilder):
    def __init__(self, cols): super().__init__(use_outlet_id=False); self.cols = cols
    def transform(self, df, _oof=None): return super().transform(df, _oof)[self.cols]

df = load_raw(sys.argv[1] if len(sys.argv) > 1 else "data/raw/Train.csv").rename(columns={"Item_Fat_Content": "Fat_Content"})
dev = df.iloc[split_items(df, 0.2, seed=42)[0]].reset_index(drop=True); y = dev[TARGET]   # same dev set as train.py
sets = {"price + outlet type (2 features)": ["Item_MRP", "Outlet_Type"],
        "+ outlet age and size (4)": ["Item_MRP", "Outlet_Type", "Outlet_Age", "Outlet_Size"],
        "all 11 features": None}
for name, cols in sets.items():
    mk = (lambda c=cols: Subset(c)) if cols else (lambda: FeatureBuilder(use_outlet_id=False))
    r = cross_validate(lambda: M.LGB(objective="gamma"), mk, dev, y, "cold_item", seeds=(0, 1))
    print(f"{name:36s} R2={r['r2']:.4f}  RMSE={r['rmse']:.1f}")
