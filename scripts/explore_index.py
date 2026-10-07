import sys; sys.path.insert(0, "src")
import warnings; warnings.filterwarnings("ignore")
from bigmart.data import load_raw, TARGET
from bigmart.features import FeatureBuilder
from bigmart.evaluate import cross_validate
from bigmart import models as M
df = load_raw("data/raw/Train.csv").rename(columns={"Item_Fat_Content": "Fat_Content"}); y = df[TARGET]
mk = lambda: M.LGB(objective="gamma")
for proto in ["warm", "cold_item", "cold_outlet"]:
    for label, fbf in [("no item index", lambda: FeatureBuilder(True, False)), ("with item index", lambda: FeatureBuilder(True, True))]:
        if proto == "cold_outlet" and label == "with item index": pass
        r = cross_validate(mk, fbf, df, y, proto, seeds=(0, 1, 2))
        print(f"{proto:12s} {label:16s} r2={r['r2']:.4f} (sd {r['r2_sd']:.4f}) rmse={r['rmse']:.1f} mae={r['mae']:.1f}", flush=True)
