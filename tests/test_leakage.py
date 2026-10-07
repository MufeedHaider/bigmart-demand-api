"""The failure mode that sank my earlier AQI project was target leakage, so it gets its own tests."""
from bigmart.features import FeatureBuilder


def test_out_of_fold_item_index_never_uses_the_rows_own_label(df):
    a = FeatureBuilder(use_outlet_id=False, use_item_index=True, seed=3)
    Xa = a.fit_transform(df)
    changed = df.copy(); i = 10
    changed.loc[changed.index[i], "Item_Outlet_Sales"] *= 50           # wreck one row's label
    b = FeatureBuilder(use_outlet_id=False, use_item_index=True, seed=3)
    Xb = b.fit_transform(changed)
    assert Xa["Item_Demand_Index"].iloc[i] == Xb["Item_Demand_Index"].iloc[i]


def test_in_sample_lookup_would_have_leaked(df):
    fb = FeatureBuilder(use_outlet_id=False, use_item_index=True)
    oof = fb.fit_transform(df)["Item_Demand_Index"]
    in_sample = fb.transform(df)["Item_Demand_Index"]                  # what a naive pipeline would feed the model
    assert (oof - in_sample).abs().mean() > 1e-3                       # the two differ, so the OOF protection is doing work
