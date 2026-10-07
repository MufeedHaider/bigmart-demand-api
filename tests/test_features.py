import numpy as np, pandas as pd
from bigmart.data import clean_fat_content
from bigmart.features import FeatureBuilder


def test_fat_content_five_spellings_become_two_labels():
    ids = pd.Series(["FDA01"] * 5)
    out = clean_fat_content(ids, pd.Series(["Low Fat", "LF", "low fat", "Regular", "reg"]))
    assert set(out) == {"Low Fat", "Regular"}


def test_non_consumables_are_never_low_fat():
    out = clean_fat_content(pd.Series(["NCA01", "FDA01"]), pd.Series(["Low Fat", "Low Fat"]))
    assert list(out) == ["Non-Edible", "Low Fat"]


def test_transform_has_no_missing_values_and_fixes_zero_visibility(df):
    fb = FeatureBuilder(use_outlet_id=False).fit(df); X = fb.transform(df)
    assert X.isna().sum().sum() == 0
    assert (X["Item_Visibility"] > 0).all()
    assert (df["Item_Visibility"] == 0).any()          # the fixture really contains the defect we repair


def test_feature_builder_never_reads_the_target(df):
    fb = FeatureBuilder(use_outlet_id=False).fit(df.drop(columns=["Item_Outlet_Sales"]))
    X = fb.transform(df.drop(columns=["Item_Outlet_Sales"]))
    assert len(X) == len(df)


def test_unseen_item_and_unknown_outlet_do_not_crash(df):
    fb = FeatureBuilder(use_outlet_id=False).fit(df)
    new = df.iloc[:2].copy(); new["Item_Identifier"] = ["FDZ99", "DRZ99"]; new["Outlet_Identifier"] = "NEW"
    new["Item_Weight"] = np.nan; new["Item_Visibility"] = 0.0
    X = fb.transform(new)
    assert X.isna().sum().sum() == 0 and (X["Item_Visibility"] > 0).all()


def test_lookups_are_learned_on_training_rows_only(df):
    train = df.iloc[: len(df) // 2]; fb = FeatureBuilder(use_outlet_id=False).fit(train)
    held = set(df.Item_Identifier) - set(train.Item_Identifier)
    assert held and not held & set(fb.item_weight_)
