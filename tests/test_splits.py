import numpy as np
from bigmart.bundle import split_items
from bigmart.evaluate import splitter


def test_cold_item_folds_never_share_an_item_and_cover_every_row_once(df):
    seen = []
    for tr, va in splitter("cold_item", df):
        assert not set(df.Item_Identifier.iloc[tr]) & set(df.Item_Identifier.iloc[va])
        seen += list(va)
    assert sorted(seen) == list(range(len(df)))


def test_cold_outlet_holds_out_exactly_one_unseen_outlet(df):
    for tr, va in splitter("cold_outlet", df):
        assert df.Outlet_Identifier.iloc[va].nunique() == 1
        assert df.Outlet_Identifier.iloc[va].iloc[0] not in set(df.Outlet_Identifier.iloc[tr])


def test_item_group_split_is_disjoint_and_complete(df):
    tr, te = split_items(df, 0.2, seed=1)
    assert not set(df.Item_Identifier.iloc[tr]) & set(df.Item_Identifier.iloc[te])
    assert len(tr) + len(te) == len(df)
    assert abs(df.Item_Identifier.iloc[te].nunique() / df.Item_Identifier.nunique() - 0.2) < 0.03
