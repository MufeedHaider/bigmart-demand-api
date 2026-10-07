import numpy as np


def test_intervals_are_ordered_and_mostly_cover_held_out_items(tiny_model, df):
    from bigmart.bundle import split_items
    _, te = split_items(df, 0.25, seed=9); held = df.iloc[te]
    out = tiny_model.predict(held)
    assert (out.lower <= out.prediction + 1e-9).all() and (out.prediction <= out.upper + 1e-9).all()
    inside = ((held.Item_Outlet_Sales >= out.lower) & (held.Item_Outlet_Sales <= out.upper)).mean()
    assert inside > 0.6          # tiny synthetic model (the model was fitted on these items, so this is only a sanity floor)


def test_predictions_are_positive_and_track_price(tiny_model, df):
    cheap, dear = df.iloc[[0]].copy(), df.iloc[[0]].copy(); cheap["Item_MRP"], dear["Item_MRP"] = 40.0, 250.0
    assert tiny_model.predict(cheap).prediction.iloc[0] < tiny_model.predict(dear).prediction.iloc[0]
    assert (tiny_model.predict(df).prediction > 0).all()


def test_serving_path_does_not_import_training_only_libraries():
    """Regression: the Docker image installs runtime requirements only; importing the model must not need xgboost/optuna/shap."""
    import subprocess, sys
    code = ("import sys, bigmart.bundle, app.main; bad = {'xgboost', 'optuna', 'shap', 'matplotlib'} & set(sys.modules); "
            "assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
