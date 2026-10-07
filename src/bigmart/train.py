"""End-to-end pipeline:  python -m bigmart.train --data data/raw/Train.csv
1) lock a test set of unseen items  2) compare models under three regimes  3) tune on dev only
4) evaluate once on the locked test with cluster-bootstrap CIs  5) check interval coverage
6) refit on all data and ship the model + metrics + figures."""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path

import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from . import models as M
from .bundle import DemandModel, split_items
from .data import TARGET, load_raw
from .evaluate import cross_validate, metrics
from .features import FeatureBuilder

optuna.logging.set_verbosity(optuna.logging.WARNING)
FB = lambda: FeatureBuilder(use_outlet_id=False)


def cluster_bootstrap(df, y, pred, reps=500, seed=0):
    rng = np.random.RandomState(seed); items = df["Item_Identifier"].to_numpy()
    groups = {k: np.where(items == k)[0] for k in np.unique(items)}; keys = list(groups)
    r2, rmse = [], []
    for _ in range(reps):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        m = metrics(y.to_numpy()[idx], pred[idx]); r2.append(m["r2"]); rmse.append(m["rmse"])
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"r2_ci95": ci(r2), "rmse_ci95": ci(rmse)}


def main(data: str, out: str = ".", trials: int = 40, quick: bool = False):
    t0 = time.time(); out = Path(out); (out / "reports/figures").mkdir(parents=True, exist_ok=True); (out / "models").mkdir(exist_ok=True)
    df = load_raw(data).rename(columns={"Item_Fat_Content": "Fat_Content"}).reset_index(drop=True); y = df[TARGET]
    res: dict = {"n_rows": len(df), "n_items": int(df.Item_Identifier.nunique()), "n_outlets": int(df.Outlet_Identifier.nunique()),
                 "data_sha256": hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]}

    # 1) locked test set: 20% of items, never touched until step 4
    dev_i, test_i = split_items(df, 0.2, seed=42)
    dev, test = df.iloc[dev_i].reset_index(drop=True), df.iloc[test_i].reset_index(drop=True)
    res["split"] = {"dev_rows": len(dev), "test_rows": len(test), "test_items_unseen_in_dev": True}
    assert not set(dev.Item_Identifier) & set(test.Item_Identifier)
    ydev, ytest = dev[TARGET], test[TARGET]

    # 2) model comparison on dev only, three regimes
    zoo = {"Outlet-type mean (baseline)": lambda: M.GroupMeanBaseline("Outlet_Type"), "Ridge (log target)": M.RidgeLog,
           "Random Forest": M.RF, "XGBoost": M.XGB, "LightGBM": M.LGB, "LightGBM (gamma loss)": lambda: M.LGB(objective="gamma")}
    if quick: zoo.pop("Random Forest")
    rows = []
    for proto, seeds in [("warm", (0, 1)), ("cold_item", (0, 1)), ("cold_outlet", (0,))]:
        for name, mk in zoo.items():
            r = cross_validate(mk, FB, dev, ydev, proto, seeds=seeds)
            rows.append({"regime": proto, "model": name, **{k: round(v, 4) for k, v in r.items()}}); print(rows[-1], flush=True)
    comp = pd.DataFrame(rows); comp.to_csv(out / "reports/model_comparison.csv", index=False)

    # 3) tuning on dev only (cold-item CV: tuned for generalising to new products)
    def objective(trial):
        p = dict(num_leaves=trial.suggest_int("num_leaves", 4, 31), min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True), n_estimators=trial.suggest_int("n_estimators", 200, 1200),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0), colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30, log=True), reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True))
        return cross_validate(lambda: M.LGB(objective="gamma", **p), FB, dev, ydev, "cold_item", seeds=(0,))["rmse"]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=6 if quick else trials)
    best = study.best_params
    d = cross_validate(lambda: M.LGB(objective="gamma"), FB, dev, ydev, "cold_item", seeds=(1, 2))      # fresh seeds: no tuning bias
    t = cross_validate(lambda: M.LGB(objective="gamma", **best), FB, dev, ydev, "cold_item", seeds=(1, 2))
    res["tuning"] = {"trials": len(study.trials), "best_params": best, "default_cv": d, "tuned_cv": t,
                     "r2_gain": round(t["r2"] - d["r2"], 4)}
    print("tuning:", res["tuning"]["r2_gain"], flush=True)

    # 4) single evaluation on the locked test (models fitted on dev only, exactly as in deployment)
    dm = DemandModel(point_params=best, seed=0).fit(dev)
    pr = dm.predict(test); pred = pr["prediction"].to_numpy()
    base = M.GroupMeanBaseline("Outlet_Type").fit(FB().fit(dev).transform(dev), ydev).predict(FB().fit(dev).transform(test))
    res["test"] = {"model": metrics(ytest, pred), "baseline_outlet_type_mean": metrics(ytest, base),
                   **cluster_bootstrap(test, ytest, pred)}
    res["test"]["rmse_reduction_vs_baseline_pct"] = round(100 * (1 - res["test"]["model"]["rmse"] / res["test"]["baseline_outlet_type_mean"]["rmse"]), 1)
    seg = []
    for col in ["Outlet_Type", "Item_Type"]:
        for k, g in test.groupby(col):
            m = metrics(g[TARGET], pred[g.index]); seg.append({"by": col, "segment": k, "n": len(g), **{a: round(b, 3) for a, b in m.items()}})
    pd.DataFrame(seg).to_csv(out / "reports/error_by_segment.csv", index=False)

    # 5) prediction-interval coverage on the locked test
    cov = ((ytest >= pr.lower) & (ytest <= pr.upper)).to_numpy()
    by_type = {k: round(float(cov[g.index].mean()), 3) for k, g in test.groupby("Outlet_Type")}
    deciles = pd.qcut(pd.Series(pred), 10, labels=False)
    cov_dec = [float(cov[deciles == i].mean()) for i in range(10)]
    res["intervals"] = {"nominal": 0.8, "qhat": dm.qhat, "coverage_test": round(float(cov.mean()), 4), "coverage_by_outlet_type": by_type,
                        "mean_width": round(float((pr.upper - pr.lower).mean()), 1), "median_width": round(float((pr.upper - pr.lower).median()), 1)}
    # new-outlet warning level (stress test, from comparison table)
    res["cold_outlet_r2_lgb_gamma"] = float(comp[(comp.regime == "cold_outlet") & (comp.model == "LightGBM (gamma loss)")].r2.iloc[0])

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(pred, ytest, s=6, alpha=.35, color="#1F3A6E"); lim = max(pred.max(), ytest.max()); ax[0].plot([0, lim], [0, lim], "k--", lw=1)
    ax[0].set_aspect("equal", adjustable="box"); ax[0].set(xlabel="Predicted sales", ylabel="Actual sales (unseen items)", title=f"Locked test: R² {res['test']['model']['r2']:.3f}")
    ax[1].bar(range(10), cov_dec, color="#1F3A6E"); ax[1].axhline(.8, color="crimson", ls="--", lw=1, label="80% target")
    ax[1].set(xlabel="Predicted-sales decile (low → high)", ylabel="Interval coverage", ylim=(0, 1), title="80% interval coverage by decile"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out / "reports/figures/test_fit_and_coverage.png", dpi=140); plt.close()
    pv = comp.pivot(index="model", columns="regime", values="r2")[["warm", "cold_item", "cold_outlet"]].sort_values("cold_item")
    pv.columns = ["Known items (random split)", "New items (grouped by item)", "New outlets (leave-one-outlet-out)"]
    ax = pv.plot.barh(figsize=(8.6, 5.2), color=["#9db3d6", "#1F3A6E", "#c0504d"], width=0.8)
    ax.set_xlabel("R\u00b2 (dev set, cross-validated)"); ax.set_ylabel(""); ax.set_title("Same models, three ways of asking 'does it generalise?'")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=1, frameon=False)
    plt.tight_layout(); plt.savefig(out / "reports/figures/model_comparison.png", dpi=140); plt.close()

    # 6) ship: refit on ALL rows, SHAP summary, metadata
    final = DemandModel(point_params=best, seed=0).fit(df)
    import shap
    Xs = final.fb.transform(df); sh = shap.TreeExplainer(final.point.m_).shap_values(Xs)
    imp = pd.Series(np.abs(sh).mean(0), index=Xs.columns).sort_values()
    res["shap_mean_abs_log_effect"] = {k: round(float(v), 4) for k, v in imp[::-1].items()}
    imp.plot.barh(figsize=(7, 4.5), color="#1F3A6E"); plt.xlabel("mean |SHAP| (effect on log expected sales)"); plt.title("What drives the estimate"); plt.tight_layout()
    plt.savefig(out / "reports/figures/shap_importance.png", dpi=140); plt.close()
    final.meta = {"version": time.strftime("%Y.%m.%d"), "trained_rows": len(df), "data_sha256": res["data_sha256"],
                  "locked_test_r2": res["test"]["model"]["r2"], "cold_outlet_r2": res["cold_outlet_r2_lgb_gamma"],
                  "mrp_range": [float(df.Item_MRP.min()), float(df.Item_MRP.max())], "interval_level": 0.8,
                  "interval_coverage_test": res["intervals"]["coverage_test"]}
    joblib.dump(final, out / "models/demand_model.joblib")
    res["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out / "reports/metrics.json", "w"), indent=2, default=float)
    print(json.dumps({k: res[k] for k in ["test", "intervals", "cold_outlet_r2_lgb_gamma"]}, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/raw/Train.csv"); ap.add_argument("--out", default=".")
    ap.add_argument("--trials", type=int, default=40); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); main(a.data, a.out, a.trials, a.quick)
