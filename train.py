"""
Training pipeline: suggested nightly price for a new Airbnb host in Madrid.

Run via:  uv run python main.py train

One command, start to finish, no manual steps:
 1. Clean the raw Inside Airbnb export (clean_data.py).
 2. Target log(price); engineer features (features.py).
 3. Split train / validation / test GROUPED BY HOST.
 4. Fit the preprocessor on the training split only.
 5. Fit ONE model, ridge regression on log(price), three ways (pipeline.py):
      a. scikit-learn Ridge (exact solver)
      b. manual PyTorch: raw tensors, autograd, hand-written update step
      c. standard PyTorch: nn.Module + nn.MSELoss + torch.optim.SGD
    Only the ridge strength (alpha) and the learning rate are tuned.
 6. Evaluate on the test split against two baselines and against the same
    ridge without the quote controls; all quotes and weekday-only quotes.
 7. Evidence and validity checks (diagnostics.py).
 8. Save everything app.py needs to artifacts/. The app never retrains.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import diagnostics as D
from clean_data import run_cleaning
from features import NUMERIC_FEATURES, QUOTE_CONTROLS, engineer_features
from pipeline import (
    ALPHA_GRID, CONVERGED_REL_GAP, EPOCHS, LOG_EVERY, SEED,
    build_preprocessor, choose_learning_rate, fit_baseline, fit_baseline_comps, fit_manual_torch,
    fit_sklearn, fit_standard_torch, gd_stability, grouped_split, metrics, predict_baseline,
    predict_baseline_comps, predict_manual, predict_standard, ridge_objective,
)

ART = Path("artifacts")
IMPLEMENTATIONS = ["sklearn", "torch_manual", "torch_nn"]

np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    t0 = time.time()
    ART.mkdir(exist_ok=True)

    # 1. Clean the raw export (writes data/madrid_listings_clean.csv.gz)
    df, cleaning_log = run_cleaning()

    # 2. Target and features
    df["log_price"] = np.log(df["price_eur"])   # errors matter in %, so model log price
    X_all = engineer_features(df)
    y_all = df["log_price"].to_numpy()

    # 3. Split grouped by host
    train_idx, val_idx, test_idx = grouped_split(df)
    X_tr, X_val, X_te = X_all.iloc[train_idx], X_all.iloc[val_idx], X_all.iloc[test_idx]
    y_tr, y_val, y_te = y_all[train_idx], y_all[val_idx], y_all[test_idx]
    hosts_te = df["host_id"].iloc[test_idx].to_numpy()
    print(f"\nSplit by host: train {len(y_tr):,}  val {len(y_val):,}  test {len(y_te):,} listings")

    # 4. Preprocessor, fitted on train only
    pre = build_preprocessor().fit(X_tr)
    Z_tr, Z_val, Z_te = pre.transform(X_tr), pre.transform(X_val), pre.transform(X_te)
    feature_names = list(pre.get_feature_names_out())
    print(f"{len(feature_names)} model inputs after preprocessing")

    # 5a. Ridge strength: chosen on validation with sklearn (exact and fast)
    alpha_val = {a: metrics(y_val, fit_sklearn(Z_tr, y_tr, a).predict(Z_val))["rmse_log"] for a in ALPHA_GRID}
    alpha = min(alpha_val, key=alpha_val.get)
    print("Validation RMSE(log) by alpha:", {k: round(v, 4) for k, v in alpha_val.items()}, "-> alpha =", alpha)
    sk_model = fit_sklearn(Z_tr, y_tr, alpha)
    sk_obj = ridge_objective(Z_tr, y_tr, sk_model.coef_, sk_model.intercept_, alpha)

    # 5b. Learning rate: fastest rate that converges to the same optimum
    stability = gd_stability(Z_tr)
    lr, lr_table = choose_learning_rate(Z_tr, y_tr, alpha, sk_obj)
    print(f"Largest stable learning rate (2 / lambda_max): {stability['max_stable_lr']:.3f}")
    print("Learning rates:", {k: (v["epochs_to_converge"] if not v["diverged"] else "diverged")
                              for k, v in lr_table.items()}, "-> lr =", lr)

    # 5c. Final fits of the two torch versions
    manual_params, manual_hist = fit_manual_torch(Z_tr, y_tr, alpha, lr)
    nn_model, nn_hist = fit_standard_torch(Z_tr, y_tr, alpha, lr)

    # 6. Comparison rows. Two baselines a host could build by hand, and the same
    #    ridge WITHOUT the quote controls (shows what the date fix is worth).
    baseline = fit_baseline(X_tr, y_tr)
    baseline_comps = fit_baseline_comps(X_tr, y_tr)
    pre_no_dates = build_preprocessor(numeric=[c for c in NUMERIC_FEATURES if c not in QUOTE_CONTROLS]).fit(X_tr)
    no_dates_model = fit_sklearn(pre_no_dates.transform(X_tr), y_tr, alpha)

    # Evaluate on the test split (first and only time it is used)
    preds = {
        "baseline": predict_baseline(baseline, X_te),
        "baseline_comps": predict_baseline_comps(baseline_comps, X_te),
        "ridge_no_dates": no_dates_model.predict(pre_no_dates.transform(X_te)),
        "sklearn": sk_model.predict(Z_te),
        "torch_manual": predict_manual(manual_params, Z_te),
        "torch_nn": predict_standard(nn_model, Z_te),
    }
    results = {name: metrics(y_te, p) for name, p in preds.items()}
    # The app always quotes a weekday stay, so also score weekday-only quotes.
    weekday = X_te["weekend_share"].to_numpy() == 0
    results_weekday = {name: metrics(y_te[weekday], p[weekday]) for name, p in preds.items()}
    comparison_rows = ["baseline", "baseline_comps", "ridge_no_dates", "sklearn"]
    test_ci = D.bootstrap_ci(y_te, {k: preds[k] for k in comparison_rows}, hosts_te)
    test_ci_weekday = D.bootstrap_ci(y_te[weekday], {k: preds[k][weekday] for k in comparison_rows}, hosts_te[weekday])

    # Do the three implementations agree? Coefficients and per-listing euros.
    coefs = pd.DataFrame({
        "feature": feature_names,
        "sklearn": sk_model.coef_,
        "torch_manual": manual_params["w"].numpy(),
        "torch_nn": nn_model.linear.weight.detach().numpy().ravel(),
    })
    intercepts = {
        "sklearn": float(sk_model.intercept_),
        "torch_manual": float(manual_params["b"].item()),
        "torch_nn": float(nn_model.linear.bias.detach().item()),
    }
    gap = (coefs.torch_manual - coefs.sklearn).abs()
    eur = {k: np.exp(preds[k]) for k in IMPLEMENTATIONS}
    agreement = {
        "max_abs_coef_diff_manual_vs_sklearn": float(gap.max()),
        "max_abs_coef_diff_nn_vs_sklearn": float((coefs.torch_nn - coefs.sklearn).abs().max()),
        "max_abs_coef_diff_manual_vs_nn": float((coefs.torch_manual - coefs.torch_nn).abs().max()),
        "feature_with_largest_gap": coefs.loc[gap.idxmax(), "feature"],
        "max_abs_pred_diff_log_manual_vs_sklearn": float(np.abs(preds["torch_manual"] - preds["sklearn"]).max()),
        "max_abs_pred_diff_log_nn_vs_sklearn": float(np.abs(preds["torch_nn"] - preds["sklearn"]).max()),
        "max_abs_pred_diff_eur_manual_vs_sklearn": float(np.abs(eur["torch_manual"] - eur["sklearn"]).max()),
        "max_abs_pred_diff_eur_nn_vs_sklearn": float(np.abs(eur["torch_nn"] - eur["sklearn"]).max()),
        "median_abs_pred_diff_eur_manual_vs_sklearn": float(np.median(np.abs(eur["torch_manual"] - eur["sklearn"]))),
        "median_abs_pred_diff_eur_nn_vs_sklearn": float(np.median(np.abs(eur["torch_nn"] - eur["sklearn"]))),
        "train_objective_sklearn": sk_obj,
        "train_objective_manual_final": manual_hist[-1],
        "train_objective_nn_final": nn_hist[-1],
    }

    # Where does the model miss? Median signed error by actual price band (test).
    actual = np.exp(y_te)
    bands = pd.qcut(actual, 5)
    signed = (np.exp(preds["sklearn"]) - actual) / actual
    error_by_band = [{"band": f"€{iv.left:.0f}-{iv.right:.0f}", "n": int((bands == iv).sum()),
                      "median_signed_pct_error": float(np.median(signed[bands == iv]))} for iv in bands.categories]

    # 7. Evidence (train + validation only) and validity checks
    evidence = D.run_evidence(df, X_all, y_all, train_idx, val_idx, alpha, lr)
    validity = D.run_validity(df, X_all, y_all, (train_idx, val_idx, test_idx), preds, pre, sk_model,
                              y_val, sk_model.predict(Z_val), alpha)

    # 8. Save artifacts for the app
    joblib.dump(pre, ART / "preprocessor.joblib")
    joblib.dump(sk_model, ART / "sklearn_ridge.joblib")
    torch.save(manual_params, ART / "torch_manual.pt")
    torch.save(nn_model.state_dict(), ART / "torch_nn.pt")
    (ART / "baseline.json").write_text(json.dumps(baseline, indent=2))
    (ART / "baseline_comps.json").write_text(json.dumps(baseline_comps, indent=2))
    coefs.to_csv(ART / "coefficients.csv", index=False)

    pd.DataFrame({
        "id": df["id"].iloc[test_idx].to_numpy(),
        "district": X_te["district"].to_numpy(),
        "room_type": df["room_type"].iloc[test_idx].to_numpy(),
        "weekday_quote": weekday.astype(int),
        "actual_eur": actual,
        **{f"pred_{k}_eur": np.exp(v) for k, v in preds.items()},
    }).to_csv(ART / "test_predictions.csv", index=False)

    train_view = X_tr.copy()
    train_view["price_eur"] = np.exp(y_tr)
    train_view["log_price"] = y_tr
    train_view.to_csv(ART / "train_features.csv", index=False)

    # Location lookup for the app's "price a listing" form: a typical
    # coordinate for each neighbourhood, from training listings only.
    (df.iloc[train_idx].groupby("neighbourhood")
       .agg(district=("district", "first"), latitude=("latitude", "median"),
            longitude=("longitude", "median"), n_listings=("id", "size"))
       .reset_index()
       .to_csv(ART / "neighbourhoods.csv", index=False))

    summary = {
        "cleaning_log": cleaning_log,
        "split_sizes": {"train": len(y_tr), "val": len(y_val), "test": len(y_te),
                        "test_weekday_only": int(weekday.sum())},
        "n_model_inputs": len(feature_names),
        "alpha": alpha, "alpha_val_rmse": alpha_val,
        "lr": lr, "lr_table": lr_table, "epochs": EPOCHS, "converged_rel_gap": CONVERGED_REL_GAP,
        "gd_stability": stability,
        "test_metrics": results,
        "test_metrics_weekday_only": results_weekday,
        "test_ci_95": test_ci,
        "test_ci_95_weekday_only": test_ci_weekday,
        "error_by_price_band": error_by_band,
        "mean_pred_over_mean_actual": float(np.exp(preds["sklearn"]).mean() / actual.mean()),
        "agreement": agreement,
        "intercepts": intercepts,
        **evidence,
        "validity": validity,
        "loss_curves": {"log_every": LOG_EVERY, "torch_manual": manual_hist, "torch_nn": nn_hist,
                        "sklearn_optimum": sk_obj},
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    D.print_summary(summary)
    print(f"\nSaved artifacts to {ART}/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
