"""
Supporting evidence and validity checks for REPORT.md and the dashboard.

Everything that decides the model (feature evidence, target, split) uses
train + validation only. The checks marked "test" describe the final model
on the test split after it has been fitted; none of them changes it.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.neighbors import BallTree

from features import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS, NUMERIC_FEATURES, QUOTE_CONTROLS
from pipeline import (
    N_BOOTSTRAP, N_CV_FOLDS, N_SPLIT_REPEATS, SEED,
    build_preprocessor, fit_manual_torch, fit_sklearn, gd_stability, metrics, predict_manual,
)

# ----------------------------------------------------------------------------
# Supporting evidence for REPORT.md (train + validation data only, never test)
# ----------------------------------------------------------------------------

def cv_rmse(X, y, groups, alpha, numeric=None, binary=None, categorical=None, raw_target=False):
    """Host-grouped K-fold: per-fold RMSE(log) and median % error for sklearn ridge."""
    rmse, ape = [], []
    for tr, va in GroupKFold(n_splits=N_CV_FOLDS).split(X, groups=groups):
        pre = build_preprocessor(numeric, binary, categorical).fit(X.iloc[tr])
        target = np.exp(y[tr]) if raw_target else y[tr]
        pred = fit_sklearn(pre.transform(X.iloc[tr]), target, alpha).predict(pre.transform(X.iloc[va]))
        pred_log = np.log(np.clip(pred, 10, None)) if raw_target else pred   # raw model can go negative
        m = metrics(y[va], pred_log)
        rmse.append(m["rmse_log"])
        ape.append(m["median_abs_pct_error"])
    return np.array(rmse), np.array(ape)


def feature_ablation(X, y, groups, alpha) -> dict:
    """Drop one feature group at a time and measure how much worse the model gets."""
    base_rmse, base_ape = cv_rmse(X, y, groups, alpha)
    out = {"_all_features": {"rmse_log": float(base_rmse.mean()), "median_abs_pct_error": float(base_ape.mean())}}
    for name, cols in FEATURE_GROUPS.items():
        rmse, ape = cv_rmse(
            X, y, groups, alpha,
            numeric=[c for c in NUMERIC_FEATURES if c not in cols],
            binary=[c for c in BINARY_FEATURES if c not in cols],
            categorical=[c for c in CATEGORICAL_FEATURES if c not in cols],
        )
        out[name] = {
            "rmse_log": float(rmse.mean()),
            "rmse_increase": float((rmse - base_rmse).mean()),
            "median_abs_pct_error": float(ape.mean()),
            "folds_worse": int((rmse > base_rmse).sum()),
        }
    return out


def compare_targets(X, y, groups, alpha) -> dict:
    """Same features, same folds: model log(price) vs raw price in euros."""
    out = {}
    for name, raw in [("log_price", False), ("raw_price", True)]:
        rmse, ape = cv_rmse(X, y, groups, alpha, raw_target=raw)
        out[name] = {"median_abs_pct_error": float(ape.mean()), "rmse_log": float(rmse.mean())}
    return out


def bootstrap_ci(y_log, preds: dict, hosts, n=N_BOOTSTRAP) -> dict:
    """95% confidence intervals for the test metrics, resampling HOSTS rather
    than listings: listings from one host are near-copies, so resampling them
    one by one would pretend the test set holds more independent evidence than
    it does. Also gives the interval for the final model minus every other row
    (baselines, model without quote controls), computed on the same resamples
    (a paired comparison)."""
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(hosts, return_inverse=True)
    rows_of = [np.nonzero(inv == i)[0] for i in range(len(uniq))]
    keys = ["median_abs_pct_error", "within_20pct", "rmse_log"]
    draws = {m: {k: [] for k in keys} for m in preds}
    for _ in range(n):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(uniq), len(uniq))])
        for m, p in preds.items():
            r = metrics(y_log[idx], p[idx])
            for k in keys:
                draws[m][k].append(r[k])
    out = {m: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
           for m, d in draws.items()}
    for other in [m for m in preds if m != "sklearn"]:
        diff = np.array(draws["sklearn"]["median_abs_pct_error"]) - np.array(draws[other]["median_abs_pct_error"])
        out[f"sklearn_minus_{other}_median_abs_pct_error"] = [float(np.percentile(diff, 2.5)),
                                                               float(np.percentile(diff, 97.5))]
    return out


def compare_split_strategies(df, X_all, y_all, alpha) -> dict:
    """Mean test metrics of sklearn ridge under host-grouped vs random splits,
    averaged over several seeds (one seed each would just compare two different
    random test sets)."""
    out = {"grouped": [], "random": []}
    for seed in range(N_SPLIT_REPEATS):
        splits = {
            "grouped": next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
                            .split(df, groups=df["host_id"])),
            "random": train_test_split(np.arange(len(df)), test_size=0.15, random_state=seed),
        }
        for name, (tr, te) in splits.items():
            pre = build_preprocessor().fit(X_all.iloc[tr])
            m = fit_sklearn(pre.transform(X_all.iloc[tr]), y_all[tr], alpha)
            out[name].append(metrics(y_all[te], m.predict(pre.transform(X_all.iloc[te]))))
    return {name: {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            for name, runs in out.items()}


def rerun_dummy_trap(X_tr, y_tr, alpha, lr) -> dict:
    """Rebuild the FIRST version's preprocessing (all 21 district dummies, no
    reference category, unscaled) and fit sklearn and the manual loop on it.
    Predictions agree, coefficients do not: the dummies sum to 1, duplicating
    the intercept, so 'intercept - c, every district + c' gives the same
    predictions for any c. Only the tiny ridge penalty pins down one solution,
    and gradient descent barely moves along that almost flat direction."""
    pre = build_preprocessor(drop_reference=False).fit(X_tr)
    Z = pre.transform(X_tr)
    names = pre.get_feature_names_out()
    sk = fit_sklearn(Z, y_tr, alpha)
    manual, _ = fit_manual_torch(Z, y_tr, alpha, lr)
    w_m = manual["w"].numpy()
    district = np.array([n.startswith("cat__") for n in names])
    gap = np.abs(w_m - sk.coef_)
    return {
        "max_abs_coef_diff": float(gap.max()),
        "max_abs_coef_diff_non_district": float(gap[~district].max()),
        "mean_district_shift_manual_minus_sklearn": float((w_m - sk.coef_)[district].mean()),
        "intercept_shift_manual_minus_sklearn": float(manual["b"].item() - sk.intercept_),
        "max_abs_pred_diff_log": float(np.abs(predict_manual(manual, Z) - sk.predict(Z)).max()),
        "hessian_lambda_min": abs(gd_stability(Z)["hessian_lambda_min"]),   # 0 = perfectly flat direction
    }


# ----------------------------------------------------------------------------
# Validity checks: is the model measuring the listing, and does it hold up?
# ----------------------------------------------------------------------------

def asks_vs_model(y_log, pred_log) -> float:
    """Median of actual/predicted - 1. Positive = listings ask MORE than the model."""
    return float(np.exp(np.median(y_log - pred_log)) - 1)


def quote_effects(pre, sk_model) -> dict:
    """The two quote controls in plain units: undo the standardisation of each
    input to get its effect per original unit."""
    scale = dict(zip(NUMERIC_FEATURES, pre.named_transformers_["num"].named_steps["scale"].scale_))
    coef = dict(zip(pre.get_feature_names_out(), sk_model.coef_))
    per_unit = {c: coef[f"num__{c}"] / scale[c] for c in QUOTE_CONTROLS}
    return {
        "all_weekend_vs_all_weekday_per_night": float(np.exp(per_unit["weekend_share"]) - 1),
        "seven_vs_one_night_quote_per_night": float(np.exp(per_unit["log_quote_nights"] * np.log(7)) - 1),
    }


def date_effect_check(X_te, y_te, pred_final, pred_listing_only) -> list:
    """How much do listings ask above/below the model, by which nights were
    quoted? Compared for the model WITHOUT the quote controls (the problem) and
    the final model (the fix), on the test set."""
    ws = X_te["weekend_share"].to_numpy()
    nights = np.exp(X_te["log_quote_nights"].to_numpy()).round()
    groups = {
        "No Fri/Sat night in the quoted stay": ws == 0,
        "Some Fri/Sat nights (up to half)": (ws > 0) & (ws <= 0.5),
        "Mostly Fri/Sat nights": ws > 0.5,
        "1-night quote": nights == 1,
        "2-3 night quote": (nights >= 2) & (nights <= 3),
        "4-7 night quote": nights >= 4,
    }
    return [{"group": g, "n": int(m.sum()),
             "without_quote_controls": asks_vs_model(y_te[m], pred_listing_only[m]),
             "final_model": asks_vs_model(y_te[m], pred_final[m])} for g, m in groups.items()]


def nights_table(X, y, df_part, alpha, min_n=20) -> list:
    """Night by night: how much 1-night quotes ask above a model WITHOUT the
    quote controls (out-of-fold, host-grouped CV on train + validation).
    Shows which nights carry the date effect: ordinary weekdays near zero,
    Friday/Saturday higher, and a few weekends far higher (likely events)."""
    num = [c for c in NUMERIC_FEATURES if c not in QUOTE_CONTROLS]
    groups = df_part["host_id"].to_numpy()
    resid = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=N_CV_FOLDS).split(X, groups=groups):
        pre = build_preprocessor(numeric=num).fit(X.iloc[tr])
        resid[va] = y[va] - fit_sklearn(pre.transform(X.iloc[tr]), y[tr], alpha).predict(pre.transform(X.iloc[va]))
    checkin = pd.to_datetime(df_part["price_quote_checkin_date"])
    one = (df_part["quote_nights"] == 1).to_numpy()
    t = (pd.DataFrame({"night": checkin[one].dt.date, "r": resid[one]})
           .groupby("night")["r"].agg(["size", "median"]))
    t = t[t["size"] >= min_n]
    return [{"night": str(d), "weekday": pd.Timestamp(d).day_name()[:3], "n": int(r["size"]),
             "asks_vs_model_without_controls": float(np.exp(r["median"]) - 1)} for d, r in t.iterrows()]


def location_check(X, y, df_part, alpha) -> dict:
    """Is location signal left in the errors? Out-of-fold residuals (host-grouped
    CV on train + validation), then (1) the share of residual variation that the
    126 neighbourhoods explain and (2) the correlation between a listing's
    residual and the mean residual of its 10 nearest listings from OTHER hosts.
    If location were badly captured, neighbours would share their errors."""
    groups = df_part["host_id"].to_numpy()
    resid = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=N_CV_FOLDS).split(X, groups=groups):
        pre = build_preprocessor().fit(X.iloc[tr])
        resid[va] = y[va] - fit_sklearn(pre.transform(X.iloc[tr]), y[tr], alpha).predict(pre.transform(X.iloc[va]))
    nb = pd.DataFrame({"n": df_part["neighbourhood"].to_numpy(), "r": resid}).groupby("n")["r"].agg(["mean", "size"])
    share_neighbourhood = float((nb["size"] * nb["mean"] ** 2).sum() / (resid ** 2).sum())

    coords = np.radians(df_part[["latitude", "longitude"]].to_numpy())
    _, ind = BallTree(coords, metric="haversine").query(coords, k=60)
    neighbour_mean = np.array([resid[row[groups[row] != groups[i]][:10]].mean() for i, row in enumerate(ind)])
    big = nb[nb["size"] >= 40].sort_values("mean")
    worst = pd.concat([big.head(3), big.tail(3)])
    return {
        "residual_share_explained_by_neighbourhood": share_neighbourhood,
        "corr_residual_vs_10_nearest_other_hosts": float(np.corrcoef(resid, neighbour_mean)[0, 1]),
        "most_systematic_neighbourhoods_asks_vs_model": {k: float(np.exp(v) - 1) for k, v in worst["mean"].items()},
    }


def calibration_check(y_te, pred_te, hosts_te, y_val, pred_val, y_train_mean, n_boot=500) -> dict:
    """Are cheap or expensive listings mispriced? Three pieces of evidence.

    1. Errors grouped by the PREDICTED price (what is known when pricing a new
       flat), with 95% host-bootstrap intervals. Grouping by the ACTUAL price
       instead always shows high-priced listings under-predicted and low-priced
       ones over-predicted, for any imperfect model (regression to the mean).
    2. A simulation of that: a model that is right on average by construction,
       with prices scattering randomly around it, grouped by actual price.
    3. Stretching predictions away from the average by a factor k (validation
       split): fixes the extreme groups, but does it lower the overall error?
    """
    rng = np.random.default_rng(SEED)
    q = np.quantile(pred_te, [0, .05, .2, .4, .6, .8, .95, 1])
    q[-1] += 1e-9
    by_pred = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (pred_te >= lo) & (pred_te < hi)
        r, h = y_te[m] - pred_te[m], hosts_te[m]
        uniq = np.unique(h)
        rows = [np.nonzero(h == u)[0] for u in uniq]
        boots = [np.median(r[np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])])
                 for _ in range(n_boot)]
        by_pred.append({"band": f"€{np.exp(lo):.0f}-{np.exp(hi):.0f}", "n": int(m.sum()),
                        "model_vs_actual": float(np.exp(-np.median(r)) - 1),
                        "ci_95": [float(np.exp(-np.percentile(boots, 97.5)) - 1), float(np.exp(-np.percentile(boots, 2.5)) - 1)]})

    simulated_actual = pred_te + rng.normal(0, np.std(y_te - pred_te), len(pred_te))
    sim_bands = pd.qcut(simulated_actual, 5)
    simulation = [float(np.median(np.exp(pred_te[sim_bands == iv] - simulated_actual[sim_bands == iv]) - 1))
                  for iv in sim_bands.categories]

    stretch = []
    for k in [1.0, 1.1, 1.2, 1.3, 1.5]:
        s = y_train_mean + k * (pred_val - y_train_mean)
        e = np.exp(s - y_val) - 1
        top = y_val >= np.quantile(y_val, 0.8)
        stretch.append({"k": k, "rmse_log": float(np.sqrt(np.mean((y_val - s) ** 2))),
                        "median_abs_pct_error": float(np.median(np.abs(e))),
                        "top_fifth_bias": float(np.median(e[top]))})
    # Does the cheapest-predicted 5% repeat on the validation split?
    bottom_val = pred_val <= np.quantile(pred_val, 0.05)
    bottom_val_effect = float(np.exp(-np.median(y_val[bottom_val] - pred_val[bottom_val])) - 1)

    # A per-band correction learned on validation, applied to test (evaluated, not adopted)
    edges = np.quantile(pred_val, [.05, .2, .4, .6, .8, .95])
    band_v, band_t = np.digitize(pred_val, edges), np.digitize(pred_te, edges)
    offsets = np.array([np.median(y_val[band_v == b] - pred_val[band_v == b]) for b in range(len(edges) + 1)])
    corrected = pred_te + offsets[band_t]
    def summarise(p):
        return {"rmse_log": float(np.sqrt(np.mean((y_te - p) ** 2))),
                "median_abs_pct_error": float(np.median(np.abs(np.exp(p - y_te) - 1)))}
    return {"by_predicted_band": by_pred, "bottom_5pct_validation_model_vs_actual": bottom_val_effect,
            "simulated_unbiased_model_by_actual_fifth": simulation,
            "stretch_check_validation": stretch,
            "per_band_correction_test": {"without": summarise(pred_te), "with": summarise(corrected)}}


def new_listing_check(n_reviews, y_te, pred) -> list:
    """The model learns from existing hosts but is meant for new ones. Do
    brand-new listings (0 reviews) ask more or less than the model suggests?
    Reviews are used here for evaluation only, never as a model input."""
    bins = [(-1, 0, "0 reviews (brand new)"), (0, 5, "1-5"), (5, 30, "6-30"), (30, 100, "31-100"), (100, 10**9, "more than 100")]
    out = []
    for lo, hi, label in bins:
        m = (n_reviews > lo) & (n_reviews <= hi)
        out.append({"reviews": label, "n": int(m.sum()), "asks_vs_model": asks_vs_model(y_te[m], pred[m]),
                    "median_abs_pct_error": float(np.median(np.abs(np.exp(pred[m] - y_te[m]) - 1)))})
    return out



# ----------------------------------------------------------------------------
# Run everything (called from train.py)
# ----------------------------------------------------------------------------

def run_evidence(df, X_all, y_all, train_idx, val_idx, alpha, lr) -> dict:
    """Evidence behind the modelling decisions. Train + validation only."""
    tv = np.concatenate([train_idx, val_idx])
    X_tv, y_tv, g_tv = X_all.iloc[tv].reset_index(drop=True), y_all[tv], df["host_id"].iloc[tv].to_numpy()
    return {
        "feature_ablation": feature_ablation(X_tv, y_tv, g_tv, alpha),
        "target_comparison": compare_targets(X_tv, y_tv, g_tv, alpha),
        "split_comparison": compare_split_strategies(df, X_all, y_all, alpha),
        "dummy_trap_rerun": rerun_dummy_trap(X_all.iloc[train_idx], y_all[train_idx], alpha, lr),
    }


def run_validity(df, X_all, y_all, idx, preds, pre, sk_model, y_val, pred_val, alpha) -> dict:
    """Does the final model measure the listing, and where does it miss?"""
    train_idx, val_idx, test_idx = idx
    tv = np.concatenate([train_idx, val_idx])
    X_tv, y_tv, df_tv = X_all.iloc[tv].reset_index(drop=True), y_all[tv], df.iloc[tv].reset_index(drop=True)
    X_te, y_te = X_all.iloc[test_idx], y_all[test_idx]
    checkin_month = pd.to_datetime(df["price_quote_checkin_date"]).dt.month
    return {
        "share_of_quotes_june_july": float(checkin_month.isin([6, 7]).mean()),
        "quote_effects": quote_effects(pre, sk_model),
        "date_effect_check": date_effect_check(X_te, y_te, preds["sklearn"], preds["ridge_no_dates"]),
        "nights_table": nights_table(X_tv, y_tv, df_tv, alpha),
        "location_check": location_check(X_tv, y_tv, df_tv, alpha),
        "new_listing_check": new_listing_check(df["number_of_reviews"].iloc[test_idx].to_numpy(), y_te, preds["sklearn"]),
        "calibration": calibration_check(y_te, preds["sklearn"], df["host_id"].iloc[test_idx].to_numpy(),
                                         y_val, pred_val, float(y_all[train_idx].mean())),
    }


def print_summary(s: dict) -> None:
    """Readable console summary of summary.json."""
    print("\nTest set, all quotes (host-grouped split):")
    print(pd.DataFrame(s["test_metrics"]).T.round(4).to_string())
    print("\nTest set, weekday-only quotes (what the app outputs):")
    print(pd.DataFrame(s["test_metrics_weekday_only"]).T.round(4).to_string())
    ci = s["test_ci_95"]
    print("\n95% CI (host bootstrap), median abs % error:",
          {k: [round(x, 3) for x in v["median_abs_pct_error"]] for k, v in ci.items() if isinstance(v, dict)})
    print("Final model minus each row (negative = final model better):",
          {k.replace("sklearn_minus_", "").replace("_median_abs_pct_error", ""): [round(x, 3) for x in v]
           for k, v in ci.items() if k.startswith("sklearn_minus_")})
    print("\nAgreement between implementations:", {k: v for k, v in s["agreement"].items()})
    print("\nFeature ablation (host-grouped CV, train+val): RMSE increase when a group is removed")
    for k, v in s["feature_ablation"].items():
        if k != "_all_features":
            print(f"  {k:30s} {v['rmse_increase']:+.4f}  (worse in {v['folds_worse']}/{N_CV_FOLDS} folds)")
    print("Target comparison:", {k: round(v["median_abs_pct_error"], 4) for k, v in s["target_comparison"].items()})
    print("Grouped vs random split, R2(log):", {k: round(v["r2_log"], 4) for k, v in s["split_comparison"].items()})
    print("Dummy-trap rerun:", {k: round(v, 4) for k, v in s["dummy_trap_rerun"].items()})

    validity = s["validity"]
    print("\nValidity checks")
    print(f"  quotes for June/July check-ins: {validity['share_of_quotes_june_july']:.1%}")
    print("  quote effects:", {k: f"{v:+.1%}" for k, v in validity["quote_effects"].items()})
    print("  date effect (test, listings ask vs model):")
    for r in validity["date_effect_check"]:
        print(f"    {r['group']:38s} n={r['n']:5d}  without controls {r['without_quote_controls']:+.1%}  final {r['final_model']:+.1%}")
    print("  1-night quotes by night (asks vs model without quote controls):",
          [(r["night"][5:], r["weekday"], f"{r['asks_vs_model_without_controls']:+.0%}") for r in validity["nights_table"]])
    lc = validity["location_check"]
    print(f"  location: neighbourhoods explain {lc['residual_share_explained_by_neighbourhood']:.1%} of leftover error; "
          f"residual correlation with 10 nearest other-host listings {lc['corr_residual_vs_10_nearest_other_hosts']:.3f}")
    print("  new listings (test):", [(r["reviews"], r["n"], f"{r['asks_vs_model']:+.1%}") for r in validity["new_listing_check"]])
    cal = validity["calibration"]
    print("  by predicted price (test, model vs actual):",
          [(r["band"], f"{r['model_vs_actual']:+.1%}", f"[{r['ci_95'][0]:+.0%},{r['ci_95'][1]:+.0%}]") for r in cal["by_predicted_band"]])
    print("  unbiased model by actual fifth (simulation):", [f"{v:+.0%}" for v in cal["simulated_unbiased_model_by_actual_fifth"]])
    print("  stretch check (validation):", [(r["k"], round(r["rmse_log"], 4)) for r in cal["stretch_check_validation"]])
    print(f"  bottom 5% on validation: {cal['bottom_5pct_validation_model_vs_actual']:+.1%};  per-band correction on test:",
          {k: (round(v["rmse_log"], 4), f"{v['median_abs_pct_error']:.2%}") for k, v in cal["per_band_correction_test"].items()})
