"""
Training pipeline: suggested nightly price for a new Airbnb host in Madrid.

Run via:  uv run python main.py train

One command, start to finish, no manual steps:
 1. Clean the raw Inside Airbnb export (clean_data.run_cleaning).
 2. Define the target, log(price), and engineer features (features.py).
 3. Split train / validation / test GROUPED BY HOST, so no host's listings
    appear on both sides of a split.
 4. Fit the preprocessor (impute, scale, one-hot) on the training split only.
 5. Fit ONE model, ridge regression on log(price), three ways:
      a. scikit-learn Ridge (exact solver)
      b. manual PyTorch: raw tensors, autograd, hand-written update step
      c. standard PyTorch: nn.Module + nn.MSELoss + torch.optim.SGD
    Tuning is limited to the ridge strength (alpha) and the learning rate.
 6. Compare the three with each other and with a naive baseline on the test set.
 7. Supporting evidence for REPORT.md: feature-group ablation, log vs raw
    target, grouped vs random split, a rerun of the dummy-variable trap that
    made the three implementations disagree in the first version, and
    validity checks (date effects, leftover location signal, new listings).
 8. Save everything app.py needs to artifacts/. The app never retrains.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clean_data import run_cleaning
from features import (
    BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS, NUMERIC_FEATURES, QUOTE_CONTROLS,
    LinearRegressionModel, engineer_features,
)

ART = Path("artifacts")
SEED = 42

ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]  # ridge strength, sklearn convention
LR_GRID = [0.01, 0.03, 0.1, 0.2, 0.3]                   # learning rates for the torch versions
EPOCHS = 3000                                           # full-batch gradient descent steps
LOG_EVERY = 10                                          # record the loss every N epochs
CONVERGED_REL_GAP = 1e-5     # "reached sklearn's optimum" = objective within 0.001% of it
REFERENCE_DISTRICT = "Centro"
N_CV_FOLDS = 5               # host-grouped folds for the ablation and target comparisons
N_SPLIT_REPEATS = 5          # seeds for the grouped-vs-random split comparison
N_BOOTSTRAP = 1000           # host-level bootstrap resamples for test-set confidence intervals

np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------------------------------------------------------
# Splitting and preprocessing
# ----------------------------------------------------------------------------

def grouped_split(df: pd.DataFrame):
    """70 / 15 / 15 split where every host lands in exactly one part."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    trainval_idx, test_idx = next(outer.split(df, groups=df["host_id"]))
    trainval = df.iloc[trainval_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEED)
    tr_idx, val_idx = next(inner.split(trainval, groups=trainval["host_id"]))
    train_idx, val_idx = trainval_idx[tr_idx], trainval_idx[val_idx]

    hosts = [set(df.iloc[i]["host_id"]) for i in (train_idx, val_idx, test_idx)]
    assert not (hosts[0] & hosts[1] or hosts[0] & hosts[2] or hosts[1] & hosts[2]), "host leak"
    return train_idx, val_idx, test_idx


def build_preprocessor(numeric=None, binary=None, categorical=None,
                       drop_reference=True) -> ColumnTransformer:
    """Median-impute and standardise numeric columns, standardise 0/1 flags,
    one-hot the district.

    The district one-hot drops Centro as the reference category. Keeping all
    21 dummies AND an intercept makes the columns linearly dependent (the
    dummies always sum to 1). drop_reference=False rebuilds that broken
    version on purpose, for the comparison in rerun_dummy_trap().
    The dummies are standardised like every other column, so gradient descent
    sees similar curvature in every direction.
    """
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    binary = BINARY_FEATURES if binary is None else binary
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical

    parts = []
    if numeric:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), numeric))
    if binary:
        parts.append(("bin", StandardScaler(), binary))
    if categorical:
        if drop_reference:
            onehot = Pipeline([
                ("onehot", OneHotEncoder(drop=[REFERENCE_DISTRICT], handle_unknown="ignore",
                                         sparse_output=False)),
                ("scale", StandardScaler()),
            ])
        else:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        parts.append(("cat", onehot, categorical))
    return ColumnTransformer(parts)


# ----------------------------------------------------------------------------
# Metrics and baseline
# ----------------------------------------------------------------------------

def metrics(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    """Business metrics in euros and %; RMSE and R2 on log price."""
    actual, pred = np.exp(y_log), np.exp(pred_log)
    ape = np.abs(pred - actual) / actual
    resid = y_log - pred_log
    return {
        "median_abs_pct_error": float(np.median(ape)),
        "within_20pct": float(np.mean(ape <= 0.20)),
        "mae_eur": float(np.mean(np.abs(pred - actual))),
        "rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        "r2_log": float(1 - np.sum(resid ** 2) / np.sum((y_log - y_log.mean()) ** 2)),
    }


def fit_baseline(X_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    """Naive baseline: median log price of the same district and room type.
    This is what a host would get by looking up comparable listings by hand."""
    tmp = X_train[["district", "is_private_room"]].copy()
    tmp["y"] = y_train
    table = tmp.groupby(["district", "is_private_room"])["y"].median()
    return {
        "table": {f"{d}|{r}": float(v) for (d, r), v in table.items()},
        "global_median": float(np.median(y_train)),
    }


def predict_baseline(baseline: dict, X: pd.DataFrame) -> np.ndarray:
    keys = X["district"].astype(str) + "|" + X["is_private_room"].astype(str)
    return keys.map(baseline["table"]).fillna(baseline["global_median"]).to_numpy()


# ----------------------------------------------------------------------------
# The three implementations
#
# sklearn Ridge minimises   ||y - Xw - b||^2 + alpha * ||w||^2        (a SUM)
# Dividing by n changes the scale but not the minimiser:
#                           mean((y - Xw - b)^2) + (alpha / n) * ||w||^2
# Both torch versions minimise that MEAN form, and none of the three
# penalises the intercept b. So all three solve the identical problem.
# ----------------------------------------------------------------------------

def fit_sklearn(X, y, alpha):
    return Ridge(alpha=alpha).fit(X, y)


def fit_manual_torch(X, y, alpha, lr, epochs=EPOCHS):
    """From scratch: tensors with requires_grad, autograd, manual update."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    n, p = X_t.shape

    w = torch.zeros(p, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []

    for epoch in range(epochs):
        pred = X_t @ w + b                               # forward pass
        mse = ((pred - y_t) ** 2).mean()
        loss = mse + (alpha / n) * (w ** 2).sum()        # ridge penalty, weights only
        loss.backward()                                  # autograd fills w.grad, b.grad

        with torch.no_grad():                            # the update must not be tracked
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                                   # gradients add up unless reset
        b.grad.zero_()

        if epoch % LOG_EVERY == 0 or epoch == epochs - 1:
            history.append(loss.item())
        if not np.isfinite(loss.item()) or loss.item() > 1e6:   # diverged: record it, stop early
            history.append(loss.item())
            break
    return {"w": w.detach(), "b": b.detach()}, history


def predict_manual(params, X):
    with torch.no_grad():
        return (torch.tensor(X, dtype=torch.float32) @ params["w"] + params["b"]).numpy()


def fit_standard_torch(X, y, alpha, lr, epochs=EPOCHS):
    """Framework way: nn.Module, a loss object, an optimizer.

    The ridge penalty becomes SGD weight decay. weight_decay=wd adds wd*w to
    the gradient, which is the gradient of (wd/2)*||w||^2. We need the
    gradient of (alpha/n)*||w||^2, so wd = 2*alpha/n. It applies to the weight
    only: the bias sits in its own parameter group with wd=0, otherwise the
    intercept would be shrunk too and the result would differ from sklearn.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    n, p = X_t.shape

    model = LinearRegressionModel(p)        # nn.Linear's default random init (not zeros)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([
        {"params": [model.linear.weight], "weight_decay": 2 * alpha / n},
        {"params": [model.linear.bias], "weight_decay": 0.0},
    ], lr=lr)

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        mse = loss_fn(model(X_t), y_t)
        mse.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == epochs - 1:
            with torch.no_grad():   # record the same objective as the manual loop
                history.append(mse.item() + (alpha / n) * (model.linear.weight ** 2).sum().item())
        if not np.isfinite(mse.item()) or mse.item() > 1e6:   # diverged: record it, stop early
            history.append(mse.item())
            break
    return model, history


def predict_standard(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def ridge_objective(X, y, w, b, alpha):
    """The mean-form objective, to measure how close GD got to sklearn's optimum."""
    r = y - X @ w - b
    return float(np.mean(r ** 2) + alpha / len(y) * np.sum(w ** 2))


# ----------------------------------------------------------------------------
# Tuning: alpha and learning rate (the only two hyperparameters touched)
# ----------------------------------------------------------------------------

def gd_stability(Z: np.ndarray) -> dict:
    """Full-batch gradient descent on this loss converges only if
    lr < 2 / lambda_max, where lambda_max is the largest eigenvalue of the
    Hessian (2/n) Z^T Z (with a column of ones for the intercept). The ridge
    term adds 2*alpha/n to every eigenvalue, which is negligible here."""
    Zc = np.hstack([Z, np.ones((len(Z), 1))])
    eig = np.linalg.eigvalsh(2 / len(Z) * Zc.T @ Zc)
    return {"hessian_lambda_max": float(eig.max()), "hessian_lambda_min": float(eig.min()),
            "max_stable_lr": float(2 / eig.max()), "condition_number": float(eig.max() / eig.min())}


def choose_learning_rate(Z_tr, y_tr, alpha, sk_objective):
    """Every stable learning rate ends at the same optimum (the loss is convex),
    so validation error cannot separate them: the learning rate only changes
    how FAST we get there. Rule: pick the rate that reaches sklearn's
    objective (within CONVERGED_REL_GAP) in the fewest epochs."""
    table = {}
    for lr in LR_GRID:
        _, hist = fit_manual_torch(Z_tr, y_tr, alpha, lr)
        h = np.array(hist)
        rel_gap = (h - sk_objective) / sk_objective
        diverged = not np.isfinite(h[-1]) or h[-1] > 10 * sk_objective   # blew up instead of settling
        reached = np.nonzero(rel_gap <= CONVERGED_REL_GAP)[0]
        table[lr] = {
            "diverged": bool(diverged),
            "final_rel_gap": None if diverged else float(rel_gap[-1]),
            "epochs_to_converge": None if diverged or len(reached) == 0 else int(reached[0] * LOG_EVERY),
        }
    converged = {lr: r["epochs_to_converge"] for lr, r in table.items() if r["epochs_to_converge"] is not None}
    if not converged:
        raise RuntimeError(f"No learning rate in {LR_GRID} reached the optimum in {EPOCHS} epochs: {table}")
    best = min(converged, key=lambda lr: (converged[lr], -lr))
    return best, table


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
    it does. Also gives the interval for the model-minus-baseline difference,
    computed on the same resamples (a paired comparison)."""
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
    diff = np.array(draws["sklearn"]["median_abs_pct_error"]) - np.array(draws["baseline"]["median_abs_pct_error"])
    out["sklearn_minus_baseline_median_abs_pct_error"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
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


def date_effect_check(X_tr, y_tr, X_te, y_te, pred_final, alpha) -> list:
    """How much do listings ask above/below the model, by which nights were
    quoted? Compared for a model WITHOUT the quote controls (the problem) and
    the final model (the fix), on the test set."""
    num = [c for c in NUMERIC_FEATURES if c not in QUOTE_CONTROLS]
    pre = build_preprocessor(numeric=num).fit(X_tr)
    pred_listing_only = fit_sklearn(pre.transform(X_tr), y_tr, alpha).predict(pre.transform(X_te))
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
# Main pipeline
# ----------------------------------------------------------------------------

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
    print(f"\nSplit by host: train {len(y_tr):,}  val {len(y_val):,}  test {len(y_te):,} listings")

    # 4. Preprocessor, fitted on train only
    pre = build_preprocessor().fit(X_tr)
    Z_tr, Z_val, Z_te = pre.transform(X_tr), pre.transform(X_val), pre.transform(X_te)
    feature_names = list(pre.get_feature_names_out())
    print(f"{len(feature_names)} model inputs after preprocessing")

    baseline = fit_baseline(X_tr, y_tr)

    # 5a. Ridge strength: chosen on validation with sklearn (exact and fast)
    alpha_val = {a: metrics(y_val, fit_sklearn(Z_tr, y_tr, a).predict(Z_val))["rmse_log"]
                 for a in ALPHA_GRID}
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

    # 6. Evaluate everything on the test split (first and only time it is used)
    preds = {
        "baseline": predict_baseline(baseline, X_te),
        "sklearn": sk_model.predict(Z_te),
        "torch_manual": predict_manual(manual_params, Z_te),
        "torch_nn": predict_standard(nn_model, Z_te),
    }
    results = {name: metrics(y_te, p) for name, p in preds.items()}
    test_ci = bootstrap_ci(y_te, {"baseline": preds["baseline"], "sklearn": preds["sklearn"]},
                           df["host_id"].iloc[test_idx].to_numpy())

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
    agreement = {
        "max_abs_coef_diff_manual_vs_sklearn": float(gap.max()),
        "max_abs_coef_diff_nn_vs_sklearn": float((coefs.torch_nn - coefs.sklearn).abs().max()),
        "max_abs_coef_diff_manual_vs_nn": float((coefs.torch_manual - coefs.torch_nn).abs().max()),
        "feature_with_largest_gap": coefs.loc[gap.idxmax(), "feature"],
        "max_abs_pred_diff_log_manual_vs_sklearn": float(np.abs(preds["torch_manual"] - preds["sklearn"]).max()),
        "max_abs_pred_diff_log_nn_vs_sklearn": float(np.abs(preds["torch_nn"] - preds["sklearn"]).max()),
        "train_objective_sklearn": sk_obj,
        "train_objective_manual_final": manual_hist[-1],
        "train_objective_nn_final": nn_hist[-1],
    }

    # Where does the model miss? Median signed error by actual price band (test).
    actual = np.exp(y_te)
    bands = pd.qcut(actual, 5)
    signed = (np.exp(preds["sklearn"]) - actual) / actual
    error_by_band = [{"band": f"€{iv.left:.0f}-{iv.right:.0f}", "median_signed_pct_error": float(np.median(signed[bands == iv])),
                      "n": int((bands == iv).sum())} for iv in bands.categories]

    # 7. Evidence for the report, on train + validation only (test stays untouched)
    tv = np.concatenate([train_idx, val_idx])
    X_tv, y_tv, g_tv = X_all.iloc[tv].reset_index(drop=True), y_all[tv], df["host_id"].iloc[tv].to_numpy()
    ablation = feature_ablation(X_tv, y_tv, g_tv, alpha)
    target_comparison = compare_targets(X_tv, y_tv, g_tv, alpha)
    split_comparison = compare_split_strategies(df, X_all, y_all, alpha)
    dummy_trap = rerun_dummy_trap(X_tr, y_tr, alpha, lr)

    # Validity checks
    checkin_month = pd.to_datetime(df["price_quote_checkin_date"]).dt.month
    validity = {
        "share_of_quotes_june_july": float(checkin_month.isin([6, 7]).mean()),
        "quote_effects": quote_effects(pre, sk_model),
        "date_effect_check": date_effect_check(X_tr, y_tr, X_te, y_te, preds["sklearn"], alpha),
        "nights_table": nights_table(X_tv, y_tv, df.iloc[tv].reset_index(drop=True), alpha),
        "location_check": location_check(X_tv, y_tv, df.iloc[tv].reset_index(drop=True), alpha),
        "new_listing_check": new_listing_check(df["number_of_reviews"].iloc[test_idx].to_numpy(), y_te, preds["sklearn"]),
        "calibration": calibration_check(y_te, preds["sklearn"], df["host_id"].iloc[test_idx].to_numpy(),
                                         y_val, sk_model.predict(Z_val), float(y_tr.mean())),
    }

    # Print a readable summary
    print("\nTest set (host-grouped split):")
    print(pd.DataFrame(results).T.round(4).to_string())
    print("95% CI (host bootstrap), median abs % error:",
          {k: [round(x, 3) for x in v["median_abs_pct_error"]] for k, v in test_ci.items() if isinstance(v, dict)},
          " model minus baseline:", [round(x, 3) for x in test_ci["sklearn_minus_baseline_median_abs_pct_error"]])
    print("\nAgreement between implementations:", json.dumps(agreement, indent=2))
    print("\nFeature ablation (host-grouped CV, train+val): RMSE increase when a group is removed")
    for k, v in ablation.items():
        if k != "_all_features":
            print(f"  {k:30s} {v['rmse_increase']:+.4f}  (worse in {v['folds_worse']}/{N_CV_FOLDS} folds)")
    print("Target comparison:", {k: round(v["median_abs_pct_error"], 4) for k, v in target_comparison.items()})
    print("Grouped vs random split, R2(log):", {k: round(v["r2_log"], 4) for k, v in split_comparison.items()})
    print("Dummy-trap rerun:", {k: round(v, 4) for k, v in dummy_trap.items()})
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

    # 8. Save artifacts for the app
    joblib.dump(pre, ART / "preprocessor.joblib")
    joblib.dump(sk_model, ART / "sklearn_ridge.joblib")
    torch.save(manual_params, ART / "torch_manual.pt")
    torch.save(nn_model.state_dict(), ART / "torch_nn.pt")
    (ART / "baseline.json").write_text(json.dumps(baseline, indent=2))
    coefs.to_csv(ART / "coefficients.csv", index=False)

    pd.DataFrame({
        "id": df["id"].iloc[test_idx].to_numpy(),
        "district": X_te["district"].to_numpy(),
        "room_type": df["room_type"].iloc[test_idx].to_numpy(),
        "actual_eur": actual,
        **{f"pred_{k}_eur": np.exp(v) for k, v in preds.items()},
    }).to_csv(ART / "test_predictions.csv", index=False)

    train_view = X_tr.copy()
    train_view["price_eur"] = np.exp(y_tr)
    train_view["log_price"] = y_tr
    train_view.to_csv(ART / "train_features.csv", index=False)

    # Location lookup for the app's "price a listing" form: a typical
    # coordinate for each neighbourhood, from training listings only.
    tr_df = df.iloc[train_idx]
    (tr_df.groupby("neighbourhood")
          .agg(district=("district", "first"), latitude=("latitude", "median"),
               longitude=("longitude", "median"), n_listings=("id", "size"))
          .reset_index()
          .to_csv(ART / "neighbourhoods.csv", index=False))

    summary = {
        "cleaning_log": cleaning_log,
        "split_sizes": {"train": len(y_tr), "val": len(y_val), "test": len(y_te)},
        "n_model_inputs": len(feature_names),
        "alpha": alpha, "alpha_val_rmse": alpha_val,
        "lr": lr, "lr_table": lr_table, "epochs": EPOCHS, "converged_rel_gap": CONVERGED_REL_GAP,
        "gd_stability": stability,
        "test_metrics": results,
        "test_ci_95": test_ci,
        "error_by_price_band": error_by_band,
        "mean_pred_over_mean_actual": float(np.exp(preds["sklearn"]).mean() / actual.mean()),
        "agreement": agreement,
        "intercepts": intercepts,
        "feature_ablation": ablation,
        "target_comparison": target_comparison,
        "split_comparison": split_comparison,
        "dummy_trap_rerun": dummy_trap,
        "validity": validity,
        "loss_curves": {"log_every": LOG_EVERY, "torch_manual": manual_hist, "torch_nn": nn_hist,
                        "sklearn_optimum": sk_obj},
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"\nSaved artifacts to {ART}/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
