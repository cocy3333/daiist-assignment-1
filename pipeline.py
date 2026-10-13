"""
Building blocks of the training pipeline: the split, the preprocessing, the
metrics, the baselines and the three implementations of ridge regression.

train.py runs these in order; diagnostics.py and the tests reuse them.
Nothing here reads or writes files.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, LinearRegressionModel

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


GUEST_CAP = 6          # stronger baseline: 6+ guests share one group
MIN_GROUP = 5          # a group needs this many training listings to be used


def guest_group(X: pd.DataFrame) -> pd.Series:
    """Guest count (from log_accommodates), capped at GUEST_CAP."""
    return np.exp(X["log_accommodates"]).round().clip(1, GUEST_CAP).astype(int)


def fit_baseline_comps(X_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    """Stronger baseline: median log price of listings in the same district,
    same room type AND same number of guests. This is a careful manual
    comparison ("2-guest entire flats in Salamanca go for about X").
    Groups with fewer than MIN_GROUP training listings fall back to district
    and room type, then to the overall median."""
    tmp = X_train[["district", "is_private_room"]].copy()
    tmp["guests"] = guest_group(X_train)
    tmp["y"] = y_train
    g = tmp.groupby(["district", "is_private_room", "guests"])["y"].agg(["median", "size"])
    g = g[g["size"] >= MIN_GROUP]
    return {
        "table": {f"{d}|{r}|{n}": float(v) for (d, r, n), v in g["median"].items()},
        "fallback": fit_baseline(X_train, y_train),
    }


def predict_baseline_comps(baseline: dict, X: pd.DataFrame) -> np.ndarray:
    keys = (X["district"].astype(str) + "|" + X["is_private_room"].astype(str)
            + "|" + guest_group(X).astype(str))
    fallback = pd.Series(predict_baseline(baseline["fallback"], X), index=X.index)
    return keys.map(baseline["table"]).fillna(fallback).to_numpy()


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

