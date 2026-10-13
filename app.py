"""
Gradio dashboard for the Madrid Airbnb pricing model.

Run via:  uv run python main.py app

Everything shown is loaded from artifacts/ written by `main.py train`.
Nothing is fitted or retrained here: the preprocessor and all three models
are loaded from disk and only used to predict. Pricing a new listing goes
through the SAME features.engineer_features() as training, so the app cannot
drift from the training pipeline.

Tabs
  1. Model comparison   test metrics, agreement, prediction vs actual
  2. Training details   loss curves, learning-rate choice, the dummy-variable trap, coefficients
  3. Feature evidence   ablation, log vs raw target, grouped vs random split, rejected features
  4. Distributions      target and feature distributions, price by feature
  5. Validity checks    date effects, leftover location signal, new listings, scope
  6. Business view      tolerance slider: how often is the price right, and what misses cost
  7. Price a listing    describe a new flat, get a weekday and a Fri/Sat price
"""

import json
import os
import re
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from features import (
    AMENITY_EXAMPLES, BINARY_FEATURES, NUMERIC_FEATURES, LinearRegressionModel, engineer_features,
    reference_stay,
)

ART = Path("artifacts")
EXPERIMENTS_MD = Path("experiments/results.md")

MODEL_LABELS = {
    "baseline": "Baseline (district × room median)",
    "sklearn": "scikit-learn Ridge",
    "torch_manual": "PyTorch manual loop",
    "torch_nn": "PyTorch nn.Module + optim",
}
IMPLEMENTATIONS = ["sklearn", "torch_manual", "torch_nn"]
# First three slots of a colour-blind-validated categorical palette, plus a neutral gray
COLORS = {"baseline": "#8a8984", "sklearn": "#2a78d6", "torch_manual": "#eb6834", "torch_nn": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#52514e"
TEMPLATE = "plotly_white"


# ----------------------------------------------------------------------------
# Load artifacts (fail with a clear message if training hasn't run)
# ----------------------------------------------------------------------------

REQUIRED = ["preprocessor.joblib", "sklearn_ridge.joblib", "torch_manual.pt", "torch_nn.pt",
            "baseline.json", "summary.json", "coefficients.csv", "test_predictions.csv",
            "train_features.csv", "neighbourhoods.csv"]
missing = [f for f in REQUIRED if not (ART / f).exists()]
if missing:
    raise SystemExit(f"Missing artifacts {missing}. Run `uv run python main.py train` first.")

preprocessor = joblib.load(ART / "preprocessor.joblib")
sk_model = joblib.load(ART / "sklearn_ridge.joblib")
manual_params = torch.load(ART / "torch_manual.pt")
nn_state = torch.load(ART / "torch_nn.pt")
nn_model = LinearRegressionModel(nn_state["linear.weight"].shape[1])
nn_model.load_state_dict(nn_state)
nn_model.eval()
baseline = json.loads((ART / "baseline.json").read_text())
summary = json.loads((ART / "summary.json").read_text())
coefs = pd.read_csv(ART / "coefficients.csv")
test = pd.read_csv(ART / "test_predictions.csv")
train_view = pd.read_csv(ART / "train_features.csv")
neighbourhoods = pd.read_csv(ART / "neighbourhoods.csv").sort_values(["district", "neighbourhood"])


def style(fig, height=420, **kw):
    fig.update_layout(template=TEMPLATE, height=height, font=dict(color=INK, size=13),
                      title_font=dict(size=16), margin=dict(l=60, r=30, t=60, b=50), **kw)
    return fig


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ----------------------------------------------------------------------------
# Tab 1: model comparison
# ----------------------------------------------------------------------------

def metrics_table() -> pd.DataFrame:
    ci = summary["test_ci_95"]
    rows = []
    for key, label in MODEL_LABELS.items():
        m = summary["test_metrics"][key]
        ci_key = key if key in ci else ("sklearn" if key in IMPLEMENTATIONS else None)
        band = ci[ci_key]["median_abs_pct_error"] if ci_key else None
        rows.append({
            "Model": label,
            "Median abs % error": pct(m["median_abs_pct_error"]),
            "95% CI (host bootstrap)": f"{pct(band[0])} to {pct(band[1])}" if band else "",
            "Within ±20%": pct(m["within_20pct"]),
            "MAE (€/night)": f"{m['mae_eur']:.2f}",
            "RMSE (log)": f"{m['rmse_log']:.4f}",
            "R² (log)": f"{m['r2_log']:.4f}",
        })
    return pd.DataFrame(rows)


def agreement_md() -> str:
    a, ci = summary["agreement"], summary["test_ci_95"]
    diff = ci["sklearn_minus_baseline_median_abs_pct_error"]
    return (
        "### Do the three implementations agree?\n"
        f"Yes. Largest coefficient gap: sklearn vs manual loop **{a['max_abs_coef_diff_manual_vs_sklearn']:.1e}**, "
        f"sklearn vs nn.Module **{a['max_abs_coef_diff_nn_vs_sklearn']:.1e}**. "
        f"Largest prediction gap on the test set: **{a['max_abs_pred_diff_log_nn_vs_sklearn']:.1e}** in log price. "
        f"Final training objective: sklearn {a['train_objective_sklearn']:.7f}, "
        f"manual {a['train_objective_manual_final']:.7f}, nn.Module {a['train_objective_nn_final']:.7f}. "
        "Gaps of this size are float32 rounding in PyTorch versus float64 in sklearn, not a different solution. "
        "The CI column is shared by the three implementations because their predictions are identical.\n\n"
        f"**Model vs baseline:** the model's median error is {pct(-diff[1])} to {pct(-diff[0])} "
        "lower than the baseline's (95% interval from resampling test hosts), so the gain is not luck of the split."
    )


def pred_vs_actual(choice: str):
    keys = ["baseline"] + IMPLEMENTATIONS if choice == "All models" else [choice]
    fig = go.Figure()
    for k in keys:
        fig.add_trace(go.Scattergl(
            x=test["actual_eur"], y=test[f"pred_{k}_eur"], mode="markers", name=MODEL_LABELS[k],
            marker=dict(size=5, opacity=0.35, color=COLORS[k]),
            hovertemplate="actual €%{x:.0f}<br>predicted €%{y:.0f}<extra>" + MODEL_LABELS[k] + "</extra>",
        ))
    lo, hi = test["actual_eur"].min(), test["actual_eur"].max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(dash="dash", color=MUTED, width=2), hoverinfo="skip"))
    note = (" (the three implementations overlap exactly: only the last one drawn is visible)"
            if choice == "All models" else "")
    ticks = [20, 50, 100, 200, 500, 1000, 2000]
    fig = style(fig, 560, title=f"Predicted vs actual nightly price, test set{note}",
                xaxis_title="Actual €/night (log scale)", yaxis_title="Predicted €/night (log scale)",
                xaxis_type="log", yaxis_type="log", legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(tickvals=ticks, ticktext=[f"€{t}" for t in ticks])
    fig.update_yaxes(tickvals=ticks, ticktext=[f"€{t}" for t in ticks])
    return fig


def error_by_band_plot():
    """Same test predictions, grouped two ways: by actual price and by predicted price."""
    from plotly.subplots import make_subplots
    a = pd.DataFrame(summary["error_by_price_band"])
    p = pd.DataFrame(summary["validity"]["calibration"]["by_predicted_band"])
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08, shared_yaxes=True, subplot_titles=(
        "Grouped by ACTUAL price (fifths)", "Grouped by PREDICTED price (5% | fifths | 5%), with 95% intervals"))
    fig.add_trace(go.Bar(x=a["band"], y=a["median_signed_pct_error"], marker_color=COLORS["baseline"],
                         text=[f"{v:+.0%}" for v in a["median_signed_pct_error"]], textposition="outside",
                         cliponaxis=False, customdata=a["n"], showlegend=False,
                         hovertemplate="actual %{x}<br>model vs actual %{y:+.1%}<br>n = %{customdata}<extra></extra>"),
                  row=1, col=1)
    lo = p["model_vs_actual"] - p["ci_95"].map(lambda c: c[0])
    hi = p["ci_95"].map(lambda c: c[1]) - p["model_vs_actual"]
    fig.add_trace(go.Bar(x=p["band"], y=p["model_vs_actual"], marker_color=COLORS["sklearn"],
                         error_y=dict(type="data", array=hi, arrayminus=lo, color=MUTED, thickness=1.5, width=4),
                         customdata=p["n"], showlegend=False,
                         hovertemplate="predicted %{x}<br>model vs actual %{y:+.1%}<br>n = %{customdata}<extra></extra>"),
                  row=1, col=2)
    for c in (1, 2):
        fig.add_hline(y=0, line_color=MUTED, line_width=1, row=1, col=c)
    lim = max(a["median_signed_pct_error"].abs().max(), p["ci_95"].map(lambda c: max(abs(c[0]), abs(c[1]))).max()) * 1.3
    fig.update_yaxes(range=[-lim, lim], tickformat="+.0%")
    fig.update_xaxes(tickangle=-30)
    return style(fig, 460, title="Is the model biased for cheap or expensive listings? (test set; median of model ÷ actual − 1)")


def calibration_md() -> str:
    c = summary["validity"]["calibration"]
    sim = ", ".join(f"{v:+.0%}" for v in c["simulated_unbiased_model_by_actual_fifth"])
    st = {r["k"]: r for r in c["stretch_check_validation"]}
    bottom = c["by_predicted_band"][0]
    pb = c["per_band_correction_test"]
    return (
        "**How to read this.** The left chart looks like a strong bias: cheap listings over-predicted, expensive ones "
        "under-predicted. But grouping by the *actual* price selects on what the model cannot see: the cheapest group "
        "is full of flats priced below what their features suggest, the most expensive full of flats priced above it. "
        "Any imperfect model shows this pattern (regression to the mean). A simulated model that is right on average "
        f"by construction shows it too: {sim} across the same fifths.\n\n"
        "The right chart is the fair test, because the predicted price is what we know when pricing a new flat. There "
        "the errors are small and mostly within noise. The one effect that repeats on both the test and the validation "
        f"split is the cheapest-predicted 5% ({bottom['band']}): real prices there are higher than predicted "
        f"(model {bottom['model_vs_actual']:+.0%} on test, {c['bottom_5pct_validation_model_vs_actual']:+.0%} on "
        "validation), likely because very cheap rooms cluster near a price floor while "
        "the straight-line model keeps extrapolating downward.\n\n"
        "**Can the pattern be used to improve predictions?** Stretching predictions away from the average fixes the "
        f"extreme groups but raises the overall error on the validation split (RMSE {st[1.0]['rmse_log']:.4f} unchanged, "
        f"{st[1.1]['rmse_log']:.4f} at 10% stretch, {st[1.3]['rmse_log']:.4f} at 30%). A separate correction per "
        "predicted-price band, learned on validation and then tried on test, improved the median error slightly "
        f"({pct(pb['without']['median_abs_pct_error'])} → {pct(pb['with']['median_abs_pct_error'])}) but made the RMSE "
        f"worse ({pb['without']['rmse_log']:.4f} → {pb['with']['rmse_log']:.4f}). The two measures disagree, seven "
        f"offsets tuned on {summary['split_sizes']['val']:,} validation listings are easy to overfit, and adopting a "
        "fix because of a test result would turn the test set into a tuning set. So no correction is applied, and "
        "the test numbers above are for the uncorrected model."
    )


# ----------------------------------------------------------------------------
# Tab 2: training details
# ----------------------------------------------------------------------------

def loss_plot():
    lc = summary["loss_curves"]
    fig = go.Figure()
    for k in ["torch_manual", "torch_nn"]:
        ys = np.array(lc[k]) - lc["sklearn_optimum"]
        xs = np.arange(len(ys)) * lc["log_every"]
        fig.add_trace(go.Scatter(x=xs, y=np.clip(ys, 1e-9, None), name=MODEL_LABELS[k],
                                 line=dict(color=COLORS[k], width=2)))
    fig.update_yaxes(type="log", exponentformat="e")
    return style(fig, 420, title="Distance to sklearn's exact optimum during training (log scale)",
                 xaxis_title="Epoch (full-batch gradient descent)",
                 yaxis_title="Objective minus sklearn optimum", legend=dict(orientation="h", y=-0.2))


def lr_table() -> pd.DataFrame:
    rows = []
    for lr, r in summary["lr_table"].items():
        if r["diverged"]:
            status = "diverged"
        elif r["epochs_to_converge"] is None:
            status = f"too slow: not converged in {summary['epochs']} epochs"
        else:
            status = f"converged after {r['epochs_to_converge']} epochs"
        rows.append({"Learning rate": lr, "Result": status,
                     "Chosen": "✓" if float(lr) == summary["lr"] else ""})
    return pd.DataFrame(rows)


def training_md() -> str:
    s = summary["gd_stability"]
    alphas = ", ".join(f"{float(k):g}: {v:.4f}" for k, v in summary["alpha_val_rmse"].items())
    return (
        "### Hyperparameters (the only two tuned)\n"
        f"**Ridge alpha = {summary['alpha']:g}**, chosen on the validation split. Validation RMSE(log) by alpha: "
        f"{alphas}. The curve is flat up to 100: with {summary['split_sizes']['train']:,} training listings and "
        f"{summary['n_model_inputs']} inputs there is little overfitting for regularisation to fix.\n\n"
        f"**Learning rate = {summary['lr']:g}.** Gradient descent on this loss is stable only if "
        f"lr < 2 / λmax, where λmax = {s['hessian_lambda_max']:.2f} is the largest eigenvalue of the Hessian "
        f"(2/n)·XᵀX. That gives a limit of **{s['max_stable_lr']:.3f}**. Every stable rate ends at the same optimum "
        "(the loss is convex), so validation error cannot separate them; the rule is the fastest rate that reaches "
        f"sklearn's objective (within {summary['converged_rel_gap']:.0e} relative).\n\n"
        "In the chart below, both PyTorch versions close the gap to sklearn's exact optimum until it is about 1e-8, "
        "where it stops shrinking: that is the precision limit of the float32 numbers PyTorch computes with, "
        "not a remaining difference in the solution."
    )


def dummy_trap_md() -> str:
    d = summary["dummy_trap_rerun"]
    return (
        "### The first version disagreed: the dummy-variable trap (rerun on every training run)\n"
        "| | First version: all 21 district dummies + intercept | Final: Centro dropped as reference |\n"
        "|---|---|---|\n"
        f"| Largest coefficient gap, sklearn vs manual loop | **{d['max_abs_coef_diff']:.3f}** | "
        f"{summary['agreement']['max_abs_coef_diff_manual_vs_sklearn']:.1e} |\n"
        f"| Largest gap outside the district dummies | {d['max_abs_coef_diff_non_district']:.3f} | (same as above) |\n"
        f"| Average shift of district coefficients (manual minus sklearn) | {d['mean_district_shift_manual_minus_sklearn']:+.3f} | 0 |\n"
        f"| Shift of the intercept (manual minus sklearn) | {d['intercept_shift_manual_minus_sklearn']:+.3f} | 0 |\n"
        f"| Largest prediction gap (log price) | {d['max_abs_pred_diff_log']:.3f} | "
        f"{summary['agreement']['max_abs_pred_diff_log_manual_vs_sklearn']:.1e} |\n"
        f"| Smallest Hessian eigenvalue | **{d['hessian_lambda_min']:.1e}** (a flat direction) | "
        f"{summary['gd_stability']['hessian_lambda_min']:.3f} |\n\n"
        "The 21 dummies always sum to 1, which duplicates the intercept column. Raising every district by c and "
        "lowering the intercept by c gives identical predictions, so the loss is flat in that direction (eigenvalue 0). "
        "sklearn solves the equations exactly; gradient descent from zero barely moves along a flat direction and "
        "stops somewhere else on it. The opposite shifts of districts and intercept in the table are that direction."
    )


def coefficient_plot():
    c = coefs.copy()
    c["label"] = c["feature"].str.replace(r"^(num|bin|cat)__", "", regex=True)
    c = c.reindex(c["sklearn"].abs().sort_values().index)
    fig = go.Figure()
    for k in IMPLEMENTATIONS:
        fig.add_trace(go.Bar(y=c["label"], x=c[k], name=MODEL_LABELS[k], orientation="h",
                             marker_color=COLORS[k],
                             hovertemplate="%{y}: %{x:.4f}<extra>" + MODEL_LABELS[k] + "</extra>"))
    fig = style(fig, 1400, barmode="group", bargap=0.25, bargroupgap=0.1,
                title="Coefficients of the three implementations (standardised inputs, effect on log price)",
                xaxis_title="Coefficient", legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_layout(margin=dict(t=110))
    return fig


# ----------------------------------------------------------------------------
# Tab 3: feature evidence
# ----------------------------------------------------------------------------

def ablation_plot():
    ab = {k: v for k, v in summary["feature_ablation"].items() if not k.startswith("_")}
    names = sorted(ab, key=lambda k: ab[k]["rmse_increase"])
    vals = [ab[k]["rmse_increase"] for k in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker_color=COLORS["sklearn"],
        text=[f"+{v:.4f}  (worse in {ab[k]['folds_worse']}/5 folds)" for k, v in zip(names, vals)],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>RMSE increase %{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, max(vals) * 1.45])
    return style(fig, 380, title="Feature ablation: how much worse the model gets without each group",
                 xaxis_title="Increase in validation RMSE (log price), 5-fold CV grouped by host")


def evidence_md() -> str:
    t, s, ab = summary["target_comparison"], summary["split_comparison"], summary["feature_ablation"]
    return (
        f"Full model in the same cross-validation: RMSE(log) {ab['_all_features']['rmse_log']:.4f}, "
        f"median error {pct(ab['_all_features']['median_abs_pct_error'])}. All numbers on this tab come from "
        "training and validation listings only; the test set is kept for the final comparison.\n\n"
        "### Why log(price) and not price in euros\n"
        "| Target | Median abs % error (5-fold CV) |\n|---|---|\n"
        f"| log(price) | **{pct(t['log_price']['median_abs_pct_error'])}** |\n"
        f"| price in € | {pct(t['raw_price']['median_abs_pct_error'])} |\n\n"
        "Pricing errors matter in proportion: €20 off a €60 room is serious, €20 off €400 is noise. "
        "A linear model on log price learns percentage effects; on raw euros it is dominated by the expensive "
        "tail and can even predict negative prices.\n\n"
        "### Why the split is grouped by host\n"
        "| Split (sklearn, mean of 5 seeds) | R² (log) | Median abs % error |\n|---|---|---|\n"
        f"| Grouped by host (used) | {s['grouped']['r2_log']:.3f} | {pct(s['grouped']['median_abs_pct_error'])} |\n"
        f"| Random | {s['random']['r2_log']:.3f} | {pct(s['random']['median_abs_pct_error'])} |\n\n"
        "A random split puts near-identical flats from the same operator in both train and test, which flatters "
        "the squared-error metrics (R² above). The gap is small and does not show in the median error, because a "
        "linear model with a few dozen inputs cannot memorise single flats; a flexible model (trees, nearest "
        "neighbours) would exploit the leak far more. The grouped split is still the right one: a new host is "
        "by definition unseen."
    )


def experiments_md() -> str:
    if EXPERIMENTS_MD.exists():
        md = EXPERIMENTS_MD.read_text()
        return re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", md, flags=re.M)   # nest the headings
    return "Run `uv run python experiments/feature_experiments.py` to produce the candidate-feature table."


# ----------------------------------------------------------------------------
# Tab 4: distributions
# ----------------------------------------------------------------------------

FEATURE_CHOICES = ["district"] + NUMERIC_FEATURES + BINARY_FEATURES


def target_plot(log_scale: bool):
    col = "log_price" if log_scale else "price_eur"
    fig = px.histogram(train_view, x=col, nbins=80, color_discrete_sequence=[COLORS["sklearn"]])
    label = "log(€ per night)" if log_scale else "€ per night"
    return style(fig, 380, bargap=0.05, title=f"Target distribution, training split ({label})",
                 xaxis_title=label, yaxis_title="Listings")


def feature_plots(feature: str):
    df = train_view
    if feature == "district" or feature in BINARY_FEATURES:
        counts = df[feature].value_counts().sort_index()
        dist_fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values, marker_color=COLORS["sklearn"]))
        med = df.groupby(feature)["price_eur"].median().sort_values()
        rel_fig = go.Figure(go.Bar(x=med.index.astype(str), y=med.values, marker_color=COLORS["sklearn"]))
        style(dist_fig, 380, title=f"Distribution of {feature}", yaxis_title="Listings")
        style(rel_fig, 380, title=f"Median price by {feature}", yaxis_title="Median €/night")
    else:
        dist_fig = px.histogram(df, x=feature, nbins=60, color_discrete_sequence=[COLORS["sklearn"]])
        style(dist_fig, 380, bargap=0.05, title=f"Distribution of {feature} (missing values not shown)",
              yaxis_title="Listings")
        binned = pd.qcut(df[feature], q=10, duplicates="drop")
        med = df.groupby(binned, observed=True)["price_eur"].median()
        rel_fig = go.Figure(go.Scatter(x=[f"{i.left:.2f} to {i.right:.2f}" for i in med.index], y=med.values,
                                       mode="lines+markers", line=dict(color=COLORS["sklearn"], width=2),
                                       marker=dict(size=8)))
        style(rel_fig, 380, title=f"Median price across {feature} (decile bins)",
              xaxis_title=feature, yaxis_title="Median €/night")
    return dist_fig, rel_fig


# ----------------------------------------------------------------------------
# Tab 5: validity checks
# ----------------------------------------------------------------------------

V = summary["validity"]
_fs = [r for r in V["nights_table"] if r["weekday"] in ("Fri", "Sat")]
N_WEEKENDS = len({pd.Timestamp(r["night"]).isocalendar().week for r in _fs})
WKND_LO = min(r["asks_vs_model_without_controls"] for r in _fs)
WKND_HI = max(r["asks_vs_model_without_controls"] for r in _fs)
_top = max(_fs, key=lambda r: r["asks_vs_model_without_controls"])
WKND_TOP = f"{_top['weekday']} {pd.Timestamp(_top['night']).day} {pd.Timestamp(_top['night']).strftime('%B')}"


def nights_plot():
    t = pd.DataFrame(V["nights_table"])
    t["label"] = t["weekday"] + " " + pd.to_datetime(t["night"]).dt.strftime("%d %b")
    is_wknd = t["weekday"].isin(["Fri", "Sat"])
    fig = go.Figure(go.Bar(
        x=t["label"], y=t["asks_vs_model_without_controls"],
        marker_color=np.where(is_wknd, COLORS["torch_manual"], COLORS["sklearn"]),
        text=[f"{v:+.0%}" for v in t["asks_vs_model_without_controls"]], textposition="outside", cliponaxis=False,
        customdata=t["n"], hovertemplate="%{x}<br>%{y:+.1%} vs model<br>%{customdata} quotes<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.update_yaxes(tickformat="+.0%", range=[-0.4, 1.25])
    return style(fig, 420, title="1-night quotes by night: asking price vs a model that ignores the dates "
                 "(Fri/Sat in orange; nights with 20+ quotes; train + validation, out-of-fold)",
                 yaxis_title="Median: actual / predicted − 1")


def date_effect_plot():
    rows = [r for r in V["date_effect_check"] if "Fri/Sat" in r["group"]]
    labels = [f"{r['group']} (n={r['n']})" for r in rows]
    fig = go.Figure()
    for key, name, color in [("without_quote_controls", "Without quote controls", COLORS["baseline"]),
                             ("final_model", "Final model (with quote controls)", COLORS["sklearn"])]:
        vals = [r[key] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=vals, name=name, marker_color=color,
                             text=[f"{v:+.0%}" for v in vals], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>listings ask %{y:+.1%} vs the model<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    lim = max(abs(r[k]) for r in rows for k in ["without_quote_controls", "final_model"]) * 1.3
    fig.update_yaxes(range=[-lim, lim], tickformat="+.0%")
    return style(fig, 420, barmode="group", title="How much listings ask above the model, by which nights were quoted (test set)",
                 yaxis_title="Median: actual / predicted − 1", legend=dict(orientation="h", y=-0.2))


def validity_md() -> str:
    q, lc = V["quote_effects"], V["location_check"]
    worst = ", ".join(f"{k} {v:+.0%}" for k, v in lc["most_systematic_neighbourhoods_asks_vs_model"].items())
    rows = "\n".join(f"| {r['group']} | {r['n']} | {r['without_quote_controls']:+.1%} | {r['final_model']:+.1%} |"
                     for r in V["date_effect_check"])
    nl = "\n".join(f"| {r['reviews']} | {r['n']} | {r['asks_vs_model']:+.1%} | {pct(r['median_abs_pct_error'])} |"
                   for r in V["new_listing_check"])
    return (
        "### 1. Which nights were quoted (fixed with two quote controls)\n"
        "The price in the data is a quote for specific dates. Friday and Saturday nights cost far more, so a model that ignores the dates mixes the listing's value with the calendar. "
        "Training with the share of Fri/Sat nights and the number of nights quoted separates the two; the app then "
        "prices a fixed reference stay. Holding the listing fixed, an all-Fri/Sat stay costs "
        f"**{q['all_weekend_vs_all_weekday_per_night']:+.0%}** per night against an all-weekday stay, and a 7-night quote "
        f"**{q['seven_vs_one_night_quote_per_night']:+.0%}** per night against a 1-night quote.\n\n"
        "| Quoted stay (test set) | Listings | Asks vs model, without controls | Asks vs model, final |\n|---|---|---|---|\n"
        f"{rows}\n\n"
        f"The weekend effect rests on only {N_WEEKENDS} weekends ({WKND_LO:+.0%} to {WKND_HI:+.0%} in the chart above, "
        f"highest on {WKND_TOP}), so the data cannot separate events from ordinary early-summer weekend demand. "
        "That is why the app's main number is the weekday base rate, and the weekend figure is shown as an uncertain "
        "average of this window.\n\n"
        "### 2. Location: little signal left in the errors\n"
        f"The 126 neighbourhoods explain **{pct(lc['residual_share_explained_by_neighbourhood'])}** of the error "
        "left after the model. A listing's error correlates "
        f"**{lc['corr_residual_vs_10_nearest_other_hosts']:.3f}** with the average error of its 10 nearest listings "
        "from other hosts: if location were badly captured, neighbours would share their errors. Remaining systematic "
        f"misses in neighbourhoods with 40+ listings: {worst}. Micro-location (floor, view, street) is not in the data.\n\n"
        "### 3. Seasonality across the year: outside the model's scope\n"
        f"**{pct(V['share_of_quotes_june_july'])}** of the quotes are for June and July check-ins, so the model "
        "cannot learn how prices move over the year. It gives an early-summer base rate. On log price, a season "
        "that moves all listings by the same percentage would only shift the intercept, so the relative comparisons "
        "might carry over, but one snapshot cannot test that.\n\n"
        "### 4. New listings: the model learns from existing hosts\n"
        "| Reviews (test set) | Listings | Asks vs model | Median abs % error |\n|---|---|---|---|\n"
        f"{nl}\n\n"
        "Brand-new listings ask more than the model suggests, heavily reviewed ones less. Listings that book often "
        "seem to settle on lower prices, while newcomers start optimistic. For a new host the suggestion therefore sits "
        "below what other newcomers ask and closer to prices that demonstrably get booked. Reviews are used here for "
        "this check only, never as a model input. The 0-review group is small, so treat this as indicative."
    )


# ----------------------------------------------------------------------------
# Tab 6: business view (the regression counterpart of a threshold slider)
# ----------------------------------------------------------------------------

def business_view(tolerance_pct: float, model_key: str):
    tol = tolerance_pct / 100
    shares = {}
    for k in MODEL_LABELS:
        err = (test[f"pred_{k}_eur"] - test["actual_eur"]) / test["actual_eur"]
        shares[k] = float((err.abs() <= tol).mean())
    fig = go.Figure(go.Bar(
        x=[MODEL_LABELS[k] for k in shares], y=list(shares.values()),
        marker_color=[COLORS[k] for k in shares], text=[pct(v) for v in shares.values()], textposition="outside",
        hovertemplate="%{x}<br>%{y:.1%} within tolerance<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    style(fig, 400, title=f"Share of test listings priced within ±{tolerance_pct:.0f}% of the real price",
          yaxis_title="Share of listings")

    pred, actual = test[f"pred_{model_key}_eur"], test["actual_eur"]
    err = (pred - actual) / actual
    under, over = err < -tol, err > tol
    lost = (actual - pred)[under]
    above = (pred - actual)[over]
    text = (
        f"### {MODEL_LABELS[model_key]} at ±{tolerance_pct:.0f}% tolerance\n"
        f"| | Share of listings | Average gap per night |\n|---|---|---|\n"
        f"| Priced about right | **{(~under & ~over).mean():.1%}** | |\n"
        f"| Underpriced by more than {tolerance_pct:.0f}% | {under.mean():.1%} | "
        f"€{lost.mean() if under.any() else 0:.0f} left on the table per booked night |\n"
        f"| Overpriced by more than {tolerance_pct:.0f}% | {over.mean():.1%} | "
        f"€{above.mean() if over.any() else 0:.0f} above the market, risking empty nights |\n\n"
        f"**Business cost:** across the test listings, following the suggestion would leave on average "
        f"**€{(actual - pred).clip(lower=0).mean():.2f} per listing per booked night** of revenue uncollected "
        "through underpricing. Underpricing costs money on every booking; overpricing costs occupancy, which this "
        "data cannot measure (no booking outcomes). A host who fills nights easily should treat the suggestion as "
        "a floor, one who struggles with occupancy as a ceiling."
    )
    return fig, text


# ----------------------------------------------------------------------------
# Tab 7: price a new listing with the saved models
# ----------------------------------------------------------------------------

PROPERTY_KINDS = {   # kind -> (property_type for an entire place, for a private room)
    "Apartment": ("Entire rental unit", "Private room in rental unit"),
    "Loft": ("Entire loft", "Private room in loft"),
    "Serviced apartment": ("Entire serviced apartment", "Private room in serviced apartment"),
    "House / villa": ("Entire home", "Private room in home"),
    "Hotel, hostel or B&B": ("Entire guesthouse", "Room in hotel"),
}
AMENITY_LABELS = {k: v for k, v in AMENITY_EXAMPLES.items()}
NB_LABELS = {f"{r.neighbourhood} ({r.district})": r for r in neighbourhoods.itertuples()}
# Typical spread of real prices around the prediction, from the test set: half of
# listings sit between these two multiples of the model's suggestion.
_ratio = np.log(test["actual_eur"] / test["pred_sklearn_eur"])
RANGE_LO, RANGE_HI = float(np.exp(_ratio.quantile(0.25))), float(np.exp(_ratio.quantile(0.75)))


def price_listing(nb_label, title, description, room_type, kind, accommodates, bedrooms, bathrooms, bath_shared,
                  min_nights, max_nights, amenities, n_amenities):
    nb = NB_LABELS[nb_label]
    private = room_type == "Private room"
    chosen = [AMENITY_LABELS[k] for k, v in AMENITY_LABELS.items() if v in (amenities or [])]
    padding = [f"Other amenity {i}" for i in range(max(0, int(n_amenities) - len(chosen)))]
    listing = pd.DataFrame([{
        "latitude": nb.latitude, "longitude": nb.longitude, "district": nb.district,
        "room_type": room_type, "property_type": PROPERTY_KINDS[kind][1 if private else 0],
        "accommodates": int(accommodates),
        # same rule as clean_data.fix_values: a private-room guest rents one bedroom
        "bedrooms": 1.0 if private else float(bedrooms), "bedrooms_missing": 0,
        "bathrooms_n": float(bathrooms), "bath_shared": int(bath_shared),
        "minimum_nights": int(min_nights), "maximum_nights": float(max_nights),
        "name": title or "", "description": description or "",
        "amenities": json.dumps(chosen + padding),
    }])
    def predict_all(stay):
        X = engineer_features(stay)                     # the training pipeline's own features
        Z = preprocessor.transform(X)
        Zt = torch.tensor(Z, dtype=torch.float32)
        with torch.no_grad():
            return X, {
                "sklearn": float(sk_model.predict(Z)[0]),
                "torch_manual": float((Zt @ manual_params["w"] + manual_params["b"])[0]),
                "torch_nn": float(nn_model(Zt)[0]),
            }

    # Price a reference stay: the host's minimum stay, on weekday nights and on Fri/Sat nights
    X, weekday = predict_all(reference_stay(listing, weekend=False))
    _, weekend = predict_all(reference_stay(listing, weekend=True))
    base = baseline["table"].get(f"{nb.district}|{int(private)}", baseline["global_median"])
    table = pd.DataFrame({
        "Model": [MODEL_LABELS[k] for k in MODEL_LABELS],
        "Weeknights €/night": [f"€{np.exp(base):.0f}"] + [f"€{np.exp(weekday[k]):.0f}" for k in IMPLEMENTATIONS],
        "Fri/Sat (window average)": ["n/a"] + [f"€{np.exp(weekend[k]):.0f}" for k in IMPLEMENTATIONS],
    })
    p, pw = np.exp(weekday["sklearn"]), np.exp(weekend["sklearn"])
    msg = (f"### Suggested base rate: €{p:.0f} per weeknight\n"
           f"Priced for a stay of {int(min_nights)} night(s), the minimum you set. "
           f"Friday and Saturday nights in this data window averaged €{pw:.0f}. That figure is uncertain: it rests on "
           f"only {N_WEEKENDS} weekends, whose premiums ranged from {WKND_LO:+.0%} to {WKND_HI:+.0%}, highest on "
           f"{WKND_TOP}; the data cannot tell event weekends from ordinary early-summer demand. "
           f"On the test set, half of real prices fell between {RANGE_LO:.2f}× and {RANGE_HI:.2f}× the model's "
           f"prediction, so a realistic weeknight range is **€{p * RANGE_LO:.0f} to €{p * RANGE_HI:.0f}**. "
           f"Location used: {nb.neighbourhood}, {nb.district} "
           f"({X['dist_sol_km'].iloc[0]:.1f} km from Puerta del Sol). "
           "The model is trained on early-summer quotes (June and July 2026), so other seasons need their own check.")
    return msg, table


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------

cl = summary["cleaning_log"]
INTRO = (
    "# Madrid Airbnb: a base nightly rate before you list\n"
    "For a prospective host or investor evaluating a flat that is not listed yet: a starting price, and how each "
    "characteristic relates to it. Airbnb's own Smart Pricing handles day-to-day prices once a listing is live; "
    "this model gives the anchor before that point.\n\n"
    "One model, ridge regression on log(price), trained three ways on the same split grouped by host. "
    f"{cl['raw_rows']:,} raw listings → {cl['final_rows']:,} after cleaning (short-stay quotes of 1 to 7 nights, "
    "entire homes and private rooms, implausible prices removed). The suggestion is a weekday base rate for "
    "early-summer short stays (the data covers June and July 2026 check-ins). "
    f"The test set holds {summary['split_sizes']['test']:,} listings from hosts never seen in training. "
    "Everything here is loaded from saved artifacts; nothing is retrained."
)

with gr.Blocks(title="Madrid Airbnb pricing") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("Model comparison"):
        gr.Dataframe(metrics_table(), interactive=False, wrap=True)
        gr.Markdown(agreement_md())
        choice = gr.Dropdown([("All models", "All models")] + [(v, k) for k, v in MODEL_LABELS.items()],
                             value="All models",
                             label="Model(s) in the prediction vs actual plot")
        pva = gr.Plot(pred_vs_actual("All models"))
        choice.change(pred_vs_actual, choice, pva)
        gr.Plot(error_by_band_plot())
        gr.Markdown(calibration_md())

    with gr.Tab("Training details"):
        gr.Markdown(training_md())
        gr.Dataframe(lr_table(), interactive=False)
        gr.Plot(loss_plot())
        gr.Markdown(dummy_trap_md())
        gr.Plot(coefficient_plot())

    with gr.Tab("Feature evidence"):
        gr.Plot(ablation_plot())
        gr.Markdown(evidence_md())
        gr.Markdown(experiments_md())

    with gr.Tab("Distributions"):
        log_toggle = gr.Checkbox(value=False, label="Show log(price) instead of €")
        tgt = gr.Plot(target_plot(False))
        log_toggle.change(target_plot, log_toggle, tgt)
        feat = gr.Dropdown(FEATURE_CHOICES, value="dist_sol_km", label="Feature")
        d0, r0 = feature_plots("dist_sol_km")
        with gr.Row():
            dist_plot, rel_plot = gr.Plot(d0), gr.Plot(r0)
        feat.change(feature_plots, feat, [dist_plot, rel_plot])

    with gr.Tab("Validity checks"):
        gr.Plot(nights_plot())
        gr.Plot(date_effect_plot())
        gr.Markdown(validity_md())

    with gr.Tab("Business view"):
        with gr.Row():
            tol = gr.Slider(5, 50, value=20, step=1, label="Acceptable pricing error (±%)")
            bmodel = gr.Dropdown([(v, k) for k, v in MODEL_LABELS.items()], value="sklearn",
                                 label="Model for the breakdown")
        f0, t0 = business_view(20, "sklearn")
        bfig, btext = gr.Plot(f0), gr.Markdown(t0)
        tol.change(business_view, [tol, bmodel], [bfig, btext])
        bmodel.change(business_view, [tol, bmodel], [bfig, btext])

    with gr.Tab("Price a listing"):
        gr.Markdown("Describe a flat that is not listed yet. The saved models price a stay of your minimum "
                    "length, on weeknights and on Friday/Saturday nights; nothing is retrained. Change one input "
                    "at a time to see how a characteristic relates to the price (an association, not a guarantee).")
        with gr.Row():
            with gr.Column():
                i_nb = gr.Dropdown(list(NB_LABELS), value="Universidad (Centro)", label="Neighbourhood")
                i_title = gr.Textbox(value="Bright apartment with terrace in Malasaña", label="Listing title")
                i_desc = gr.Textbox(value="Renovated flat with a contemporary design, two minutes from the metro.",
                                    label="Listing description", lines=3)
                i_room = gr.Radio(["Entire home/apt", "Private room"], value="Entire home/apt", label="Room type")
                i_kind = gr.Dropdown(list(PROPERTY_KINDS), value="Apartment", label="Property kind")
                i_acc = gr.Slider(1, 16, value=2, step=1, label="Guests")
                i_bed = gr.Slider(0, 8, value=1, step=1, label="Bedrooms (private rooms count as 1)")
                i_bath = gr.Slider(0.5, 6, value=1, step=0.5, label="Bathrooms")
                i_shared = gr.Checkbox(value=False, label="Bathroom is shared")
                i_min = gr.Slider(1, 7, value=2, step=1, label="Minimum nights")
                i_max = gr.Slider(7, 1125, value=365, step=1, label="Maximum nights")
            with gr.Column():
                i_am = gr.CheckboxGroup(list(AMENITY_LABELS.values()),
                                        value=["Air conditioning", "Washer", "Heating", "Elevator"],
                                        label="Amenities")
                i_nam = gr.Slider(5, 100, value=35, step=1, label="Total number of amenities listed")
                btn = gr.Button("Suggest a price", variant="primary")
                out_md = gr.Markdown()
                out_tbl = gr.Dataframe(interactive=False, wrap=True)
        inputs = [i_nb, i_title, i_desc, i_room, i_kind, i_acc, i_bed, i_bath, i_shared, i_min, i_max, i_am, i_nam]
        btn.click(price_listing, inputs, [out_md, out_tbl])


if __name__ == "__main__":
    # No browser tab in CI (GitHub Actions sets CI=true); locally one opens.
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=not os.environ.get("CI"))
