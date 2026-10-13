"""
How the features were chosen. NOT part of main.py train.

    uv run python experiments/feature_experiments.py      (about 3 minutes)

Writes experiments/results.md: the rule, every candidate that was tried, the
decision for each, and four diagnostics on how much signal is left unused.

THE PROCEDURE
1. Starting model: features chosen by reasoning before any testing (location,
   size, room and property type, 16 amenity concepts, minimum nights), plus the
   two quote controls that fix the date effect in the target (REPORT.md,
   "Validity").
2. Each candidate is added to the starting model ON ITS OWN and kept only if it
   passes all four parts of THE RULE:
     a. Legitimate: known before the listing goes live, and does not identify
        a particular host or operator.
     b. Lowers the error (RMSE of log price) by at least 0.001 on average over a
        5-round cross-validation grouped by host.
     c. Lowers it in at least 4 of the 5 rounds.
     d. Also lowers it on the validation split (trained on the training split),
        a second, independent check.
3. Final model = starting model + every candidate that passed, checked together.

All numbers use training + validation listings only. The test set is never
touched, so it stays an honest final check.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import features as F  # noqa: E402
import pipeline as T  # noqa: E402

ALPHA = 0.1                 # same ridge strength as the final model
MIN_GAIN = 0.001            # rule part b
MIN_ROUNDS = 4              # rule part c
OUT = ROOT / "experiments" / "results.md"

START_NUM = ["dist_sol_km", "log_dist_sol", "log_accommodates", "bedrooms", "bathrooms_n",
             "log_min_nights", "log_n_amenities"] + F.QUOTE_CONTROLS
START_BIN = (["is_private_room", "bedrooms_missing", "bath_shared"] + list(F.PROPERTY_PATTERNS)
             + [a for a in F.AMENITY_PATTERNS if a != "am_breakfast"])


# ----------------------------------------------------------------------------
# Data: the clean file plus a few raw columns only needed for candidates
# ----------------------------------------------------------------------------

def load():
    clean = pd.read_csv(ROOT / "data/madrid_listings_clean.csv.gz")
    raw = (pd.read_csv(ROOT / "data/raw/madrid_listings_raw.csv.gz", low_memory=False)
             .set_index("id").loc[clean["id"]].reset_index())
    for c in ["license", "last_scraped", "beds"]:
        clean[c] = raw[c].to_numpy()
    train_idx, val_idx, _ = T.grouped_split(clean)            # same split as train.py
    return clean, train_idx, val_idx


def candidate_columns(df):
    """Every candidate feature, computed row by row (nothing fitted here)."""
    X = F.engineer_features(df)
    text = df["name"].fillna("").str.lower() + " . " + df["description"].fillna("").str.lower()
    am = df["amenities"].map(json.loads).map(lambda L: " | ".join(a.lower() for a in L))
    lic = df["license"].fillna("")
    checkin = pd.to_datetime(df["price_quote_checkin_date"])
    X["beds"] = df["beds"].clip(upper=10)
    X["log_desc_len"] = np.log1p(df["description"].fillna("").str.len())
    X["dist_sq"] = X["dist_sol_km"] ** 2
    X["private_x_dist"] = X["is_private_room"] * X["dist_sol_km"]
    X["lic_tourist_number"] = lic.str.contains(r"ESFCNT|VT-|VT -|ESHFTU", regex=True).astype(int)
    X["lic_seasonal_exempt"] = lic.str.contains("seasonal", case=False).astype(int)
    X["lic_missing"] = (lic == "").astype(int)
    X["seasonal_broad"] = text.str.contains(r"corta duraci|temporada|temporary|seasonal|lease|contrato|agreement|mid-term|media estancia", regex=True).astype(int)
    X["lease_required"] = text.str.contains(r"lease agreement|rental agreement|rental contract|contrato de arrendamiento|contrato de alquiler|requires (?:the )?sign", regex=True).astype(int)
    X["basic_words"] = text.str.contains(r"simplicity|sencill|basic|básic|budget|económic|cheap", regex=True).astype(int)
    X["am_safe"] = am.str.contains(r"\bsafe\b", regex=True).astype(int)
    X["am_building_staff"] = am.str.contains(r"building staff|doorman|concierge", regex=True).astype(int)
    X["am_private_living_room"] = am.str.contains("private living room").astype(int)
    X["am_family_kit"] = am.str.contains(r"high chair|crib|babysitter|children", regex=True).astype(int)
    X["am_room_lock"] = am.str.contains("lock on bedroom door").astype(int)
    X["operator_brand_words"] = text.str.contains(r"\bwaou\b|\bdobo\b|corta duraci", regex=True).astype(int)
    X["log_lead_days"] = np.log1p((checkin - pd.to_datetime(df["last_scraped"])).dt.days.clip(lower=0))
    X["checkin_month"] = checkin.dt.month
    X["neighbourhood"] = df["neighbourhood"]
    return X


# ----------------------------------------------------------------------------
# Features that must be learned inside each round (they use other listings' prices)
# ----------------------------------------------------------------------------

def nearby_prices(fit_df, fit_y, query_df, k, exclude_same_host):
    """Median log price of the k nearest listings of the same room type."""
    out = np.full(len(query_df), np.nan)
    for rt in ["Entire home/apt", "Private room"]:
        fm, qm = (fit_df["room_type"] == rt).to_numpy(), (query_df["room_type"] == rt).to_numpy()
        tree = BallTree(np.radians(fit_df.loc[fm, ["latitude", "longitude"]].to_numpy()), metric="haversine")
        _, ind = tree.query(np.radians(query_df.loc[qm, ["latitude", "longitude"]].to_numpy()),
                            k=min(k + (60 if exclude_same_host else 0), fm.sum()))
        fh, fy, qh = fit_df.loc[fm, "host_id"].to_numpy(), fit_y[fm], query_df.loc[qm, "host_id"].to_numpy()
        vals = []
        for row, h in zip(ind, qh):
            if exclude_same_host:
                row = row[fh[row] != h]
            vals.append(np.median(fy[row[:k]]))
        out[qm] = vals
    return out


def neighbourhood_level(fit_df, fit_y, query_df, m=20):
    """Neighbourhood mean log price, pulled toward its district mean (weight m)."""
    t = pd.DataFrame({"n": fit_df["neighbourhood"].values, "d": fit_df["district"].values, "y": fit_y})
    dmean = t.groupby("d")["y"].mean()
    g = t.groupby("n").agg(mean=("y", "mean"), count=("y", "size"), d=("d", "first"))
    level = (g["count"] * g["mean"] + m * dmean.loc[g["d"]].values) / (g["count"] + m)
    return (query_df["neighbourhood"].map(level).fillna(query_df["district"].map(dmean))
            .fillna(fit_y.mean()).to_numpy())


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

class Scorer:
    def __init__(self, df, X, y, train_idx, val_idx):
        self.tv = np.concatenate([train_idx, val_idx])
        self.df_tv = df.iloc[self.tv].reset_index(drop=True)
        self.X_tv, self.y_tv = X.iloc[self.tv].reset_index(drop=True), y[self.tv]
        self.folds = list(GroupKFold(n_splits=5).split(self.X_tv, groups=self.df_tv["host_id"]))
        self.df, self.X, self.y, self.tr, self.va = df, X, y, train_idx, val_idx

    @staticmethod
    def _pre(num, binary, extra_cats):
        pre = T.build_preprocessor(num, binary)
        parts = list(pre.transformers)
        for c in extra_cats:   # extra categorical candidates: one-hot, rare levels pooled
            parts.append((c, Pipeline([("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                                           min_frequency=10)),
                                       ("s", StandardScaler())]), [c]))
        return ColumnTransformer(parts)

    def _fit_predict(self, df_a, X_a, y_a, df_b, X_b, num, binary, cats, fitted):
        X_a, X_b = X_a.copy(), X_b.copy()
        if fitted:
            name, f_train, f_query = fitted
            X_a[name] = f_train(df_a, y_a)
            X_b[name] = f_query(df_a, y_a, df_b)
            num = num + [name]
        pre = self._pre(num, binary, cats).fit(X_a)
        return T.fit_sklearn(pre.transform(X_a), y_a, ALPHA).predict(pre.transform(X_b))

    def score(self, num, binary, cats=(), fitted=None):
        """Per-round RMSE (5 rounds) and the RMSE on the validation split."""
        rounds = []
        for a, b in self.folds:
            da, db = self.df_tv.iloc[a].reset_index(drop=True), self.df_tv.iloc[b].reset_index(drop=True)
            pred = self._fit_predict(da, self.X_tv.iloc[a], self.y_tv[a], db, self.X_tv.iloc[b],
                                     list(num), list(binary), list(cats), fitted)
            rounds.append(np.sqrt(np.mean((self.y_tv[b] - pred) ** 2)))
        da, db = self.df.iloc[self.tr].reset_index(drop=True), self.df.iloc[self.va].reset_index(drop=True)
        pred = self._fit_predict(da, self.X.iloc[self.tr], self.y[self.tr], db, self.X.iloc[self.va],
                                 list(num), list(binary), list(cats), fitted)
        return np.array(rounds), float(np.sqrt(np.mean((self.y[self.va] - pred) ** 2)))


# ----------------------------------------------------------------------------
# The candidates, in plain words
# ----------------------------------------------------------------------------
# name: (what it measures, extra numeric, extra binary, extra categorical, fitted feature)

def candidates():
    nb_level = ("neighbourhood_level", lambda d, y: neighbourhood_level(d, y, d), neighbourhood_level)
    near_ok = ("nearby_price", lambda d, y: nearby_prices(d, y, d, 25, True),
               lambda d, y, q: nearby_prices(d, y, q, 25, False))
    near_leaky = ("nearby_price", lambda d, y: nearby_prices(d, y, d, 26, False),
                  lambda d, y, q: nearby_prices(d, y, q, 25, False))
    return {
        "Title words (6 flags)": ("Title mentions luxury/design, penthouse/ático, terrace, view, studio, or cozy/small", [], list(F.TITLE_KEYWORDS), [], None),
        "Upscale words in title or description": ("Luxury, design, stylish, contemporary, prestigious, elegant, exclusive, premium, boutique", [], ["upscale_wording"], [], None),
        "Private room × guests": ("Lets the number of guests count differently for a private room than for a whole flat", ["private_x_log_accommodates"], [], [], None),
        "Maximum nights": ("Longest stay the host allows (log)", ["log_max_nights"], [], [], None),
        "Amenity: breakfast": ("Breakfast is listed as an amenity", [], ["am_breakfast"], [], None),
        "Beds": ("Number of beds", ["beds"], [], [], None),
        "Description length": ("Number of characters in the description (log)", ["log_desc_len"], [], [], None),
        "Budget words": ("Description mentions simple, basic, budget, cheap", [], ["basic_words"], [], None),
        "Seasonal-rental words (broad)": ("Description mentions temporary/seasonal rental, lease, contract", [], ["seasonal_broad"], [], None),
        "Lease agreement required (narrow)": ("Description says a lease or rental contract must be signed (a narrower second definition, tried once after the broad one failed)", [], ["lease_required"], [], None),
        "Amenity: safe": ("A safe is listed", [], ["am_safe"], [], None),
        "Amenity: building staff": ("Building staff or doorman is listed", [], ["am_building_staff"], [], None),
        "Amenity: private living room": ("A private living room is listed", [], ["am_private_living_room"], [], None),
        "Amenity: family equipment": ("High chair, crib, babysitter tips or children's items", [], ["am_family_kit"], [], None),
        "Amenity: lock on bedroom door": ("Typical of rooms rented inside a shared flat", [], ["am_room_lock"], [], None),
        "Tourist licence status": ("Registered tourist licence, seasonal-rental exemption, or none", [], ["lic_tourist_number", "lic_seasonal_exempt", "lic_missing"], [], None),
        "Distance to Sol, squared": ("Lets the distance effect curve more", ["dist_sq"], [], [], None),
        "Private room × distance": ("Lets distance count differently for private rooms", ["private_x_dist"], [], [], None),
        "All 128 neighbourhoods": ("Each of the 128 neighbourhoods as its own category, on top of the 21 districts", [], [], ["neighbourhood"], None),
        "Neighbourhood price level": ("Average price of the neighbourhood, pulled toward its district average", [], [], [], nb_level),
        "Prices of the 25 nearest listings": ("Median price of the 25 nearest similar listings from other hosts", [], [], [], near_ok),
    }


PRINCIPLE = {
    # name: (why it is excluded whatever it scores, extra numeric, binary, categorical, fitted)
    "Lead time (days until the quoted check-in)": ("Inside Airbnb quotes the first FREE stay, so a late date means a booked-up listing: demand information nobody has before listing", ["log_lead_days"], [], [], None),
    "Check-in month": ("94% of quotes are June/July check-ins, so other months cannot be learned", [], [], ["checkin_month"], None),
    "Operator brand words (waou, dobo, corta duración)": ("Template wording of about 10 professional operators: identifies the host", [], ["operator_brand_words"], [], None),
    "Prices of the 25 nearest listings, own host included": ("Leaks the answer: the listing's own price and its host's other prices are in the feature", [], [], [], None),
}


# ----------------------------------------------------------------------------
# Diagnostics: how much signal is left that the final model does not use?
# ----------------------------------------------------------------------------

def diagnostics(s: Scorer, final_num, final_bin, final_rounds):
    out = []
    text = (s.df_tv["name"].fillna("") + " . " + s.df_tv["description"].fillna("")).str.lower()

    rm = []
    for a, b in s.folds:
        pre = T.build_preprocessor(final_num, final_bin).fit(s.X_tv.iloc[a])
        v = TfidfVectorizer(min_df=20, ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
        Za = sp.hstack([sp.csr_matrix(pre.transform(s.X_tv.iloc[a])), v.fit_transform(text.iloc[a])]).tocsr()
        Zb = sp.hstack([sp.csr_matrix(pre.transform(s.X_tv.iloc[b])), v.transform(text.iloc[b])]).tocsr()
        pr = Ridge(alpha=10.0).fit(Za, s.y_tv[a]).predict(Zb)
        rm.append(np.sqrt(np.mean((s.y_tv[b] - pr) ** 2)))
    out.append(("Every word and word pair in title + description (3,000 columns)", np.mean(rm) - final_rounds.mean(),
                "Real extra signal, but 3,000 unnamed columns make the model unreadable, and part of it is operator template wording"))

    am = s.df_tv["amenities"].map(json.loads).map(lambda L: {a.strip().lower() for a in L})
    counts = pd.Series([a for x in am for a in x]).value_counts()
    common = counts[(counts >= 0.01 * len(am)) & (counts <= 0.99 * len(am))].index
    A = pd.DataFrame({f"amx_{i}": am.map(lambda x, a=a: int(a in x)) for i, a in enumerate(common)})
    XA = pd.concat([s.X_tv, A], axis=1)
    rm = []
    for a, b in s.folds:
        pre = T.build_preprocessor(final_num, final_bin + list(A.columns)).fit(XA.iloc[a])
        pr = T.fit_sklearn(pre.transform(XA.iloc[a]), s.y_tv[a], ALPHA).predict(pre.transform(XA.iloc[b]))
        rm.append(np.sqrt(np.mean((s.y_tv[b] - pr) ** 2)))
    out.append((f"Every amenity listed by at least 1% of listings ({len(common)} columns)", np.mean(rm) - final_rounds.mean(),
                "The individual amenities that pass the rule were tested above; the rest is many tiny effects"))

    rm = []
    for a, b in s.folds:
        pre = T.build_preprocessor(final_num, final_bin).fit(s.X_tv.iloc[a])
        m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, min_samples_leaf=30, random_state=0)
        pr = m.fit(pre.transform(s.X_tv.iloc[a]), s.y_tv[a]).predict(pre.transform(s.X_tv.iloc[b]))
        rm.append(np.sqrt(np.mean((s.y_tv[b] - pr) ** 2)))
    out.append(("Gradient-boosted trees on the final features", np.mean(rm) - final_rounds.mean(),
                "How much curves and interactions could add; the assignment requires a linear model"))
    return out


def pre_discount_check(clean, s: Scorer, final_num, final_bin, final_rounds):
    raw = (pd.read_csv(ROOT / "data/raw/madrid_listings_raw.csv.gz", low_memory=False, usecols=["id", "price_quote_raw"])
             .set_index("id").loc[s.df_tv["id"]])
    q = raw["price_quote_raw"].map(lambda t: json.loads(t)["quote"])
    sub = q.map(lambda d: d.get("nightly_subtotal"))
    pre_rate = np.where(sub.notna(), pd.to_numeric(sub, errors="coerce") / s.df_tv["quote_nights"].to_numpy(),
                        s.df_tv["price_eur"].to_numpy())
    discounted = pre_rate > s.df_tv["price_eur"].to_numpy() * 1.001
    y_pre = np.log(pre_rate)
    rm = []
    for a, b in s.folds:
        pre = T.build_preprocessor(final_num, final_bin).fit(s.X_tv.iloc[a])
        pr = T.fit_sklearn(pre.transform(s.X_tv.iloc[a]), y_pre[a], ALPHA).predict(pre.transform(s.X_tv.iloc[b]))
        rm.append(np.sqrt(np.mean((y_pre[b] - pr) ** 2)))
    new = s.df_tv["number_of_reviews"].to_numpy() == 0
    return {"share_discounted": float(discounted.mean()), "share_discounted_new": float(discounted[new].mean()),
            "share_discounted_other": float(discounted[~new].mean()),
            "rmse_change": float(np.mean(rm) - final_rounds.mean())}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def decide(d_rounds, d_val):
    """Name every part of the rule that fails; show the exact gain when it is close to the threshold."""
    gain, better = -d_rounds.mean(), int((d_rounds < 0).sum())
    fails = []
    if gain < MIN_GAIN:
        fails.append("makes the error worse" if gain <= 0 else
                     f"gain {gain:.5f} is below {MIN_GAIN}" if gain > 0.0005 else "gain too small")
    if better < MIN_ROUNDS:
        fails.append(f"better in only {better} of 5 rounds")
    if d_val >= 0:
        fails.append("worse on the validation split")
    if fails:
        return "Rejected: " + "; ".join(fails)
    return f"Accepted (narrowly: gain {gain:.5f})" if gain < 1.2 * MIN_GAIN else "Accepted"


def main():
    clean, train_idx, val_idx = load()
    X = candidate_columns(clean)
    y = np.log(clean["price_eur"].to_numpy())
    s = Scorer(clean, X, y, train_idx, val_idx)

    start_rounds, start_val = s.score(START_NUM, START_BIN)
    no_qc_rounds, no_qc_val = s.score([c for c in START_NUM if c not in F.QUOTE_CONTROLS], START_BIN)
    print(f"Starting model: error {start_rounds.mean():.4f} (validation split {start_val:.4f})")

    rows, accepted_num, accepted_bin = [], [], []
    for name, (what, num, binary, cats, fitted) in candidates().items():
        r, v = s.score(START_NUM + num, START_BIN + binary, cats, fitted)
        d, dv = r - start_rounds, v - start_val
        decision = decide(d, dv)
        if decision.startswith("Accepted"):
            accepted_num += num
            accepted_bin += binary
        rows.append({"Candidate": name, "What it measures": what, "Error change": d.mean(),
                     "Better in": f"{(d < 0).sum()} of 5", "Validation split": dv, "Decision": decision})
        print(f"  {name:45s} {d.mean():+.4f}  {(d < 0).sum()}/5  val {dv:+.4f}  {decision}")

    principle = []
    for name, (why, num, binary, cats, fitted) in PRINCIPLE.items():
        if name.startswith("Prices of the 25 nearest"):
            fitted = ("nearby_price", lambda d, y: nearby_prices(d, y, d, 26, False),
                      lambda d, y, q: nearby_prices(d, y, q, 25, False))
        r, v = s.score(START_NUM + num, START_BIN + binary, cats, fitted)
        principle.append({"Candidate": name, "Error change": (r - start_rounds).mean(),
                          "Better in": f"{(r < start_rounds).sum()} of 5", "Why excluded": why})
        print(f"  [principle] {name:45s} {(r - start_rounds).mean():+.4f}")

    final_num, final_bin = START_NUM + accepted_num, START_BIN + accepted_bin
    final_rounds, final_val = s.score(final_num, final_bin)
    assert sorted(final_num) == sorted(F.NUMERIC_FEATURES) and sorted(final_bin) == sorted(F.BINARY_FEATURES), \
        "features.py does not match the candidates accepted by the rule"
    diag = diagnostics(s, final_num, final_bin, final_rounds)
    disc = pre_discount_check(clean, s, final_num, final_bin, final_rounds)

    # ---------------- write results.md ----------------
    def table(rs, cols, fmt):
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in rs:
            lines.append("| " + " | ".join(fmt(c, r[c]) for c in cols) + " |")
        return lines

    def fmt(col, val):
        if col == "Error change":
            return f"{val:+.4f}"
        if col == "Validation split":
            return "improves" if val < 0 else "worse or equal"
        return str(val)

    acc = [r for r in rows if r["Decision"].startswith("Accepted")]
    rej = [r for r in rows if not r["Decision"].startswith("Accepted")]
    L = [
        "# How the features were chosen",
        "",
        "Produced by `experiments/feature_experiments.py`. All numbers come from training + validation listings "
        f"({len(s.tv):,}); the test set is never used for these decisions.",
        "",
        "**Error** = RMSE of log price: the typical size of a prediction mistake, with big mistakes counting extra. "
        "Lower is better; 0.001 is a small change, 0.01 a large one. **Error change** is measured against the "
        "starting model, averaged over 5 rounds in which 4/5 of the hosts train the model and the other 1/5 are "
        "predicted (no host ever appears on both sides).",
        "",
        "## The procedure",
        "",
        "1. **Starting model**, chosen by reasoning before any test: location (distance to Puerta del Sol, district), "
        "size (guests, bedrooms, bathrooms), room and property type, 16 amenity concepts, minimum nights, plus the two "
        "quote controls that remove the date effect from the target "
        f"(error {start_rounds.mean():.4f}; without the quote controls {no_qc_rounds.mean():.4f}).",
        "2. **Each candidate is added on its own** and kept only if it passes all four parts of the rule:",
        "   - **a. Legitimate:** known before the listing goes live, and does not identify a particular host.",
        f"   - **b. Big enough:** lowers the error by at least {MIN_GAIN} on average.",
        f"   - **c. Consistent:** lowers it in at least {MIN_ROUNDS} of the 5 rounds.",
        "   - **d. Confirmed:** also lowers it on the validation split (model trained on the training split).",
        "3. **Final model** = starting model + every accepted candidate, checked together.",
        "",
        "**Where the candidates came from.** Most were reasoned out in advance (title words, beds, the two "
        "interactions, distance squared, description length, licence, neighbourhoods, nearby prices) or found by going "
        "through the unused columns of the raw file (maximum nights). The others (upscale and budget words, the "
        "seasonal-rental and lease wording, and the six single amenities) came from inspecting which words and "
        "amenities were associated with the model's errors on the **training split**. Because that inspection used "
        "the training split, check d scores every candidate on the validation split, which played no part in it.",
        "",
        "## Accepted",
        "",
        *table(acc, ["Candidate", "What it measures", "Error change", "Better in", "Validation split", "Decision"], fmt),
        "",
        f"**All accepted together:** error {start_rounds.mean():.4f} → {final_rounds.mean():.4f} "
        f"({(final_rounds - start_rounds).mean():+.4f}, better in {(final_rounds < start_rounds).sum()} of 5 rounds; "
        f"validation split {'improves' if final_val < start_val else 'does not improve'}).",
        "",
        "## Rejected: did not pass the rule",
        "",
        *table(rej, ["Candidate", "What it measures", "Error change", "Better in", "Decision"], fmt),
        "",
        "Some words looked strong when inspected, notably \"corta duración\" (short-term rental), which went with "
        "prices about 30% above the model. It turned out to appear in 134 listings from only 9 hosts: template "
        "wording of a few operators. Such signals do not carry over to hosts the model has not seen, which is exactly "
        "what the host-grouped rounds test, so the seasonal-rental candidates fail.",
        "",
        "## Rejected on principle (rule part a), whatever they score",
        "",
        *table(principle, ["Candidate", "Error change", "Better in", "Why excluded"], fmt),
        "",
        "## How much signal is left unused? (diagnostics, compared with the final model)",
        "",
        "| What was tried | Error change vs final model | What it means |",
        "|---|---|---|",
        *[f"| {n} | {d:+.4f} | {m} |" for n, d, m in diag],
        f"| Pre-discount price as the target | {disc['rmse_change']:+.4f} | {disc['share_discounted']:.1%} of quotes carry a "
        f"discount (brand-new listings {disc['share_discounted_new']:.0%}, others {disc['share_discounted_other']:.0%}); "
        "the price after discount predicts at least as well, so the target stays as it is |",
        "",
    ]
    OUT.write_text("\n".join(L))
    print(f"\nFinal: {final_rounds.mean():.4f}  (start {start_rounds.mean():.4f})")
    for n, d, m in diag:
        print(f"  diagnostic {n}: {d:+.4f}")
    print(f"  pre-discount target: {disc}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
