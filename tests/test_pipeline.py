"""
Checks for the logic the report relies on.  Run:  uv run pytest -q

Fast (a few seconds): the cleaning and feature rules on hand-made rows, the
host-grouped split on the real data, and the claim that the three
implementations solve the same problem, on small synthetic data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clean_data as C  # noqa: E402
import features as F  # noqa: E402
import train as T  # noqa: E402


def raw_row(**overrides):
    """A minimal raw Inside Airbnb row that passes every filter."""
    row = {
        "id": 1, "host_id": 10, "price": "$120.00",
        "price_quote_checkin_date": "2026-07-01", "price_quote_checkout_date": "2026-07-03",
        "room_type": "Entire home/apt", "property_type": "Entire rental unit",
        "accommodates": 2, "bedrooms": 1.0, "bathrooms_text": "1 bath",
        "neighbourhood_group_cleansed": "Centro", "neighbourhood_cleansed": "Sol",
        "latitude": 40.4169, "longitude": -3.7035, "minimum_nights": 2,
        "name": "Flat", "description": "A flat.", "amenities": json.dumps(["Wifi"]), "number_of_reviews": 0,
        "maximum_nights": 365,
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------- cleaning

def test_price_and_bathroom_parsing():
    df = C.parse_types(pd.DataFrame([
        raw_row(price="$1,234.00", bathrooms_text="1.5 shared baths"),
        raw_row(bathrooms_text="Shared half-bath"),
        raw_row(bathrooms_text=None),
    ]))
    assert df["price_eur"].tolist() == [1234.0, 120.0, 120.0]
    assert df["bathrooms_n"].iloc[0] == 1.5 and df["bath_shared"].iloc[0] == 1
    assert df["bathrooms_n"].iloc[1] == 0.5 and df["bath_shared"].iloc[1] == 1
    assert np.isnan(df["bathrooms_n"].iloc[2])
    assert df["quote_nights"].iloc[0] == 2


def test_filters_remove_out_of_scope_rows():
    raw = pd.DataFrame([
        raw_row(id=1),
        raw_row(id=2, price=None),                                         # no price
        raw_row(id=3, price_quote_checkout_date="2026-07-31"),             # 30-night quote
        raw_row(id=4, room_type="Hotel room"),                             # out of scope
        raw_row(id=5, price="$15,000.00"),                                 # placeholder price
        raw_row(id=6, price="$10.00"),                                     # below EUR 20
    ])
    out, log = C.clean(raw)
    assert out["id"].tolist() == [1]
    assert log["dropped_no_price"] == 1
    assert log["dropped_long_stay_quote"] == 1
    assert log["dropped_room_type_out_of_scope"] == 1
    assert log["dropped_implausible_price"] == 2


def test_value_fixes():
    raw = pd.DataFrame([
        raw_row(id=1, room_type="Private room", bedrooms=None),           # blank private room
        raw_row(id=2, room_type="Private room", bedrooms=12.0),           # whole-flat count
        raw_row(id=3, bathrooms_text="0 baths"),                          # entire home, 0 baths
        raw_row(id=4, bedrooms=None),                                     # entire home, blank
    ])
    out, _ = C.clean(raw)
    out = out.set_index("id")
    assert out.loc[1, "bedrooms"] == 1 and out.loc[1, "bedrooms_missing"] == 1   # flag taken BEFORE the fill
    assert out.loc[2, "bedrooms"] == 1 and out.loc[2, "bedrooms_missing"] == 0
    assert np.isnan(out.loc[3, "bathrooms_n"])                                   # unknown, not zero
    assert np.isnan(out.loc[4, "bedrooms"])                                      # left for the train-split imputer


# --------------------------------------------------------------------------- features

@pytest.mark.parametrize("concept,example", F.AMENITY_EXAMPLES.items())
def test_each_amenity_example_matches_only_its_own_concept(concept, example):
    flags = F.parse_amenities(pd.Series([json.dumps([example])])).iloc[0]
    assert flags[concept] == 1
    assert flags.drop([concept, "log_n_amenities"]).sum() == 0


def test_amenity_false_friends():
    flags = F.parse_amenities(pd.Series([json.dumps(["Hair dryer", "Pool table", "Gym nearby"])])).iloc[0]
    assert flags["am_dryer"] == 0 and flags["am_pool"] == 0 and flags["am_gym"] == 0


def test_title_keywords_spanish_and_english():
    kw = F.parse_title(pd.Series(["Ático de lujo con terraza", "Cozy studio", "Piso"]))
    assert kw.loc[0, ["kw_penthouse", "kw_luxury", "kw_terrace"]].tolist() == [1, 1, 1]
    assert kw.loc[1, ["kw_cozy", "kw_studio"]].tolist() == [1, 1]
    assert kw.loc[2].sum() == 0


def test_distance_to_sol():
    df = pd.DataFrame([raw_row(latitude=40.4169, longitude=-3.7035)])
    clean, _ = C.clean(df)
    assert F.engineer_features(clean)["dist_sol_km"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    # Puerta del Sol to Retiro's Puerta de Alcalá is about 1.3 km
    assert F.haversine_km(40.4200, -3.6887, *F.PUERTA_DEL_SOL) == pytest.approx(1.3, abs=0.15)


# --------------------------------------------------------------------------- split and models

def test_grouped_split_keeps_every_host_on_one_side():
    df = pd.read_csv(ROOT / "data/madrid_listings_clean.csv.gz", usecols=["host_id"])
    tr, va, te = T.grouped_split(df)
    hosts = [set(df["host_id"].iloc[i]) for i in (tr, va, te)]
    assert not (hosts[0] & hosts[1]) and not (hosts[0] & hosts[2]) and not (hosts[1] & hosts[2])
    assert len(tr) + len(va) + len(te) == len(df)


def test_three_implementations_solve_the_same_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = X @ np.array([0.5, -1.0, 0.0, 2.0, 0.3, -0.7]) + 4.0 + rng.normal(scale=0.3, size=400)
    alpha = 50.0                       # strong enough that a wrong penalty translation would show
    sk = T.fit_sklearn(X, y, alpha)
    manual, _ = T.fit_manual_torch(X, y, alpha, lr=0.1, epochs=3000)
    nn_model, _ = T.fit_standard_torch(X, y, alpha, lr=0.1, epochs=3000)
    np.testing.assert_allclose(manual["w"].numpy(), sk.coef_, atol=1e-4)
    np.testing.assert_allclose(nn_model.linear.weight.detach().numpy().ravel(), sk.coef_, atol=1e-4)
    assert manual["b"].item() == pytest.approx(sk.intercept_, abs=1e-4)
    assert nn_model.linear.bias.item() == pytest.approx(sk.intercept_, abs=1e-4)


def test_wrong_weight_decay_would_be_caught():
    """Guards the 2*alpha/n translation: using alpha/n instead gives different weights."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.2, size=200)
    alpha = 100.0
    sk = T.fit_sklearn(X, y, alpha)
    wrong, _ = T.fit_standard_torch(X, y, alpha / 2, lr=0.1, epochs=3000)   # equals wd = alpha/n
    assert np.abs(wrong.linear.weight.detach().numpy().ravel() - sk.coef_).max() > 1e-2


# --------------------------------------------------------------------------- quote controls

@pytest.mark.parametrize("checkin,checkout,expected", [
    ("2026-06-26", "2026-06-28", 1.0),    # Fri + Sat nights
    ("2026-06-21", "2026-06-23", 0.0),    # Sun + Mon nights
    ("2026-06-25", "2026-06-27", 0.5),    # Thu + Fri nights
    ("2026-06-22", "2026-06-29", 2 / 7),  # a full week: Fri and Sat out of 7 nights
])
def test_weekend_share_counts_friday_and_saturday_nights(checkin, checkout, expected):
    df = C.parse_types(pd.DataFrame([raw_row(price_quote_checkin_date=checkin, price_quote_checkout_date=checkout)]))
    assert df["weekend_share"].iloc[0] == pytest.approx(expected)


def test_reference_stay_prices_the_hosts_minimum_stay():
    clean, _ = C.clean(pd.DataFrame([raw_row(minimum_nights=3,
                                             price_quote_checkin_date="2026-06-26",
                                             price_quote_checkout_date="2026-06-27")]))
    weekday = F.engineer_features(F.reference_stay(clean, weekend=False)).iloc[0]
    weekend = F.engineer_features(F.reference_stay(clean, weekend=True)).iloc[0]
    assert weekday["weekend_share"] == 0 and weekend["weekend_share"] == 1
    assert weekday["log_quote_nights"] == pytest.approx(np.log(3))
    # everything that describes the listing itself is unchanged
    listing_cols = [c for c in weekday.index if c not in F.QUOTE_CONTROLS]
    assert weekday[listing_cols].equals(weekend[listing_cols])


def test_reviews_are_never_a_model_input():
    assert not any("review" in c for c in F.NUMERIC_FEATURES + F.BINARY_FEATURES + F.CATEGORICAL_FEATURES)


def test_upscale_wording_reads_title_and_description():
    flags = F.has_upscale_wording(pd.Series(["Piso", "Piso de lujo", "Flat"]),
                                  pd.Series(["Sencillo", "", "Contemporary design, stylish"]))
    assert flags.tolist() == [0, 1, 1]


def test_beds_was_dropped_by_the_feature_rule():
    """Beds failed the rule (redundant with guests); make sure it does not creep back in."""
    assert "beds" not in F.NUMERIC_FEATURES + F.BINARY_FEATURES
