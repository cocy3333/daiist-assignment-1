"""
Data cleaning: raw Inside Airbnb export  ->  clean dataset used for training.

This is the first stage of `uv run python main.py train`: train.py calls
run_cleaning() before anything else, so one command goes from the raw export
to trained models. It can also be run on its own to inspect the result:
    uv run python clean_data.py

Input : data/raw/madrid_listings_raw.csv.gz   (untouched export, 22,708 rows x 90 columns)
Output: data/madrid_listings_clean.csv.gz     (the dataset train.py reads)
        data/cleaning_log.json                (how many rows each rule removed)

What belongs here, and what doesn't
-----------------------------------
Cleaning = deciding which rows are in scope and making each recorded value
correct. Every rule looks at ONE row at a time, so nothing learned from other
rows (in particular, nothing from future test rows) leaks in.

Missing values are deliberately NOT filled here. Filling them with, say, the
dataset median would use test rows to compute that median. Imputation happens
in train.py's preprocessor, fitted on the training split only.

Feature engineering (distance to Sol, amenity flags, logs) lives in features.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/madrid_listings_raw.csv.gz")
CLEAN_PATH = Path("data/madrid_listings_clean.csv.gz")
LOG_PATH = Path("data/cleaning_log.json")

# Business rules (see REPORT.md, "Business / real-life framing")
MAX_QUOTE_NIGHTS = 7                 # short stays only; longer quotes carry monthly discounts
ROOM_TYPES_IN_SCOPE = ["Entire home/apt", "Private room"]
MIN_PRICE_EUR = 20.0                 # below: placeholder or error prices
MAX_PRICE_PER_GUEST_EUR = 250.0      # above: blocked-calendar / placeholder prices

# Columns kept in the clean file. Deliberately excluded as model inputs:
# reviews, occupancy, availability and host history, which a NEW host does not
# have yet. number_of_reviews is kept for ONE evaluation check only (does the
# model price brand-new listings differently?); features.py never reads it.
OUTPUT_COLUMNS = [
    "id", "host_id", "district", "neighbourhood", "latitude", "longitude",
    "property_type", "room_type", "accommodates",
    "bedrooms", "bedrooms_missing", "bathrooms_n", "bath_shared",
    "name", "description", "amenities", "minimum_nights", "maximum_nights",
    "price_eur", "quote_nights", "weekend_share", "price_quote_checkin_date",
    "number_of_reviews",
]


# ----------------------------------------------------------------------------
# Step 1: fix types and parse text fields
# ----------------------------------------------------------------------------

def parse_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # "$1,234.00" -> 1234.0. Inside Airbnb writes "$" even though Madrid prices are EUR.
    out["price_eur"] = pd.to_numeric(
        out["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")

    # The price is a quote for a specific stay: how many nights was it for?
    checkin = pd.to_datetime(out["price_quote_checkin_date"], errors="coerce")
    checkout = pd.to_datetime(out["price_quote_checkout_date"], errors="coerce")
    out["quote_nights"] = (checkout - checkin).dt.days

    # Which nights were quoted matters: Friday and Saturday nights cost far more
    # (see REPORT.md, "Date effects"). Share of the quoted nights that are
    # Fri/Sat: busday_count with weekmask Fri+Sat counts those days in
    # [check-in, check-out), i.e. the nights actually stayed.
    ok = checkin.notna() & checkout.notna() & (checkout > checkin)
    fri_sat = np.zeros(len(out))
    fri_sat[ok.to_numpy()] = np.busday_count(checkin[ok].values.astype("datetime64[D]"),
                                             checkout[ok].values.astype("datetime64[D]"),
                                             weekmask="0000110")
    out["weekend_share"] = np.where(ok, fri_sat / out["quote_nights"].where(ok, 1), np.nan)

    # "1.5 shared baths" -> 1.5 and shared=1; "Shared half-bath" -> 0.5 and shared=1.
    # The numeric `bathrooms` column is empty for 24% of rows; the text almost never is.
    t = out["bathrooms_text"].fillna("").str.lower()
    n = pd.to_numeric(t.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    n = n.where(~t.str.contains("half-bath"), 0.5)
    out["bathrooms_n"] = n.where(t != "", np.nan)
    out["bath_shared"] = t.str.contains("shared").astype(int)

    out["district"] = out["neighbourhood_group_cleansed"]
    out["neighbourhood"] = out["neighbourhood_cleansed"]
    return out


# ----------------------------------------------------------------------------
# Step 2: remove rows that are out of scope or unusable
# ----------------------------------------------------------------------------

def filter_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"raw_rows": len(df)}

    def apply(frame, mask, name):
        log[name] = int((~mask).sum())
        return frame[mask]

    # No price: mostly listings carried over from older scrapes that are no
    # longer bookable (78% of those vs 1.5% of current-scrape listings).
    df = apply(df, df["price_eur"].notna(), "dropped_no_price")
    # Long-stay quotes: monthly discounts make "price per night" a different quantity.
    df = apply(df, df["quote_nights"].between(1, MAX_QUOTE_NIGHTS), "dropped_long_stay_quote")
    # Hotel rooms and shared rooms are different businesses from a new private host.
    df = apply(df, df["room_type"].isin(ROOM_TYPES_IN_SCOPE), "dropped_room_type_out_of_scope")
    # Implausible prices: e.g. EUR 15,000 for a 1-bed flat, EUR 1,892 for a single room.
    per_guest = df["price_eur"] / df["accommodates"].clip(lower=1)
    ok = (df["price_eur"] >= MIN_PRICE_EUR) & (per_guest <= MAX_PRICE_PER_GUEST_EUR)
    df = apply(df, ok, "dropped_implausible_price")
    # Rows that can't be placed on the map (none in this scrape, but guard anyway).
    df = apply(df, df[["latitude", "longitude"]].notna().all(axis=1), "dropped_no_coordinates")

    log["final_rows"] = len(df)
    return df.copy(), log


# ----------------------------------------------------------------------------
# Step 3: correct values that are wrong or inconsistent
# ----------------------------------------------------------------------------

def fix_values(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    out = df.copy()
    private = out["room_type"] == "Private room"
    entire = out["room_type"] == "Entire home/apt"

    # Record missingness BEFORE fixing: a blank bedrooms field carries signal
    # (blank private rooms have a median of ~EUR 68 vs ~EUR 87 when filled in).
    out["bedrooms_missing"] = out["bedrooms"].isna().astype(int)

    # Private rooms: hosts inconsistently enter either 1 (the guest's room) or
    # the bedrooms of the whole flat (up to 50 for hostels), and price does not
    # rise with it. The guest always rents one room, so set it to 1.
    log["private_room_bedrooms_blank_set_to_1"] = int((private & out["bedrooms"].isna()).sum())
    log["private_room_bedrooms_over_1_set_to_1"] = int((private & (out["bedrooms"] > 1)).sum())
    out.loc[private, "bedrooms"] = 1.0

    # "0 baths" on an entire home is an entry error, not a flat without a
    # bathroom. Mark as unknown; train.py imputes it from the training split.
    zero_bath = entire & (out["bathrooms_n"] == 0)
    log["fixed_entire_home_zero_baths"] = int(zero_bath.sum())
    out.loc[zero_bath, "bathrooms_n"] = np.nan

    # Cap extreme values that are real but rare, so a few hostels with
    # "19 baths" don't dominate a linear model.
    out["bathrooms_n"] = out["bathrooms_n"].clip(upper=6)
    out["bedrooms"] = out["bedrooms"].clip(upper=8)
    out["accommodates"] = out["accommodates"].clip(upper=16)
    out["minimum_nights"] = out["minimum_nights"].clip(lower=1, upper=MAX_QUOTE_NIGHTS)
    # Airbnb's own ceiling is 1,125 nights; a few rows exceed it as placeholders.
    out["maximum_nights"] = out["maximum_nights"].clip(lower=1, upper=1125)
    return out


# ----------------------------------------------------------------------------
# Checks that the clean file is what we think it is
# ----------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    assert df["id"].is_unique, "duplicate listing ids"
    assert df["price_eur"].between(MIN_PRICE_EUR, 16 * MAX_PRICE_PER_GUEST_EUR).all()
    assert df["quote_nights"].between(1, MAX_QUOTE_NIGHTS).all()
    assert set(df["room_type"]) <= set(ROOM_TYPES_IN_SCOPE)
    assert (df.loc[df["room_type"] == "Private room", "bedrooms"] == 1).all()
    assert df["district"].notna().all()
    assert df["neighbourhood"].notna().all()
    assert df["weekend_share"].between(0, 1).all()


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pure function: raw export in, clean frame and cleaning log out."""
    df = parse_types(raw)
    df, log = filter_rows(df)
    df = fix_values(df, log)
    out = df[OUTPUT_COLUMNS].reset_index(drop=True)
    validate(out)
    return out, log


def run_cleaning(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Read the raw file, clean it, write the clean file and log, return both."""
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    out, log = clean(raw)
    # mtime=0 makes the gzip byte-identical on every run, so git only shows
    # the clean file as changed when a cleaning rule actually changed it.
    out.to_csv(CLEAN_PATH, index=False, compression={"method": "gzip", "mtime": 0})
    LOG_PATH.write_text(json.dumps(log, indent=2) + "\n")
    if verbose:
        print("Cleaning log:", json.dumps(log, indent=2))
        print(f"Wrote {CLEAN_PATH}: {len(out):,} rows x {out.shape[1]} columns. "
              "Missing values left for the training-split imputer:",
              out[["bedrooms", "bathrooms_n", "maximum_nights"]].isna().sum().to_dict())
    return out, log


if __name__ == "__main__":
    run_cleaning()
