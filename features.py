"""
Feature engineering, shared by train.py and app.py.

Both stages call the same engineer_features(), so the app turns a new listing
into model inputs exactly the way training did. If this logic were copied into
app.py instead, the two copies could drift apart without anyone noticing.

Input is the CLEAN dataset written by clean_data.py. Everything here works on
one row at a time and is deterministic: nothing is *fitted* in this file.
Anything that learns from data (median imputation, scaling, one-hot categories)
lives in the sklearn preprocessor built in train.py and is fitted on the
training split only.

Every listing feature uses only information available before the listing
goes live (location, the flat itself, the title and description the host
writes, the amenities and booking rules they set). Reviews, occupancy and
host history are excluded on purpose: none of them exists before listing.

Which features are in here, and why, is decided by one explicit rule, applied
in experiments/feature_experiments.py (see REPORT.md, "How features were
chosen").

Two further inputs describe the QUOTE rather than the listing: the share of
Friday/Saturday nights in the quoted stay and the number of nights quoted.
The target is a price for specific dates, and those dates move it a lot
(weekend and event nights cost far more). Training with these "quote controls"
separates the date effect from the listing effect. At prediction time they are
set to a fixed reference stay (see reference_stay), so the suggestion is a
clean base weekday rate.
"""

import json
import re

import numpy as np
import pandas as pd
import torch
from torch import nn

PUERTA_DEL_SOL = (40.4169, -3.7035)

# ----------------------------------------------------------------------------
# Amenities: Airbnb has 3,000+ distinct amenity strings ("AC - split type
# ductless system", "Central air conditioning", ...). Each CONCEPT below is a
# regex matched against every string in a listing's amenity list.
# ----------------------------------------------------------------------------
AMENITY_PATTERNS = {
    "am_air_conditioning": r"air conditioning|\bAC\b",
    "am_elevator": r"\belevator\b",
    "am_dishwasher": r"dishwasher",
    "am_washer": r"\bwasher\b",
    "am_dryer": r"(?<!hair )\bdryer\b",            # "Hair dryer" is not a dryer
    "am_self_checkin": r"self check-in",
    "am_workspace": r"dedicated workspace",
    "am_pool": r"\bpool\b(?! table| view)",         # "Pool table" is not a pool
    "am_parking_on_premises": r"parking on premises",
    "am_outdoor_space": r"patio|balcony|terrace|backyard",
    "am_nice_view": r"skyline view|park view|mountain view|garden view",
    "am_gym": r"\bgym\b(?! nearby)",
    "am_heating": r"heating",
    "am_long_term_allowed": r"long term stays allowed",
    "am_pets_allowed": r"pets allowed",
    "am_host_greets": r"host greets you",
    "am_breakfast": r"\bbreakfast\b",
}
_AMENITY_REGEX = {k: re.compile(p, re.IGNORECASE) for k, p in AMENITY_PATTERNS.items()}

# One real Airbnb string per concept. The app uses these to build an amenity
# list for a new listing, and the tests check each one matches its own pattern.
AMENITY_EXAMPLES = {
    "am_air_conditioning": "Air conditioning",
    "am_elevator": "Elevator",
    "am_dishwasher": "Dishwasher",
    "am_washer": "Washer",
    "am_dryer": "Dryer",
    "am_self_checkin": "Self check-in",
    "am_workspace": "Dedicated workspace",
    "am_pool": "Pool",
    "am_parking_on_premises": "Free parking on premises",
    "am_outdoor_space": "Patio or balcony",
    "am_nice_view": "City skyline view",
    "am_gym": "Gym",
    "am_heating": "Heating",
    "am_long_term_allowed": "Long term stays allowed",
    "am_pets_allowed": "Pets allowed",
    "am_host_greets": "Host greets you",
    "am_breakfast": "Breakfast",
}

PROPERTY_PATTERNS = {
    # professionally run accommodation, priced differently from a private flat
    "prop_hotel_like": r"hotel|hostel|aparthotel|bed and breakfast|guesthouse",
    "prop_serviced_apartment": r"serviced apartment",
    "prop_loft": r"\bloft\b",
    "prop_house": r"entire home|villa|townhouse|casa particular|chalet",
}

# Words in the listing title (Spanish and English). The title is written by
# the host before the first booking, and it signals how they position the flat.
TITLE_KEYWORDS = {
    "kw_luxury": r"lux|lujo|premium|exclusiv|boutique|design|diseño",
    "kw_penthouse": r"penthouse|[aá]tico|duplex|dúplex",
    "kw_terrace": r"terra[cz]",
    "kw_view": r"view|vista",
    "kw_studio": r"studio|estudio",
    "kw_cozy": r"cozy|acogedor|small|pequeñ",
}

# Upscale wording anywhere in the title OR the description, Spanish and English.
UPSCALE_WORDS = r"luxur|lujo|design|diseño|stylish|contemporary|contempor|prestig|elegan|exclusiv|premium|boutique"


def haversine_km(lat, lon, lat0, lon0):
    """Great-circle distance in km."""
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def parse_amenities(amenities: pd.Series) -> pd.DataFrame:
    """JSON list of amenity strings -> one 0/1 flag per concept + log(count)."""
    lists = amenities.fillna("[]").map(json.loads)
    out = {}
    for name, rx in _AMENITY_REGEX.items():
        out[name] = lists.map(lambda items, rx=rx: int(any(rx.search(a) for a in items)))
    out["log_n_amenities"] = np.log1p(lists.map(len))
    return pd.DataFrame(out, index=amenities.index)


def parse_title(name: pd.Series) -> pd.DataFrame:
    title = name.fillna("").str.lower()
    return pd.DataFrame({k: title.str.contains(p, regex=True).astype(int)
                         for k, p in TITLE_KEYWORDS.items()}, index=name.index)


def has_upscale_wording(name: pd.Series, description: pd.Series) -> pd.Series:
    text = name.fillna("").str.lower() + " . " + description.fillna("").str.lower()
    return text.str.contains(UPSCALE_WORDS, regex=True).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean listings (clean_data.py output) -> model features."""
    X = pd.DataFrame(index=df.index)

    # Location: price falls fast in the first kilometres from the centre and
    # then flattens, so distance enters both linearly and as a log.
    dist = haversine_km(df["latitude"], df["longitude"], *PUERTA_DEL_SOL)
    X["dist_sol_km"] = dist
    X["log_dist_sol"] = np.log(dist + 0.2)   # +0.2 km so flats next to Sol don't blow up
    X["district"] = df["district"]

    # Room and property type
    X["is_private_room"] = (df["room_type"] == "Private room").astype(int)
    ptype = df["property_type"].fillna("").str.lower()
    for name, pat in PROPERTY_PATTERNS.items():
        X[name] = ptype.str.contains(pat, regex=True).astype(int)

    # Size. Logs because the 2nd guest adds more value than the 10th.
    X["log_accommodates"] = np.log(df["accommodates"].clip(lower=1))
    # An extra guest is worth a different amount in a private room than in a
    # whole flat, so the slope on size is allowed to differ by room type.
    X["private_x_log_accommodates"] = X["is_private_room"] * X["log_accommodates"]
    X["bedrooms"] = df["bedrooms"]                   # NaNs imputed later, on train only
    X["bedrooms_missing"] = df["bedrooms_missing"]
    X["bathrooms_n"] = df["bathrooms_n"]             # NaNs imputed later, on train only
    X["bath_shared"] = df["bath_shared"]

    # Booking rules the host sets
    X["log_min_nights"] = np.log(df["minimum_nights"])
    X["log_max_nights"] = np.log(df["maximum_nights"])

    # Quote controls: which nights the price was quoted for (not a listing property)
    X["weekend_share"] = df["weekend_share"]
    X["log_quote_nights"] = np.log(df["quote_nights"])

    X = X.join(parse_amenities(df["amenities"]))
    X = X.join(parse_title(df["name"]))
    X["upscale_wording"] = has_upscale_wording(df["name"], df["description"])
    return X


QUOTE_CONTROLS = ["weekend_share", "log_quote_nights"]

NUMERIC_FEATURES = [
    "dist_sol_km", "log_dist_sol", "log_accommodates", "private_x_log_accommodates",
    "bedrooms", "bathrooms_n", "log_min_nights", "log_max_nights", "log_n_amenities",
] + QUOTE_CONTROLS
BINARY_FEATURES = (
    ["is_private_room", "bedrooms_missing", "bath_shared"]
    + list(PROPERTY_PATTERNS)
    + list(AMENITY_PATTERNS)
    + list(TITLE_KEYWORDS)
    + ["upscale_wording"]
)
CATEGORICAL_FEATURES = ["district"]

# Feature groups for the ablation study in train.py ("what is each group worth?")
FEATURE_GROUPS = {
    "Location": ["dist_sol_km", "log_dist_sol", "district"],
    "Size": ["log_accommodates", "private_x_log_accommodates", "bedrooms", "bedrooms_missing",
             "bathrooms_n", "bath_shared"],
    "Room & property type": ["is_private_room"] + list(PROPERTY_PATTERNS),
    "Amenities": list(AMENITY_PATTERNS) + ["log_n_amenities"],
    "Title & description wording": list(TITLE_KEYWORDS) + ["upscale_wording"],
    "Booking rules (min/max nights)": ["log_min_nights", "log_max_nights"],
    "Quote controls (dates quoted)": QUOTE_CONTROLS,
}


def reference_stay(df: pd.DataFrame, weekend: bool = False) -> pd.DataFrame:
    """Set the quote controls to the stay we want a price for: the host's
    minimum stay, on weekday nights (weekend=False) or Fri/Sat nights
    (weekend=True). Used by the app; evaluation uses the real quoted dates."""
    out = df.copy()
    out["quote_nights"] = out["minimum_nights"].clip(lower=1)
    out["weekend_share"] = 1.0 if weekend else 0.0
    return out


# ----------------------------------------------------------------------------
# The torch model for the "standard workflow" (the app loads it too)
# ----------------------------------------------------------------------------

class LinearRegressionModel(nn.Module):
    """y_hat = X @ w + b. A single nn.Linear, nothing else."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)
