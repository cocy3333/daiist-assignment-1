<!-- Draft written with AI assistance. Before submitting: rewrite each section in your own words,
     fill in the header, check the AI disclosure at the end, and delete this comment. -->

# Assignment 1 Report

- **Name**: Constantin Hatecke
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

**How to run.** `uv run python main.py train` goes from the raw export to saved models in one step (about 40 seconds); `uv run python main.py app` opens the dashboard. Optional: `uv run pytest -q` (34 checks) and `uv run python experiments/feature_experiments.py` (about 90 seconds; reproduces every feature decision in `experiments/results.md`).

## Dataset

**Source.** Inside Airbnb, detailed listings export for Madrid, scraped between 20 June and 2 July 2026. One row is one Airbnb listing as it appeared on the scrape date: 22,708 rows and 90 columns.

**In the repo.** The untouched export is committed compressed at `data/raw/madrid_listings_raw.csv.gz` (10 MB; the CSV is 59 MB). `clean_data.py` turns it into `data/madrid_listings_clean.csv.gz` (14,723 listings from 5,646 hosts, 23 columns, 3 MB) and writes `data/cleaning_log.json` with the number of rows each rule removed. `main.py train` runs this cleaning first, so the whole pipeline starts from the raw file.

**Why this dataset.** Pricing a flat is a real decision with a cost on both sides, and the data forces genuine feature work:

- the price column is not what it seems (next section);
- amenities come as 3,290 distinct free-text strings;
- bathrooms are stored as text;
- a few professional operators run hundreds of near-identical flats.

## Business / real-life framing

**The decision.** A prospective host or small investor in Madrid is looking at a flat that is not listed yet. Should it go on Airbnb, at what starting price, and which of its characteristics are associated with higher prices? Price too high and the calendar stays empty; too low and money is lost on every booking. The model gives a **base weekday rate for an early-summer short stay**, using only information available before the listing goes live, and shows how each characteristic relates to the price. It is a market anchor (what comparable hosts ask), not a revenue-optimal price: the data contains asking prices, not bookings.

**Why not Airbnb's own pricing tool.** Airbnb's Smart Pricing already adjusts nightly prices for live listings, and this model does not compete with it. It serves the decision before that point. Smart Pricing cannot price a flat that is not listed, and it gives a number without the reasons behind it. That is why the model uses only pre-listing information, and why an interpretable linear model is a deliberate choice here rather than only a requirement of the assignment. After launch, this model gives the starting anchor and the reasons behind it, then after getting feedback the in-house tool takes them where they want to go. 

Each of the following pipeline decisions follows from that framing.

**1. Target: the price of a short stay, as log(price).** The `price` column is not a base rate. It is the price Airbnb quoted for specific check-in and check-out dates, and long stays carry monthly discounts:

| Length of the quoted stay | Median price per night |
|---|---|
| 1 to 7 nights | €130 to €147 |
| 15 to 60 nights | €84 to €88 |
| more than 60 nights | €16 |

Only quotes of 1 to 7 nights are kept. The target is log(price) because pricing errors matter in proportion: €20 off a €60 room is serious, €20 off a €400 flat is noise. On the same features and folds, the log model reaches a **median error of 20.9% against 26.1%** for a model on raw euros, which also predicts negative prices for some listings.

7.4% of the kept quotes include a discount (for brand-new listings 23%, likely Airbnb's new-listing promotion). The target stays the price after discount, because a model on the pre-discount rate predicts no better (`experiments/results.md`).

**2. Which nights were quoted: controlled for, then fixed at prediction time.** Because the price belongs to specific dates, the target mixes the listing's value with the calendar. Friday and Saturday nights cost far more, and longer quotes are cheaper per night (see Validity). The model therefore gets two **quote controls**: the share of Fri/Sat nights in the quoted stay and the number of nights quoted. They are used in training so the date effect does not leak into the listing coefficients. When pricing a new listing, they are set to a fixed **reference stay**: the host's minimum stay, on weekday nights.

**3. Scope.** Only entire homes and private rooms are kept. Hotel rooms and shared rooms (164 rows) are a different business from a private host.

**4. Features: pre-listing information only.** Excluded on purpose:

- reviews and review scores;
- occupancy, availability and estimated revenue;
- response rates, host listing counts, and wording that identifies a particular operator.

None of these exists, or is fair to use, before a listing goes live; a model using them would test well and fail at the moment it is used.

**5. Split: grouped by host, not random and not by time.**

- **Why grouped.** Professional operators run many near-identical flats. The largest host has 449 listings, and hosts with 10 or more listings make up 41% of the raw rows. A random split puts copies of the same flat in train and test, so the test set no longer represents a new host. The split is therefore 70/15/15 by `host_id` (`GroupShuffleSplit`), with a check that no host appears in two parts.
- **Why not time-based.** The data is one snapshot taken over two weeks, so there is no past and future to separate. The leakage to prevent is between hosts, not across time.

**6. Metric.** For the host, the error that matters is relative, so the headline metrics are:

- the **median absolute percentage error**;
- the **share of listings priced within ±20%** of the market.

Tuning and feature selection use **RMSE on log price** instead. It is the quantity the model minimises, and its squared form penalises the large misses that cost a host the most. Because this is regression, the dashboard replaces a classification threshold slider with a **tolerance slider**: how often the suggestion lands within ±X%. It splits the misses into two costs:

- **underpricing:** euros lost per booked night;
- **overpricing:** risk of empty nights.

## Data preparation & feature engineering

### Cleaning (`clean_data.py`)

Every rule looks at one row at a time, so nothing learned from other rows (in particular test rows) can leak in.

**Rows removed:**

| Rule | Rows removed | Why |
|---|---|---|
| No price | 3,536 | Mostly listings carried over from older scrapes that are no longer bookable: 78% of those have no price, against 1.5% in the current scrape. Dropped, not imputed. |
| Quote longer than 7 nights | 4,095 | A different quantity (monthly discounts). |
| Hotel or shared room | 164 | Out of scope. |
| Price below €20, or above €250 per guest | 190 | Placeholder prices, e.g. €15,000 for a one-bedroom flat or €1,892 for a single room. A per-guest cap avoids removing genuinely expensive large flats. |
| **Remaining** | **14,723** | |

**Values corrected or derived:**

- **Bedrooms for private rooms.** Hosts fill this inconsistently: some enter 1 (the guest's room), others the bedrooms of the whole flat (190 private rooms report 2 to 50), and 2,363 leave it blank. The guest always rents one room, so every private room is set to 1.
- **The missing-bedrooms flag is recorded before that fill.** A blank field carries signal: blank private rooms have a median price of €68 against €87 when filled in.
- **"0 baths" on 61 entire homes** is an entry error, so these become unknown rather than zero.
- **Rare extremes are capped** (bathrooms 6, bedrooms 8, guests 16, maximum nights at Airbnb's ceiling of 1,125) so a few odd listings cannot dominate a linear model. Minimum nights is clipped to 1 to 7, consistent with the short-stay scope.
- **Quote description:** the number of nights quoted, and the share of them that are Friday or Saturday nights.
- **Checked and deliberately kept:** 177 rows look like duplicates (same host, coordinates, size and price). They have different listing IDs and separate review histories: identical units in one building registered at the building's address. They are real listings, and the host-grouped split keeps them on one side.

**Missing values are not filled during cleaning.** Filling them with a dataset median would let test rows shape that median. 1,091 bedrooms, 77 bathrooms and 1 maximum-nights value stay missing in the clean file and are imputed later from the training split only. `number_of_reviews` is kept for one evaluation check (Validity, point 4) and is never a model input; a test enforces this.

### Features (`features.py`)

Training and the app share `features.py`, so a new listing in the dashboard goes through exactly the same code.

| Group | Features | Reasoning |
|---|---|---|
| Location | Haversine distance to Puerta del Sol, its log, district (one-hot, Centro as reference) | Price falls fast in the first kilometres from the centre, then flattens: the linear plus log terms let the model bend. The district captures what distance misses (Salamanca vs Usera at similar distances). |
| Size | log(guests), bedrooms, bathrooms parsed from text plus a shared-bathroom flag, missing-bedrooms flag, private room × log(guests) | Logs because the second guest adds more value than the tenth. The interaction lets an extra guest count differently in a private room than in a whole flat. |
| Room and property type | Private-room flag; hotel-like, serviced apartment, loft, house (parsed from 44 `property_type` strings) | Professionally run and unusual properties are priced differently. |
| Amenities | 17 concept flags from 3,290 strings via regular expressions (incl. breakfast), plus log(number of amenities) | "AC - split type ductless system" and "Central air conditioning" both mean air conditioning. Regex guards stop "Hair dryer" counting as a dryer and "Pool table" as a pool (checked in the tests). |
| Wording | 6 title-word flags (luxury/design, penthouse/ático, terrace, view, studio, cozy/small) and one flag for upscale words anywhere in the title or description | The host writes these before the first booking; they signal how the flat is positioned. |
| Booking rules | log(minimum nights), log(maximum nights) | Set by the host. Minimum nights also defines the reference stay when pricing. |
| Quote controls | Share of Fri/Sat nights in the quoted stay, log(nights quoted) | Describe the quote, not the listing (framing, point 2). |

**Preprocessing** is a scikit-learn `ColumnTransformer` fitted on the training split only, then saved:

- **Numeric columns:** median imputation, then standardisation.
- **Binary flags:** standardised.
- **District:** one-hot with Centro dropped as the reference category (see the modelling section for why), then standardised.

That gives **62 model inputs**.

### How the features were chosen

Full details, with every candidate and number, are in `experiments/results.md`, produced by `experiments/feature_experiments.py`. All of it uses training + validation listings only; the test set played no part.

**The procedure.**

1. **Starting model**, chosen by reasoning before any test: location, size (guests, bedrooms, bathrooms), room and property type, 16 amenity concepts, minimum nights, plus the two quote controls.
2. **Each candidate is added to it on its own** and kept only if it passes all four parts of one rule:
   - **a. Legitimate:** known before the listing goes live, and does not identify a particular host.
   - **b. Big enough:** lowers the error (RMSE of log price) by at least 0.001 on average over 5 rounds of cross-validation grouped by host.
   - **c. Consistent:** lowers it in at least 4 of the 5 rounds.
   - **d. Confirmed:** also lowers it on the validation split.
3. **Final model** = starting model + every candidate that passed.

**Where the candidates came from.** Most were reasoned out in advance or found by going through the unused raw columns. Some (upscale and budget words, seasonal-rental wording, six single amenities) came from inspecting which words and amenities went with the model's errors on the training split. Check d scores every candidate on the validation split, which played no part in that inspection.

**Accepted (5 of 21 candidates):**

| Candidate | Error change | Better in | Validation split |
|---|---|---|---|
| Upscale words in title or description | −0.0027 | 5 of 5 | improves |
| Title words (6 flags) | −0.0022 | 4 of 5 | improves |
| Maximum nights | −0.0013 | 4 of 5 | improves |
| Private room × guests | −0.0010 (narrowly: 0.00104) | 4 of 5 | improves |
| Amenity: breakfast | −0.0010 (narrowly: 0.00103) | 4 of 5 | improves |

Together they lower the error from 0.3665 to 0.3596, in all 5 rounds.

**Rejected examples worth knowing:**

- **Beds** (−0.0003): on its own a strong predictor, but it correlates 0.85 with guests. Once guests and bedrooms are in, beds add almost nothing. A feature's value depends on what is already in the model.
- **All 128 neighbourhoods as categories** made the error worse (+0.0017): too many small categories. A smoothed neighbourhood price level (−0.0010, only 3 of 5 rounds) and the prices of the 25 nearest listings (−0.0007) also fall short. Location is already well captured (Validity, point 2).
- **Seasonal-rental wording.** "Corta duración" went with prices about 30% above the model, but it appears in 134 listings from only 9 hosts: operator template text. The host-grouped rounds show it does not carry over to new hosts.

**Rejected on principle, whatever they score:**

- **Lead time** (−0.0025): Inside Airbnb quotes the first *free* stay, so a late date reveals a booked-up listing. That is demand information nobody has before listing.
- **Check-in month** (−0.0026): 94% of quotes are June/July, so other months cannot be learned.
- **Operator brand words** (−0.0011): they identify the host.
- **A leakage demonstration:** nearby prices computed *with* the listing's own price and its host's other listings made the model worse in all 5 rounds (+0.0039). In training the feature partly contains the answer; for a new host that crutch is gone.

**How much signal is left unused?** Three diagnostics against the final model:

| Diagnostic | Error change | What it means |
|---|---|---|
| Every word and word pair in title + description (3,000 columns) | −0.0074 | Real but diffuse; 3,000 unnamed columns would make the model unreadable, and part of it is operator template text |
| Every amenity seen in at least 1% of listings (142 columns) | −0.0030 | The single amenities with a real effect were tested above; the rest is many tiny effects |
| Gradient-boosted trees on the same features | −0.0092 | How much curves and interactions could add; the assignment requires a linear model |

So the final model is within about 0.01 of what more flexible approaches extract from the same data. The larger remaining error comes from information the data does not contain (Limitations).

**Ablation of the final model.** Removing each group in turn, same 5 rounds:

| Group removed | Error increase | Rounds worse |
|---|---|---|
| Size | +0.0764 | 5/5 |
| Location | +0.0217 | 5/5 |
| Quote controls | +0.0176 | 5/5 |
| Amenities | +0.0086 | 5/5 |
| Room & property type | +0.0053 | 5/5 |
| Wording (title + description) | +0.0037 | 5/5 |
| Booking rules (min/max nights) | +0.0013 | 4/5 |

## Modeling: three implementations, one model

**Model: linear regression with an L2 penalty (ridge) on log(price).** The target is continuous, so linear rather than logistic regression. Interpretability is part of the value in this framing: each coefficient reads as the percentage difference in price associated with a characteristic, other things equal. The small penalty keeps the 62 standardised coefficients stable where features are correlated (distance and log distance, guests and bedrooms).

**Making the three solve the identical problem.**

- **scikit-learn** `Ridge` minimises the *sum* of squared errors + α·‖w‖² and does not penalise the intercept.
- **Dividing by n** gives the same minimiser in *mean* form: MSE + (α/n)·‖w‖².
- **Manual PyTorch loop** (`fit_manual_torch`): raw tensors with `requires_grad`, the penalty added to the loss by hand, `loss.backward()`, the update `w -= lr * w.grad` inside `torch.no_grad()`, then the gradients reset to zero.
- **Standard workflow** (`fit_standard_torch`): `nn.Linear` in an `nn.Module`, `nn.MSELoss` and `torch.optim.SGD`. The penalty becomes weight decay: `weight_decay = wd` adds wd·w to the gradient, the gradient of (wd/2)·‖w‖², so **wd = 2α/n**, applied to the weight only through a separate parameter group with zero decay for the bias. A test shows that using α/n instead gives visibly different weights.
- **Shared setup.** Both PyTorch versions use full-batch gradient descent on the same standardised inputs. The manual loop starts from zeros and `nn.Linear` from its default random start; the loss is convex, so both must reach the same optimum.

**Tuning (only the two allowed hyperparameters).**

- **Ridge α**, chosen on validation: RMSE(log) is 0.3355 for α between 0.1 and 10, 0.3356 at 100, 0.3373 at 1,000 and 0.3633 at 10,000, so **α = 0.1**. With 10,619 training listings and 62 inputs there is little overfitting for the penalty to fix.
- **Learning rate.** Validation error cannot choose it: every stable rate ends at the same optimum and only the speed differs. Theory gives the limit: gradient descent on this loss is stable only if lr < 2/λmax, where λmax = 10.01 is the largest eigenvalue of the Hessian (2/n)·XᵀX, so **lr < 0.1997**. The rule is the fastest rate that reaches sklearn's objective within 0.001%:

| Learning rate | Result |
|---|---|
| 0.01 | too slow: not converged in 3,000 epochs |
| 0.03 | converged after 1,060 epochs |
| **0.1** | **converged after 320 epochs (chosen)** |
| 0.2 | diverged (just above the 0.1997 limit) |
| 0.3 | diverged |

**Test results.** 1,967 listings from hosts never seen in training, each predicted for its actual quoted stay. The 95% intervals come from resampling *hosts*, not listings: a host's listings are near-copies, so resampling listings would overstate the evidence.

| Model | Median abs % error (95% CI) | Within ±20% | MAE (€/night) | RMSE (log) | R² (log) |
|---|---|---|---|---|---|
| Baseline: median of district × room type | 27.3% (25.8 to 29.4) | 37.5% | 64.42 | 0.4777 | 0.388 |
| scikit-learn Ridge | 20.0% (18.4 to 21.4) | 49.7% | 47.07 | 0.3420 | 0.687 |
| PyTorch manual loop | 20.0% | 49.7% | 47.07 | 0.3420 | 0.687 |
| PyTorch nn.Module + optim | 20.0% | 49.7% | 47.07 | 0.3420 | 0.687 |

The model beats the baseline by **5.7 to 9.5 percentage points** of median error (95% interval of the paired difference). Part of that gap comes from the quote controls, information the baseline does not use; the ablation separates the two.

**Do the three agree? Yes.**

- **Coefficients:** largest gap 6.6e-07 (sklearn vs manual) and 6.0e-07 (sklearn vs nn.Module).
- **Predictions:** largest gap 1.7e-06 in log price.
- **Training objective:** 0.1270980 for all three.

The remaining gap is float32 arithmetic in PyTorch against float64 in sklearn. The dashboard's loss chart shows the gap to sklearn's optimum shrinking until it stalls at about 1e-8, float32's precision limit.

**The first version did not agree, and why.** It one-hot encoded all 21 districts and kept an intercept. `train.py` rebuilds that version on every run (`rerun_dummy_trap`), so these numbers are reproduced, not remembered:

| | All 21 dummies + intercept | Centro dropped (final) |
|---|---|---|
| Largest coefficient gap, sklearn vs manual loop | 0.267 | 6.6e-07 |
| Largest gap outside the district dummies | 0.034 | |
| Average shift of district coefficients (manual − sklearn) | +0.221 | 0 |
| Shift of the intercept (manual − sklearn) | −0.251 | 0 |
| Largest prediction gap (log price) | 0.098 | 1.7e-06 |
| Smallest Hessian eigenvalue | 6.9e-16 | 0.055 |

The 21 dummies always sum to 1, which is exactly the intercept column. Raising every district coefficient by c and lowering the intercept by c gives the same predictions for any c, so the loss is perfectly flat in that direction: the zero eigenvalue in the table.

- **sklearn** solves the equations exactly and lands on the one point the ridge penalty prefers.
- **Gradient descent** starts at zero and barely moves along a flat direction, so it stops at a different point on it.

The opposite shifts of the districts (+0.22) and the intercept (−0.25) are that direction. The slow direction also holds up convergence elsewhere (0.034 outside the dummies), so predictions match only roughly.

The fix had two parts:

- **Drop one category** (Centro, the reference), which removes the flat direction.
- **Standardise the dummies** like every other column, so all directions have similar curvature.

**Did the host-grouped split matter?** Averaged over 5 seeds with sklearn:

| Split | R² (log) | Median error |
|---|---|---|
| Grouped by host | 0.650 | 20.5% |
| Random | 0.676 | 20.4% |

The random split flatters the squared-error metrics but barely the median, so the leak is small *for this model*: a linear model with 62 inputs cannot memorise individual flats. Operator template wording and the leaky nearby-prices feature show how a more flexible model would exploit it. The grouped split stays because it matches how the tool is used.

**What the model learned.** Standardised coefficients on log price; each holds the other features fixed:

- private room −0.18;
- bedrooms +0.15;
- log(guests) +0.13;
- share of Fri/Sat nights +0.11;
- log(nights quoted) −0.09;
- distance to Sol −0.08 (linear) and −0.05 (log);
- shared bathroom −0.07;
- private room × guests +0.06;
- dishwasher +0.04, air conditioning +0.04;
- log(maximum nights) −0.03;
- upscale wording +0.03, breakfast +0.03, a luxury/design word in the title +0.03;
- Usera −0.03 and Salamanca +0.03 relative to Centro.

They are associations, not causes: listings with air conditioning price higher other things equal, but adding air conditioning to a flat does not guarantee that difference, because flats with it also differ in ways the data does not record. The amenity count gets a small negative coefficient (−0.025). That is not "more amenities lower the price": once the valuable amenities are in the model, a long list mostly means many cheap items, typical of budget listings.

## Validity: what the model can and cannot claim

These checks are computed in `train.py` on every run and shown in the dashboard's *Validity checks* tab.

**1. Which nights were quoted (the issue found and fixed).** Night by night, here is how much 1-night quotes ask above a model that ignores the dates (out-of-fold, train + validation):

- **ordinary weekdays:** between −3% and +2% in the first week;
- **Saturday 20 June:** +35%;
- **Friday 26 / Saturday 27 June:** +99% / +86%;
- **Friday 3 / Saturday 4 July:** +37% / +50%.

Holding the listing fixed, an all-Fri/Sat stay costs **+54%** per night against an all-weekday stay, and a 7-night quote **−25%** per night against a 1-night quote. On the test set, the quote controls remove the bias:

| Quoted stay (test) | Listings | Asks vs model, without controls | Asks vs model, final |
|---|---|---|---|
| No Fri/Sat night | 1,572 | −4.5% | −0.2% |
| Some Fri/Sat nights (up to half) | 293 | +3.5% | −2.3% |
| Mostly Fri/Sat nights | 102 | +28.5% | −4.5% |

Without the controls, the weekday price (the number a host needs) was **inflated by about 5%** by the expensive weekends; with them it is unbiased. The weekend effect rests on only three weekends, so the data cannot separate events from ordinary early-summer weekend demand. That is why the app's main number is the weekday base rate, and the weekend figure is shown as an uncertain average of this window.

**2. Location is captured about as well as this data allows.** The model knows each listing's coordinates (shifted by up to about 150 m by Airbnb for privacy), its distance to Sol and its district. Two checks on the leftover error:

- **The 126 neighbourhoods explain 3.8%** of the leftover error.
- **A listing's error correlates 0.045** with the average error of its 10 nearest listings from other hosts. If location were badly captured, neighbours would share their errors.

Location candidates that use other listings' actual prices fail the feature rule. Some neighbourhoods still have systematic misses (Recoletos +16%, Hispanoamérica +17%, Vallehermoso −20%). Micro-location (floor, view, street) is not in the data and blends into the "quality" error below.

**3. Seasonality across the year is outside the scope.** 94.2% of quotes are for June and July check-ins, so the model cannot learn how prices move over the year. It gives an early-summer rate. On log price, a season that raised all listings by the same percentage would only shift the intercept, so relative comparisons might carry over to other seasons. One snapshot cannot test that assumption, so the report does not claim it.

**4. New listings vs the hosts the model learned from.** The model learns from existing hosts but serves new ones. On the test set:

| Reviews | Listings | Asks vs model |
|---|---|---|
| 0 (brand new) | 137 | +7.3% |
| 1 to 5 | 237 | +5.6% |
| 6 to 30 | 495 | +1.1% |
| 31 to 100 | 530 | −1.6% |
| more than 100 | 568 | −4.5% |

New listings ask *more* than the model suggests; heavily reviewed ones, which demonstrably get booked, ask less. A reasonable reading is that newcomers start optimistic and frequently booked listings settle on competitive prices. For a new host, the suggestion therefore sits below what other newcomers ask and closer to prices that book. The 0-review group is small, so this is indicative.

**5. Are cheap or expensive listings mispriced?** Grouped by *actual* price, the test errors look like a strong bias:

| Actual price | Model vs actual |
|---|---|
| €26 to 87 | +16% |
| €87 to 122 | +14% |
| €122 to 157 | +4% |
| €157 to 220 | −6% |
| €220 and above | −23% |

This pattern is mostly **regression to the mean**, not a flaw of this model. Grouping by the actual price selects on what the model cannot see: the cheapest group is full of flats priced below what their features suggest, the most expensive full of flats priced above it. A simulated model that is right on average by construction, with prices scattering randomly around it, shows the same pattern across the same fifths (+27%, +17%, +6%, −11%, −23%).

The fair test groups by the *predicted* price, which is what is known when pricing a new flat (test set, 95% host-bootstrap intervals):

| Predicted price | Model vs actual | 95% interval |
|---|---|---|
| €34 to 55 (cheapest 5%) | −8.4% | −13% to −1% |
| €55 to 91 | −1.1% | −10% to +5% |
| €91 to 128 | +0.8% | −2% to +6% |
| €128 to 157 | +1.1% | −3% to +5% |
| €157 to 205 | +5.0% | −0% to +10% |
| €205 to 299 | +3.0% | −3% to +10% |
| €299 and above (top 5%) | −3.2% | −10% to +3% |

When the model predicts €250, flats like that are priced around €250 on average: the model is calibrated. The middle bands move around between the test and validation splits, so they are noise. The one effect that repeats is the cheapest-predicted 5%: −8% on test and −14% on validation. Real prices there are higher than predicted, likely because very cheap rooms cluster near a price floor while the straight-line model keeps extrapolating downward.

**Can this be used to make better predictions? No.**

- **Stretching every prediction away from the average** (validation split): the bias in the extreme groups shrinks, but the overall error rises (RMSE 0.3355 unchanged, 0.3410 at 10% stretch, 0.3722 at 30%), because it also pushes every flat the model rates as premium higher, including those that are not.
- **A separate correction per predicted-price band,** learned on validation and then tried on test: the median error improves slightly (20.0% → 19.4%) but the RMSE gets worse (0.3420 → 0.3446). The two measures disagree, and seven offsets tuned on 2,137 validation listings are easy to overfit. Adopting a fix because of a test result would also turn the test set into a tuning set.

So no correction is applied. The cheap-end floor effect is listed as a limitation below.

## Limitations & next steps

**Unexplained quality differences.** Much of the remaining error comes from what makes one flat better than a similar one: photo and renovation quality, the specific building, the view from a particular floor. The wording features target this and help a little. The diagnostics show the ceiling with this data: all 3,000 words gain 0.007, trees 0.009. Next steps:

- features from the listing photos (not reachable from this environment);
- a nonlinear model compared against this one on the same host-grouped split.

**Price floor at the cheap end.** For the cheapest-predicted 5% of flats (under about €55), real prices are 8 to 14% higher than predicted, on both evaluation splits. Treat the suggestion there as a lower bound. A nonlinear model, or a target that accounts for the €20 cut-off, could address it.

**A season- and calendar-aware target.** The fix in Validity point 1 controls for the dates; it does not model them. Next step: stack several quarterly scrapes. Each gives quotes on different dates, which makes check-in month and weekday estimable and turns the weekend figure into a measured premium.

**The suggestion is a median, not a mean.** Exponentiating a prediction of log price gives a typical price, so the average prediction is 8% below the average actual price. For a host a typical price is the right anchor; the dashboard shows a range instead of one number (half of real prices fall between 0.82× and 1.21× the prediction).

**Overpricing cost cannot be measured.** The data has asking prices, not bookings. Underpricing is quantified (€31 per listing per booked night left uncollected on average); the cost of empty nights is not. Next step: use booking outcomes, if a source with them is available, to weight the two errors by their real cost.

**A quoted price is not a transaction price.** Cleaning fees and whether anyone booked at that price are unknown. The target is what comparable hosts ask: a fair anchor, not proof of what guests pay.

## Generative AI use disclosure

*Check this against what you actually did and edit it before submitting.*

I used Claude (Anthropic) as an assistant throughout:

- **Choosing the dataset and framing:** comparing candidate datasets and discussing the business framing, including how the model differs from Airbnb's own pricing tool.
- **Code:** drafting the cleaning, feature engineering, training and dashboard code.
- **Analysis:**
  - diagnosing why the three implementations disagreed in the first version (the dummy-variable trap);
  - testing the validity concerns about location and seasonality that I raised, which led to the quote controls;
  - setting up the explicit feature-selection rule after I asked for clearer feature decisions;
  - checking whether the error pattern across price levels that I noticed could improve the predictions.
- **Report:** drafting this report.

I ran the full pipeline myself, checked the outputs against the dashboard, and rewrote the report in my own words. All decisions and numbers in this report come from the committed code and can be reproduced with `uv run python main.py train` and `uv run python experiments/feature_experiments.py`.
