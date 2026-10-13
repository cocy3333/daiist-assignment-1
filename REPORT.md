# Assignment 1 Report

- **Name**: Constantin Hatecke
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

**How to run.** `uv run python main.py train` (about 1 minute, raw data to saved models), then `uv run python main.py app`. Optional: `uv run pytest -q` (34 checks), `uv run python experiments/feature_experiments.py` (reproduces `experiments/results.md`).

**Code.** `clean_data.py` cleaning · `features.py` features · `pipeline.py` split, preprocessing, baselines, the three implementations · `diagnostics.py` evidence and validity checks · `train.py` runs it all · `app.py` dashboard · `experiments/` feature selection.

## Dataset

Inside Airbnb, Madrid listings (22,708 rows). Target: log of the nightly price quoted for a short stay.

| Cleaning step | Rows |
|---|---|
| Raw | 22,708 |
| No price | −3,536 |
| Price quoted for a stay longer than 7 nights | −4,095 |
| Hotel / shared rooms | −164 |
| Implausible price (< €20 or placeholder) | −190 |
| **Final** | **14,723** |

Fixes instead of drops: blank bedrooms of private rooms set to 1 (2,363), private rooms showing the whole flat's bedrooms set to 1 (190), "0 baths" on entire homes set to unknown (61). No imputation in cleaning; gaps are filled from the training split only.

Data facts that matter: prices are quotes for specific dates, 94% of them for June/July check-ins, and they are asking prices, not bookings.

**Your take:** why this dataset, and what worried you about it at the start?

## Business framing

Decision supported: the starting weekday price for a Madrid flat that is not listed yet, and which characteristics go with a higher price. Airbnb's Smart Pricing only works once a listing is live.

How it drove the pipeline:
- only information known before listing (no reviews, no booking lead time)
- split grouped by host
- metrics in %: median absolute % error and share within ±20%
- the app prices a weekday stay, so the test set is also scored on weekday-only quotes

**Your take:** who is the user, what does a wrong price cost them in each direction, and why % error?

## Features

| Group | Features |
|---|---|
| Location | distance to Puerta del Sol (plain + log), 21 districts (Centro = reference) |
| Size | guests (log), guests × private room, bedrooms (+ missing flag), bathrooms, shared bathroom |
| Room & property type | private room, property kind flags |
| Amenities | 17 amenity flags, number of amenities (log) |
| Wording | 6 title keywords, upscale words in title or description |
| Booking rules | minimum and maximum nights (log) |
| Quote controls | share of Fri/Sat nights in the quote, quote length (log); in the app set to a weekday stay of the host's minimum length |

Rule for accepting a feature (CV grouped by host, train + validation only): (a) known before listing and not identifying the host, (b) lowers RMSE(log) by at least 0.001, (c) better in at least 4 of 5 folds, (d) also better on the validation split. Full table: `experiments/results.md`.

- Accepted: title words (−0.0022), upscale wording (−0.0027), private room × guests (−0.0010), maximum nights (−0.0013), breakfast (−0.0010)
- Rejected by the numbers: beds (−0.0003, same information as guests), neighbourhood price level (−0.00099, 3/5 folds), all 128 neighbourhoods (+0.0017), licence, distance², several amenities
- Rejected on principle: booking lead time and check-in month (demand information / only two months), operator brand words (identify the host), nearby prices including the host's own (leak)

Value of each group (error increase when removed): Size +0.076 · Location +0.022 · Quote controls +0.018 · Amenities +0.009 · Room/property +0.005 · Wording +0.004 · Booking rules +0.001.

Log vs raw price target (CV): 20.9% vs 26.1% median error.

Left unused on purpose: gradient-boosted trees would lower the error by about 0.009, all 3,000 text words by about 0.007.

**Your take:** why a strict rule, what surprised you (beds?), and what the unused signal says about a linear model.

## Split and three implementations

Split 70/15/15 grouped by host (many hosts post near-copies). Random split R² 0.676 vs grouped 0.650 over 5 seeds. Preprocessor fitted on train only. Test used once.

Same model three ways, ridge regression on log price:
1. scikit-learn `Ridge`
2. manual PyTorch: tensors, autograd, hand-written update, penalty added to the loss
3. `nn.Module` + `torch.optim.SGD`: penalty as weight decay = 2α/n, weights only (bias excluded)

Tuning: α chosen on validation = 0.1 (flat from 0.1 to 100). Learning rate: stable only below 2/λmax ≈ 0.20; 0.2 and 0.3 diverge; 0.1 reaches sklearn's optimum in 320 epochs (0.03 needs 1,060), so lr = 0.1, 3,000 epochs.

Agreement: largest coefficient gap 6.6e-7, same training objective 0.1270980, predictions at most €0.0009 apart (float32 vs float64).

First version bug: all 21 district dummies + intercept are linearly dependent, so sklearn and the manual loop disagreed on coefficients by up to 0.27. Fix: drop Centro, standardise the dummies (rerun shown in the app).

Test set, 1,967 listings from unseen hosts (median abs % error, 95% host-bootstrap interval):

| | All quotes | Weekday only (1,572) | Within ±20% | R² (log) |
|---|---|---|---|---|
| Baseline A: district × room type median | 27.3% | 26.8% | 37.5% | 0.39 |
| Baseline B: district × room type × guests median | 25.1% | 24.3% | 41.2% | 0.53 |
| Ridge without quote controls | 21.1% | 20.8% | 47.2% | 0.67 |
| Final model (all three implementations) | 20.0% [18.4–21.4] | 19.9% | 49.7% | 0.69 |

Final model minus Baseline B: −3.4 to −6.6 points. Minus the no-quote-controls model: −0.5 to −2.2 points.

**Your take:** why the three agree (what had to be translated), why validation error cannot pick the learning rate, and is a 5-point gain over Baseline B worth it for a host?

## Validity checks

- Dates: quotes that are mostly Fri/Sat ask +28.5% over the model without quote controls, −4.5% with them. Only 3 weekends in the data.
- Location: neighbourhoods explain 3.8% of the leftover error; error correlation with the 10 nearest other-host listings 0.045.
- New listings (0 reviews) ask +7.3% above the model; 100+ reviews −4.5%.
- By actual price: cheap listings look over-predicted (+16%), expensive under-predicted (−23%). A simulated model that is right on average shows the same (+27% to −23%): regression to the mean. By predicted price the errors are small; only the cheapest-predicted 5% repeats on validation (−8% test, −14% validation). Corrections tested and not adopted (stretching raises error; per-band offsets help median error, hurt RMSE).

**Your take:** what these mean for a real user.

## What did not work / limitations

**Your take** (3 to 5 bullets): e.g. the dummy-variable trap, the first version without date controls, features you expected to help, the bias correction you dropped, early-summer only, asking prices not bookings.

## Generative AI use disclosure

I used Claude (Anthropic) for most of the code (cleaning, features, the three implementations, diagnostics, tests, Gradio app) and to structure this report and fill in the results tables. The choices of dataset, business framing and decision rules were mine, made in discussion with it, and the written analysis ("Your take" parts) is my own.
