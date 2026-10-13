# Assignment 1 Report

- **Name**: Constantin Hatecke
- **Student ID**: 20241
- **Email**: chatecke.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

**How to run.** `uv run python main.py train` (about 1 minute, raw data to saved models), then `uv run python main.py app`. Optional: `uv run pytest -q` (34 checks), `uv run python experiments/feature_experiments.py` (reproduces `experiments/results.md`).

**Code.** `clean_data.py` cleaning · `features.py` features · `pipeline.py` split, preprocessing, baselines, the three implementations · `diagnostics.py` evidence and validity checks · `train.py` runs it all · `app.py` dashboard · `experiments/` feature selection.

## Dataset

I chose an Airbnb dataset that contains a listing export for Madrid, scraped between 20 June and 2 July 2026. One row is one Airbnb listing as it appeared on the scrape date: 22,708 rows and 90 columns.

Cleaning: Of the original 22,708 rows, many were not applicable because of missing data or listings that weren’t actually Airbnbs. We also removed stays longer than 7 days because they are rare, and the applied discount would negatively impact the model’s accuracy.

| Cleaning step | Rows |
|---|---|
| Raw | 22,708 |
| No price | −3,536 |
| Price quoted for a stay longer than 7 nights | −4,095 |
| Hotel / shared rooms | −164 |
| Implausible price (< €20 or placeholder) | −190 |
| **Final** | **14,723** |

Fixes instead of drops: blank bedrooms of private rooms set to 1 (2,363), private rooms showing the whole flat's bedrooms set to 1 (190), "0 baths" on entire homes set to unknown (61). No imputation in cleaning; gaps are filled only from the training split.

**Why this dataset** Pricing a flat is a real decision many new Airbnb hosts struggle with. Additionally, it was important that the data forces genuine feature work, which it does. As an Airbnb customer, I was also interested in what drives property prices and how they're set.

## Business framing

Decision supported: the starting weekday price for a Madrid flat not yet listed, and which characteristics drive a higher price. Airbnb's Smart Pricing only works once a listing is live. This model gives new landlords a starting point to maximize revenue from the start. 

How it drove the pipeline:
- only information known before listing (no reviews, no booking lead time)
- split grouped by host
- metrics in %: median absolute % error and share within ±20%
- the app prices a weekday stay, so the test set is also scored on weekday-only quotes

For inexperienced landlords, this is a very important product. Getting the price wrong could also mean the listing goes unbooked while they are away, especially if they list it for a short time, e.g., 2 weeks while they are on holiday. This helps avoid that. Using percentages instead of € also helps, as it works across cheaper and more expensive bookings. 

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
- Rejected by the numbers: beds (−0.0003, same information as guests), neighborhood price level (−0.00099, 3/5 folds), all 128 neighborhoods (+0.0017), license, distance², several amenities
- Rejected on principle: booking lead time and check-in month (demand information / only two months), operator brand words (identify the host), nearby prices including the host's own (leak)

Value of each group (error increase when removed): Size +0.076 · Location +0.022 · Quote controls +0.018 · Amenities +0.009 · Room/property +0.005 · Wording +0.004 · Booking rules +0.001.

Log vs raw price target (CV): 20.9% vs 26.1% median error.

Left unused on purpose: gradient-boosted trees would lower the error by about 0.009, all 3,000 text words by about 0.007.

The implementation of a strict rule was used as there were many features that did not add much to the model. There’s a chance it looks helpful just by luck in that particular data split. The fixed rule decided before testing, with a minimum gain, most folds, and a separate validation check, filters out the lucky ones. Interestingly, the number of beds didn't add much to the model, but it turns out it was because it was heavily correlated with the number of guests, which was already in the model. Using a linear model lets us show the landlord which aspects of their apartment drive the price. This means they can identify ways to change the apartment to increase its value.  

## Split and three implementations

Split 70/15/15 grouped by host (many hosts post near-copies). Random split R²: 0.676 vs. grouped: 0.650 over 5 seeds. Preprocessor fitted on train only. Test set used once.

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

The three models agree because they all solve exactly the same math problem. They find the weights that minimize the squared error, plus a small penalty for large weights. Sklearn solves it directly. The two PyTorch versions walk downhill step by step until they reach the same bottom. For them to land on the same bottom, two things had to be translated:

sklearn writes the penalty as α, but PyTorch defines weight decay differently, so it had to be set to 2α/n.
The penalty must hit the weights only, not the bias. Otherwise, PyTorch would shrink the intercept too and land somewhere slightly different
After that, results matched to the 7th decimal, which is just due to the rounding. 

The validation error couldn't pick the learning rate because it was the same bottom. So, as long as the learning rate didn't overshoot, it gives an identical prediction, meaning that validation can't tell them apart. The only difference is how quickly they get there. 

Baseline B is what a careful host could do by hand without any model with a simple Airbnb search of similar apartments in the area. That is why I used it as a baseline. The test shows that this baseline was around 25% off the real price while the model was around 20% off. The 5% is what even the smart landlords are able to gain from this model; however, all landlords can achieve this now without a problem. Additionally they recieve an explination for the price which they could not get on their own. 
## Validity checks

- Dates: quotes that are mostly Fri/Sat ask +28.5% over the model without quote controls, −4.5% with them. Only 3 weekends in the data.
- Location: neighbourhoods explain 3.8% of the leftover error; error correlation with the 10 nearest other-host listings 0.045.
- New listings (0 reviews) ask +7.3% above the model; 100+ reviews −4.5%.
- By actual price: cheap listings look over-predicted (+16%), expensive under-predicted (−23%). A simulated model that is right on average shows the same (+27% to −23%): regression to the mean. By predicted price the errors are small; only the cheapest-predicted 5% repeats on validation (−8% test, −14% validation). Corrections tested and not adopted (stretching raises error; per-band offsets help median error, hurt RMSE).

The validity-check results are communicated to the user, so they know they can charge more on weekends. However, this is already a listing mechanism in Airbnb, meaning that the user does not have to manually adjust the rates. 
The location check validates that the location information the model uses (distance to Sol plus district) captures almost all of the location effect the data contains. The mistakes left are about the flat itself, like quality, view or renovation, not where it is.
The new listing check shows that new landlords struggle to price their listings and are very optimistic about what they can earn. Additionally, the lack of reviews makes their place even less likely to be booked. 
When I grouped the test listings by their real price, I noticed that cheap listings look over-predicted and expensive under-predicted. At first I thought it was bias. But after checking, I found that it was actually explained by regression to the mean, which showed that the pattern comes from sorting listings by their real price, not from the model doing something wrong. When I grouped by the predicted price instead, the errors were small, and only the cheapest 5% stood out (about −8%). I tested corrections but did not use them, because stretching the predictions made the overall error worse and the per-band fix improved one measure but hurt the

## What did not work / limitations
The error pattern by price level looked like bias but wasn’t. An attempt was made to correct it, and it was dropped because it made RMSE worse.
The first model ignored the quoted dates. Weekend quotes inflated the weekday price by about 5%. It was fixed by adding the two quote controls.
Beds didn’t help, even though you expected it to be the best predictor. It carries the same information as guests.

Limitations:
Only June and July data, so the dataset excludes other seasons. Since these are holiday seasons, we can assume prices will be quite different for the rest of the year. 
Unseen quality is a big factor. Photos, renovation, floor and view aren’t in the data, and that’s where most of the remaining error comes from.
There are missing costs. Cleaning fees and discounts aren’t fully included in the target.
The data is only from one scrape, a snapshot of a single moment, so the model needs retraining as the market changes.
Airbnb shifts each listing’s location by up to about 150 m for privacy, this means that the location data could be more accurate.

## Generative AI use disclosure

I used Claude for most of the code (cleaning, features, the three implementations, diagnostics, tests, Gradio app) and to structure this report and fill in the results tables. The choices of dataset, business framing and decision rules were mine, at times made in discussion with the AI to get feedback, and the written analysis is my own.
