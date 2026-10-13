# How the features were chosen

Produced by `experiments/feature_experiments.py`. All numbers come from training + validation listings (12,756); the test set is never used for these decisions.

**Error** = RMSE of log price: the typical size of a prediction mistake, with big mistakes counting extra. Lower is better; 0.001 is a small change, 0.01 a large one. **Error change** is measured against the starting model, averaged over 5 rounds in which 4/5 of the hosts train the model and the other 1/5 are predicted (no host ever appears on both sides).

## The procedure

1. **Starting model**, chosen by reasoning before any test: location (distance to Puerta del Sol, district), size (guests, bedrooms, bathrooms), room and property type, 16 amenity concepts, minimum nights, plus the two quote controls that remove the date effect from the target (error 0.3665; without the quote controls 0.3846).
2. **Each candidate is added on its own** and kept only if it passes all four parts of the rule:
   - **a. Legitimate:** known before the listing goes live, and does not identify a particular host.
   - **b. Big enough:** lowers the error by at least 0.001 on average.
   - **c. Consistent:** lowers it in at least 4 of the 5 rounds.
   - **d. Confirmed:** also lowers it on the validation split (model trained on the training split).
3. **Final model** = starting model + every accepted candidate, checked together.

**Where the candidates came from.** Most were reasoned out in advance (title words, beds, the two interactions, distance squared, description length, licence, neighbourhoods, nearby prices) or found by going through the unused columns of the raw file (maximum nights). The others (upscale and budget words, the seasonal-rental and lease wording, and the six single amenities) came from inspecting which words and amenities were associated with the model's errors on the **training split**. Because that inspection used the training split, check d scores every candidate on the validation split, which played no part in it.

## Accepted

| Candidate | What it measures | Error change | Better in | Validation split | Decision |
|---|---|---|---|---|---|
| Title words (6 flags) | Title mentions luxury/design, penthouse/ático, terrace, view, studio, or cozy/small | -0.0022 | 4 of 5 | improves | Accepted |
| Upscale words in title or description | Luxury, design, stylish, contemporary, prestigious, elegant, exclusive, premium, boutique | -0.0027 | 5 of 5 | improves | Accepted |
| Private room × guests | Lets the number of guests count differently for a private room than for a whole flat | -0.0010 | 4 of 5 | improves | Accepted (narrowly: gain 0.00104) |
| Maximum nights | Longest stay the host allows (log) | -0.0013 | 4 of 5 | improves | Accepted |
| Amenity: breakfast | Breakfast is listed as an amenity | -0.0010 | 4 of 5 | improves | Accepted (narrowly: gain 0.00103) |

**All accepted together:** error 0.3665 → 0.3596 (-0.0068, better in 5 of 5 rounds; validation split improves).

## Rejected: did not pass the rule

| Candidate | What it measures | Error change | Better in | Decision |
|---|---|---|---|---|
| Beds | Number of beds | -0.0003 | 4 of 5 | Rejected: gain too small; worse on the validation split |
| Description length | Number of characters in the description (log) | -0.0001 | 4 of 5 | Rejected: gain too small |
| Budget words | Description mentions simple, basic, budget, cheap | -0.0002 | 3 of 5 | Rejected: gain too small; better in only 3 of 5 rounds |
| Seasonal-rental words (broad) | Description mentions temporary/seasonal rental, lease, contract | -0.0002 | 4 of 5 | Rejected: gain too small |
| Lease agreement required (narrow) | Description says a lease or rental contract must be signed (a narrower second definition, tried once after the broad one failed) | -0.0008 | 4 of 5 | Rejected: gain 0.00076 is below 0.001; worse on the validation split |
| Amenity: safe | A safe is listed | -0.0005 | 4 of 5 | Rejected: gain 0.00052 is below 0.001; worse on the validation split |
| Amenity: building staff | Building staff or doorman is listed | -0.0005 | 4 of 5 | Rejected: gain too small |
| Amenity: private living room | A private living room is listed | -0.0006 | 4 of 5 | Rejected: gain 0.00064 is below 0.001 |
| Amenity: family equipment | High chair, crib, babysitter tips or children's items | -0.0004 | 2 of 5 | Rejected: gain too small; better in only 2 of 5 rounds |
| Amenity: lock on bedroom door | Typical of rooms rented inside a shared flat | -0.0002 | 4 of 5 | Rejected: gain too small; worse on the validation split |
| Tourist licence status | Registered tourist licence, seasonal-rental exemption, or none | +0.0002 | 2 of 5 | Rejected: makes the error worse; better in only 2 of 5 rounds |
| Distance to Sol, squared | Lets the distance effect curve more | -0.0002 | 3 of 5 | Rejected: gain too small; better in only 3 of 5 rounds; worse on the validation split |
| Private room × distance | Lets distance count differently for private rooms | +0.0001 | 3 of 5 | Rejected: makes the error worse; better in only 3 of 5 rounds; worse on the validation split |
| All 128 neighbourhoods | Each of the 128 neighbourhoods as its own category, on top of the 21 districts | +0.0017 | 1 of 5 | Rejected: makes the error worse; better in only 1 of 5 rounds; worse on the validation split |
| Neighbourhood price level | Average price of the neighbourhood, pulled toward its district average | -0.0010 | 3 of 5 | Rejected: gain 0.00099 is below 0.001; better in only 3 of 5 rounds; worse on the validation split |
| Prices of the 25 nearest listings | Median price of the 25 nearest similar listings from other hosts | -0.0007 | 5 of 5 | Rejected: gain 0.00074 is below 0.001 |

Some words looked strong when inspected, notably "corta duración" (short-term rental), which went with prices about 30% above the model. It turned out to appear in 134 listings from only 9 hosts: template wording of a few operators. Such signals do not carry over to hosts the model has not seen, which is exactly what the host-grouped rounds test, so the seasonal-rental candidates fail.

## Rejected on principle (rule part a), whatever they score

| Candidate | Error change | Better in | Why excluded |
|---|---|---|---|
| Lead time (days until the quoted check-in) | -0.0025 | 5 of 5 | Inside Airbnb quotes the first FREE stay, so a late date means a booked-up listing: demand information nobody has before listing |
| Check-in month | -0.0026 | 5 of 5 | 94% of quotes are June/July check-ins, so other months cannot be learned |
| Operator brand words (waou, dobo, corta duración) | -0.0011 | 4 of 5 | Template wording of about 10 professional operators: identifies the host |
| Prices of the 25 nearest listings, own host included | +0.0039 | 0 of 5 | Leaks the answer: the listing's own price and its host's other prices are in the feature |

## How much signal is left unused? (diagnostics, compared with the final model)

| What was tried | Error change vs final model | What it means |
|---|---|---|
| Every word and word pair in title + description (3,000 columns) | -0.0074 | Real extra signal, but 3,000 unnamed columns make the model unreadable, and part of it is operator template wording |
| Every amenity listed by at least 1% of listings (142 columns) | -0.0030 | The individual amenities that pass the rule were tested above; the rest is many tiny effects |
| Gradient-boosted trees on the final features | -0.0092 | How much curves and interactions could add; the assignment requires a linear model |
| Pre-discount price as the target | +0.0020 | 7.4% of quotes carry a discount (brand-new listings 23%, others 6%); the price after discount predicts at least as well, so the target stays as it is |
