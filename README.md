# When the Lights Go Out: Analyzing U.S. Power Outage Severity

By Yuki Saeli | DSC 80, UCSD Spring 2026

---

## Introduction

In this project, I examined a dataset of major power outages across the continental U.S. from January 2000 to July 2016. These outages were compiled by Purdue University's Laboratory for Advancing Sustainable Critical Infrastructure and include geographic, climatic, economic, and infrastructure characteristics of each event.

**The central question of this project is: Can we predict the duration of a major power outage given information known at the time the outage begins?**

This question matters because outage duration directly determines how long communities are left without power, driving economic losses, public safety risks, and emergency response costs. If energy companies can predict duration at the moment an outage starts, they can pre-position repair crews, issue accurate restoration estimates, and allocate resources more effectively — before the situation worsens.

The original dataset contains **1,534 rows**, each corresponding to one major outage event, and 57 columns. The columns most relevant to our question are described below:

| Column | Description |
|---|---|
| `OUTAGE.DURATION` | Duration of the outage event in minutes — our prediction target |
| `CAUSE.CATEGORY` | Category of the cause of the outage (e.g., severe weather, intentional attack) |
| `CLIMATE.CATEGORY` | El Niño/La Niña climate episode at the time of the outage (warm, cold, or normal) |
| `U.S._STATE` | U.S. state where the outage occurred |
| `MONTH` | Month the outage began — used to engineer a SEASON feature |
| `CUSTOMERS.AFFECTED` | Number of customers affected by the outage |
| `TOTAL.PRICE` | Average monthly electricity price in the state (cents/kilowatt-hour) |
| `OUTAGE.START` | Combined timestamp of outage start date and time |
| `OUTAGE.RESTORATION` | Combined timestamp of power restoration date and time |

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

We performed the following cleaning steps, each motivated by how the data was originally collected and stored:

1. **Loaded the Excel file with `header=5`** to skip 5 rows of metadata, then dropped the first data row which contained measurement units (e.g., "yr", "min") rather than actual data.

2. **Combined date and time columns into Timestamps.** The outage start date and time were stored in two separate columns (`OUTAGE.START.DATE` and `OUTAGE.START.TIME`). We combined these into a single `pd.Timestamp` column called `OUTAGE.START` using `pd.to_datetime`. The same was done for restoration, creating `OUTAGE.RESTORATION`. The original split columns were dropped.

3. **Selected relevant columns** — we kept only the 12 columns relevant to our research question and prediction task, dropping the remaining 45 columns.

4. **Converted numeric columns** from object dtype to numeric using `pd.to_numeric(..., errors='coerce')`, since Excel files often read numbers as strings.

5. **Replaced 0 values** in `OUTAGE.DURATION` and `CUSTOMERS.AFFECTED` with `NaN`. A true zero would mean an outage lasted 0 minutes or affected 0 customers — neither is meaningful for a major outage event, so these are treated as missing data.

The first few rows of the cleaned DataFrame are shown below:

| YEAR | MONTH | U.S._STATE | CLIMATE.CATEGORY | CAUSE.CATEGORY | OUTAGE.DURATION | CUSTOMERS.AFFECTED | OUTAGE.START |
|---|---|---|---|---|---|---|---|
| 2011 | 7 | Minnesota | normal | severe weather | 3060.0 | 70000.0 | 2011-07-01 17:00:00 |
| 2014 | 5 | Minnesota | normal | intentional attack | 1.0 | NaN | 2014-05-11 18:38:00 |
| 2010 | 10 | Minnesota | cold | severe weather | 3000.0 | 70000.0 | 2010-10-26 20:00:00 |
| 2012 | 6 | Minnesota | normal | severe weather | 2550.0 | 68200.0 | 2012-06-19 04:30:00 |
| 2015 | 7 | Minnesota | warm | severe weather | 1740.0 | 250000.0 | 2015-07-18 02:00:00 |

### Univariate Analysis

The distribution of outage duration is heavily right-skewed. Most outages last under 5,000 minutes (~3.5 days), but a small number of extreme events extend well beyond 20,000 minutes (~2 weeks). This skew motivates the log-transformation of related features in our model.

<iframe
  src="assets/outage_duration_dist.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

Severe weather is by far the most common cause of major outages, accounting for nearly half of all events. Intentional attacks are the second most frequent cause.

<iframe
  src="assets/cause_counts.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

### Bivariate Analysis

Fuel supply emergencies have the longest and most variable durations. Intentional attacks resolve the fastest. Severe weather shows many extreme outliers despite a low median — meaning most weather outages are short, but a few can last weeks. This pattern suggests cause category is a strong predictor of outage duration.

<iframe
  src="assets/duration_by_cause.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

There is a loose positive relationship between customers affected and outage duration, but the pattern varies heavily by cause category, confirming that cause plays an important moderating role.

<iframe
  src="assets/customers_vs_duration.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

### Interesting Aggregates

The table below shows average outage severity metrics grouped by cause category, sorted by mean duration. Fuel supply emergencies stand out as the most severe on average, while intentional attacks are the least severe despite being the second most common cause.

| Cause Category | Mean Duration (min) | Median Duration (min) | Mean Customers Affected | Count |
|---|---|---|---|---|
| fuel supply emergency | 10226.5 | 5970.5 | 212583.7 | 38 |
| severe weather | 3659.7 | 2250.0 | 189533.6 | 741 |
| public appeal | 1980.3 | 1320.0 | 143950.0 | 69 |
| equipment failure | 1810.4 | 540.0 | 119952.6 | 35 |
| system operability disruption | 899.5 | 240.0 | 211923.7 | 83 |
| intentional attack | 429.4 | 283.0 | 3567.2 | 418 |
| islanding | 200.6 | 95.5 | 34169.3 | 14 |

---

## Assessment of Missingness

### NMAR Analysis

Several columns contain missing values, but `CUSTOMERS.AFFECTED` (655 missing, ~43%) is likely **NMAR (Not Missing At Random)**. Utility companies self-report outage data to federal agencies, and smaller outages affecting fewer customers are less likely to be formally logged. This means the missingness is tied to the actual value itself — low customer counts go unreported — making it NMAR rather than MAR or MCAR.

To make this column MAR, we would need additional data such as each utility company's internal reporting threshold — the minimum number of customers required before a count gets logged. With that information, we could condition on reporting behavior and explain the missingness without needing the missing value itself.

### Missingness Dependency

We tested whether the missingness of `CUSTOMERS.AFFECTED` depends on other observed columns using permutation tests.

**Test 1: Does missingness depend on `CAUSE.CATEGORY`?**

We used TVD (Total Variation Distance) as the test statistic since `CAUSE.CATEGORY` is categorical. The observed TVD was 0.7558, and the p-value was 0.0000 — we reject H₀. The missingness of `CUSTOMERS.AFFECTED` depends on cause category: certain outage types (e.g., small equipment failures) are systematically less likely to have customer counts recorded than others (e.g., major weather events).

<iframe
  src="assets/missingness_test.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

**Test 2: Does missingness depend on `TOTAL.PRICE`?**

We used the absolute difference in group means as the test statistic since `TOTAL.PRICE` is numeric. The observed difference was 0.3045, and the p-value was 0.0390 — we reject H₀. The missingness also depends on electricity price, suggesting states with different economic characteristics have different reporting behaviors.

<iframe
  src="assets/missingness_price.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

Both tests suggest the missingness of `CUSTOMERS.AFFECTED` is **MAR** — it can be explained by other observed columns rather than being purely random.

---

## Hypothesis Testing

**Question:** Do weather-related outages last significantly longer than non-weather-related outages?

This question is motivated by the box plot above, which suggested severe weather outages have longer durations and more extreme outliers. We formally test this before using `CAUSE.CATEGORY` as a model feature.

- **Null Hypothesis (H₀):** Weather-related and non-weather-related outages have the same mean duration. Any observed difference is due to random chance.
- **Alternative Hypothesis (H₁):** Weather-related outages have a longer mean duration than non-weather-related outages.
- **Test statistic:** Difference in group means (weather mean − non-weather mean)
- **Significance level:** α = 0.05
- **Method:** Permutation test (1,000 shuffles)

Weather-related outages had a mean duration of 3,899.7 minutes vs. 1,499.9 minutes for non-weather outages — an observed difference of **2,399.9 minutes**. In 1,000 permutations, no simulated statistic came close to this value.

<iframe
  src="assets/hypothesis_test.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

**p-value = 0.0000.** We reject H₀. The data is consistent with weather-related outages lasting significantly longer. This is not a randomized controlled trial — we cannot claim causation — but the result strongly justifies including `CAUSE.CATEGORY` as a feature in our prediction model.

---

## Framing a Prediction Problem

**Prediction task:** Predict `OUTAGE.DURATION` (in minutes) — a **regression** problem.

We chose `OUTAGE.DURATION` as our response variable because it is the most direct measure of outage severity. It determines how long communities are left without power and drives economic and safety impacts downstream.

At the moment an outage begins, a utility company would know:
- `CAUSE.CATEGORY` — the reported cause is identified within the first hours
- `CLIMATE.CATEGORY` — climate conditions are known before the outage
- `U.S._STATE` — location is immediately known
- `MONTH` — calendar month is always known (used to engineer SEASON)
- `CUSTOMERS.AFFECTED` — estimated from outage scope and SCADA data at onset

We would **NOT** know `OUTAGE.DURATION` at prediction time — that is what we are trying to predict. We also would not know `OUTAGE.RESTORATION`, since that is the endpoint we're estimating.

**Evaluation metric:** RMSE (Root Mean Squared Error), measured in minutes. We chose RMSE over R² because it is in the same units as our target, making it more interpretable. It also penalizes large errors heavily, which matters since severely underestimating a long outage has serious operational consequences.

---

## Baseline Model

Our baseline uses **Linear Regression** in a single sklearn `Pipeline` with two features:

- `CAUSE.CATEGORY` — **nominal**, one-hot encoded using `OneHotEncoder`
- `CUSTOMERS.AFFECTED` — **quantitative**, median imputed using `SimpleImputer`

These two features were chosen as the most directly observable at outage onset. The pipeline handles all preprocessing internally to prevent data leakage — the imputer and encoder are fit only on training data.

| Metric | Value |
|---|---|
| Train RMSE | 5,516.99 minutes |
| Test RMSE | 7,614.74 minutes |
| Mean outage duration | 2,771.88 minutes |

The test RMSE of 7,614 minutes is nearly **3x the mean outage duration**, indicating the model is a poor predictor. Linear Regression cannot capture the non-linear interactions between cause, climate, and geography that likely drive outage duration. This motivates our final model.

---

## Final Model

We upgraded to **RandomForestRegressor** with four features, all in a single sklearn Pipeline:

**New features engineered:**
- `SEASON` (from `MONTH`): Winter storms and summer heat waves cause longer outages than other seasons. Grouping months into seasons captures this cyclical pattern better than raw month numbers, which treat December and January as far apart.
- `LOG_CUSTOMERS` (log of `CUSTOMERS.AFFECTED`): Customer counts are heavily right-skewed. Log-transforming compresses extreme values and reduces the influence of outliers on tree splits.

**Features retained from baseline:**
- `CAUSE.CATEGORY` — one-hot encoded
- `CLIMATE.CATEGORY` — one-hot encoded (added to final model)

We used `GridSearchCV` with 5-fold cross validation to tune:
- `n_estimators` ∈ {100, 200}: number of trees — more trees reduce variance but increase computation
- `max_depth` ∈ {3, 5, 10, None}: controls overfitting — shallower trees generalize better

**Best hyperparameters:** `max_depth=5`, `n_estimators=200`

| Model | Train RMSE | Test RMSE | Overfit Gap |
|---|---|---|---|
| Baseline (LinearRegression) | 5,516.99 min | 7,614.74 min | 2,097 min |
| Final (RandomForestRegressor) | 4,558.26 min | 7,010.52 min | 2,452 min |
| **Improvement** | | **604.22 min (~7.9%)** | |

The final model improved test RMSE by 604 minutes over the baseline. `RandomForestRegressor` captures non-linear interactions between features that `LinearRegression` cannot model. We also experimented with including `U.S._STATE` as a feature, but found it caused severe overfitting (train RMSE 2,622, test RMSE 7,381, gap of 4,759 min) — removing it improved generalization significantly.

---

## Fairness Analysis

**Question:** Does the model perform worse for outages in extreme climate states (warm or cold El Niño/La Niña conditions) compared to normal climate states?

- **Group X (extreme):** `CLIMATE.CATEGORY` = `'warm'` or `'cold'`
- **Group Y (normal):** `CLIMATE.CATEGORY` = `'normal'`
- **Evaluation metric:** RMSE
- **Test statistic:** Absolute difference in RMSE between the two groups
- **Null Hypothesis (H₀):** The model is fair. Its RMSE for extreme and normal climate groups is roughly the same — any difference is due to random chance.
- **Alternative Hypothesis (H₁):** The model is unfair. Its RMSE for extreme climate states is higher than for normal climate states.
- **Significance level:** α = 0.05
- **Method:** Permutation test (1,000 shuffles)

| Group | RMSE |
|---|---|
| Extreme climate (warm/cold) | 6,164.88 minutes |
| Normal climate | 8,383.15 minutes |
| Observed difference | 2,218.27 minutes |
| **P-value** | **0.6620** |

<iframe
  src="assets/fairness_test.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

With a p-value of **0.6620 > 0.05**, we **fail to reject H₀**. The observed RMSE difference between extreme and normal climate groups is not statistically significant — it falls well within the range we would expect by random chance alone. We cannot conclude that the model is unfair with respect to climate category. Interestingly, the model actually performs better on extreme climate outages (RMSE 6,164) than normal climate outages (RMSE 8,383), suggesting normal climate outages have more varied and harder-to-predict causes.