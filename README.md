# Overview

Customer Intelligence & Support Copilot: a fictional Instamart capstone using synthetic data. Parts A–D are implemented. The foundation audits and cleans customer, product and order exports, stores them in SQLite, and performs seven SQL analyses with pandas cross-checks. Part B segments customers and predicts lapse using pre-cutoff behaviour; Part C masks and routes support tickets with guarded offline model calls; Part D answers policy questions and runs a bounded, resumable simulated refund agent. Money is INR throughout.

This repository contains the code, data, SQL, required outputs and checkpoints for Parts A–D. All paths below are relative to the repository root. The supplied `generate_data.py` and `mock_llm.py` are retained unchanged. Model calls run offline; only dependency installation needs internet access.

# Reproduce

Use Python 3.12 (the tested version). Run these numbered PowerShell commands from the repository root; activation is unnecessary.

1. `python -m venv .venv`
2. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
3. `.\.venv\Scripts\python.exe -B generate_data.py`
4. `.\.venv\Scripts\python.exe -B part_a.py`
5. `.\.venv\Scripts\python.exe -B validate_part_a.py`
6. `.\.venv\Scripts\python.exe -B part_b.py`
7. `.\.venv\Scripts\python.exe -B validate_part_b.py`
8. `.\.venv\Scripts\python.exe -B part_c.py`
9. `.\.venv\Scripts\python.exe -B validate_part_c.py`
10. `.\.venv\Scripts\python.exe -B part_d.py`
11. `.\.venv\Scripts\python.exe -B validate_part_d.py`

On macOS/Linux, use `.venv/bin/python` in steps 2–11. Package versions in `requirements.txt` were captured with `pip freeze`; the additional packages are dependencies of pandas, scikit-learn and joblib.

The generator prints `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The validators print `ALL CHECKPOINTS PASSED` for A and `ALL PART B CHECKPOINTS PASSED`, `ALL PART C CHECKPOINTS PASSED`, `ALL PART D CHECKPOINTS PASSED` for B–D.

Supplementary reports are regenerated in `.reports/`, which Git ignores. Run each Part before its validator. Required artifacts stay in `data/`, `outputs/` and `state/`; Part D appends new runs and checkpoints on each execution. Do not upload `.venv/`, `.reports/` or bytecode caches. No files from the parent directory are needed.

Submit this repository on the single branch `main`, using a public link ending in `/tree/main`. Keep the required database and model file: the brief explicitly requires them despite its text-only reporting rule.

# Part A

## Raw audit

The table below reports shape, inferred dtype, missing count and missing percentage for every column. Missing percentages use each table's row count as denominator. Tickets are generated now but are not processed in Part A; later ticket reads must use `keep_default_na=False`.

| table | rows | columns | column | dtype | missing | missing_pct |
| --- | --- | --- | --- | --- | --- | --- |
| customers | 1200 | 5 | customer_id | str | 0 | 0 |
| customers | 1200 | 5 | city | str | 0 | 0 |
| customers | 1200 | 5 | signup_date | str | 0 | 0 |
| customers | 1200 | 5 | age | float64 | 61 | 5.0833 |
| customers | 1200 | 5 | gender | str | 30 | 2.5 |
| products | 40 | 4 | product_id | str | 0 | 0 |
| products | 40 | 4 | product_name | str | 0 | 0 |
| products | 40 | 4 | category | str | 0 | 0 |
| products | 40 | 4 | price_inr | int64 | 0 | 0 |
| orders | 13084 | 9 | order_id | str | 0 | 0 |
| orders | 13084 | 9 | customer_id | str | 0 | 0 |
| orders | 13084 | 9 | product_id | str | 0 | 0 |
| orders | 13084 | 9 | quantity | int64 | 0 | 0 |
| orders | 13084 | 9 | order_ts | str | 0 | 0 |
| orders | 13084 | 9 | status | str | 0 | 0 |
| orders | 13084 | 9 | discount_pct | int64 | 0 | 0 |
| orders | 13084 | 9 | order_amount | str | 53 | 0.4051 |
| orders | 13084 | 9 | delivery_minutes | float64 | 2541 | 19.4207 |

Specific raw-data checks:

| Measure | Value |
| --- | --- |
| raw_orders | 13084 |
| exact_duplicates | 119 |
| pending | 656 |
| unique_orders | 12309 |
| nonnumeric_amounts | 120 |
| negative_delivery | 106 |
| blank_delivery | 2541 |
| blank_age | 61 |
| invalid_age | 6 |
| raw_cities | 24 |
| raw_genders | 10 |
| blank_gender | 30 |

## Cleaning rules and measured changes

Customer cities are stripped and title-cased. Gender spellings are normalised to male/female, with blanks becoming unknown. Ages outside 10–100 become missing; all missing ages are filled using the within-city median via `groupby().transform()`. Signup dates are parsed as datetimes (1,200 values).

Orders are processed in the prescribed sequence: parse timestamps (13,084 values), remove exact duplicate rows, then retain the latest timestamp per order identifier. Non-numeric amounts become missing and are reconstructed as `round(price_inr * quantity * (1 - discount_pct / 100), 2)`. Negative delivery times become missing; completed orders receive the completed-order category median via `groupby().transform()`. Cancelled orders retain missing delivery times. Products require no cleaning.

| Measure | Value |
| --- | --- |
| city_values_standardised | 476 |
| gender_values_standardised | 950 |
| ages_imputed | 67 |
| exact_duplicates_removed | 119 |
| older_order_records_removed | 656 |
| amounts_reconstructed | 112 |
| negative_delivery_cleared | 105 |
| completed_delivery_imputed | 774 |

Final customer city counts:

| Measure | Value |
| --- | --- |
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

Final gender counts:

| Measure | Value |
| --- | --- |
| female | 610 |
| male | 560 |
| unknown | 30 |

Final table and revenue checks:

| Measure | Value |
| --- | --- |
| customers | 1,200 |
| products | 40 |
| orders | 12,309 |
| completed | 11,110 |
| cancelled | 1,199 |
| revenue | 2,998,668.8 |

All ages are present and between 18 and 60. There are no pending rows or missing amounts. Completed orders have no missing delivery times; all 1,199 cancelled orders have missing delivery times. Clean CSVs are under `data/clean/`; `data/instamart.db` contains customers, products and orders, with timestamps stored as `YYYY-MM-DD HH:MM:SS`. Validation checks the actual persisted CSVs and database, foreign-key coverage, audit figures, SQL results and outlier counts.

## SQL findings

Queries are in `sql/queries.sql`; all are executed with `pd.read_sql_query`, with complete result tables saved as `.reports/q1.csv` through `.reports/q7.csv`. Spend and revenue use only completed orders.

| Query | Result |
| --- | --- |
| Q1: WHERE then HAVING | 30 customers have more than INR 8,000 from completed orders individually above INR 500. |
| Q2: three-table join | 30 city/category rows; total INR 2,998,668.80; largest cell Bengaluru/Electronics, INR 233,909.45. |
| Q3: scalar subquery | 11 products priced above the catalogue average. |
| Q4: LEFT JOIN | 25 customers without a completed order. |
| Q5: two CTEs | 550 customers above mean completed-customer spend of INR 2,552.06. |
| Q6: ranking windows | 1,175 customers ranked within city; 8 rows have RANK different from ROW_NUMBER. |
| Q7: running total | 183 daily rows; final cumulative revenue INR 2,998,668.80. |

Q6 leaders (spend in INR):

| city | customer_id | total_spend |
| --- | --- | --- |
| Bengaluru | C0082 | 12,856.2 |
| Chennai | C0930 | 10,151.4 |
| Delhi | C0291 | 13,742.6 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.1 |

Ties share a RANK and DENSE_RANK while ROW_NUMBER assigns distinct positions, so tied customers can have a RANK different from ROW_NUMBER; RANK also leaves gaps after ties.

## pandas cross-check and exploratory analysis

The pivot table in `.reports/revenue_pivot.csv` uses margins and its grand total is asserted equal to SQL Q2: INR 2,998,668.80. Named aggregations by city (sum and mean in INR):

| city | sum | mean | count |
| --- | --- | --- | --- |
| Bengaluru | 523,643.4 | 276.4749 | 1894 |
| Chennai | 463,629.15 | 272.723 | 1700 |
| Delhi | 502,829.8 | 243.6191 | 2064 |
| Hyderabad | 528,589.9 | 284.9541 | 1855 |
| Mumbai | 479,058.5 | 296.9985 | 1613 |
| Pune | 500,918.05 | 252.4789 | 1984 |

Completed-order amount statistics (monetary measures in INR):

| Measure | Value |
| --- | --- |
| mean | 269.9072 |
| median | 135 |
| skewness | 4.6088 |
| q1 | 64.8 |
| q2 | 135 |
| q3 | 297 |
| lower_fence | -283.5 |
| upper_fence | 645.3 |
| below_fence | 0 |
| above_fence | 897 |

Correlation matrix over completed orders:

| feature | order_amount | discount_pct | quantity | delivery_minutes |
| --- | --- | --- | --- | --- |
| order_amount | 1 | -0.0951 | 0.0873 | -0.0031 |
| discount_pct | -0.0951 | 1 | -0.0004 | -0.0098 |
| quantity | 0.0873 | -0.0004 | 1 | -0.0126 |
| delivery_minutes | -0.0031 | -0.0098 | -0.0126 | 1 |

**Interpretation (under 200 words).** This synthetic dataset describes customers, catalogue products and grocery-delivery orders across six cities from April through September 2025. Completed order amounts are strongly right-skewed: the mean is INR 269.91, compared with a median of INR 135.00, and skewness is 4.61. The IQR rule flags 897 orders above INR 645.30 and none below the lower fence. High amounts can reflect legitimate expensive products rather than errors, so these orders are retained. Raw duplicates, invalid ages, non-numeric amounts and negative delivery times are documented data-quality anomalies. Hyderabad contributes the largest city revenue (INR 528,589.90). Among the four measured numeric variables, order amount and discount percentage have the largest absolute correlation, approximately −0.095; amount and quantity are approximately +0.087. These relationships are weak, and correlation of this size does not establish causation.

# Part B

Implemented by [part_b.py](<part_b.py>) and verified by [validate_part_b.py](<validate_part_b.py>). Random choices use seed 42. No charts are generated or committed.

## RFM and segmentation

RFM uses completed orders with snapshot `2025-10-01 00:00:00`. Recency uses whole elapsed days, frequency counts distinct orders, and monetary value sums completed amounts in INR. Customers without a completed order have no segment row.

| Measure | Value |
| --- | --- |
| customers | 1,175 |
| recency_median | 19 |
| recency_min | 0 |
| recency_max | 180 |
| frequency_mean | 9.4553 |
| frequency_max | 33 |
| monetary_mean | 2,552.0586 |
| monetary_max | 14,766.55 |

K-means uses StandardScaler on the three RFM columns, `n_init=10`, `random_state=42`. Results are saved in `.reports/cluster_comparison.csv`:

| k | inertia | silhouette |
| --- | --- | --- |
| 2 | 1,820.7606 | 0.465 |
| 3 | 1,268.2346 | 0.4452 |
| 4 | 743.0711 | 0.5255 |
| 5 | 617.3666 | 0.4805 |
| 6 | 503.1706 | 0.4819 |
| 7 | 408.4605 | 0.4163 |
| 8 | 366.5035 | 0.4123 |

We choose **K=4**: the elbow from three to four clusters is substantial, and its silhouette is the strongest among the permitted K=3/4/5 choices. Four groups also support distinct service, repeat-purchase, occasional-shopper and win-back actions. Increasing K gives the optimiser more centres and therefore more capacity to reduce within-cluster squared distances; the observed K=8 inertia is lower than K=2. Lower inertia alone cannot choose K because adding clusters can keep improving fit while making segments less useful, and it does not measure separation between groups.

Segment profiles (recency in days, monetary value in INR):

| segment_id | size | mean_recency | mean_frequency | mean_monetary | segment_name | recommended_action |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 436 | 28.8624 | 3.7729 | 1,057.6182 | Occasional Shoppers | Encourage a second routine purchase with relevant reminders. |
| 1 | 138 | 24.1159 | 6.7681 | 7,129.6957 | Premium | Offer priority service and tailored high-value retention. |
| 2 | 419 | 8.7661 | 19.4033 | 3,410.0205 | Frequent Essentials | Reward repeat purchases with essentials bundles. |
| 3 | 182 | 109.044 | 2.2033 | 685.9926 | Dormant | Test a capped win-back offer and measure response. |

Cluster numbers are labels with no ordering. Premium is exactly the segment with the highest mean monetary value (segment ID 1). Names are assigned from measured profiles, rather than assuming that a particular numeric cluster ID means Premium.

PCA with two components on scaled RFM explains **0.6343** and **0.2180** of variance, total **0.8523**. PCA is descriptive and does not enter the lapse model. `outputs/customer_segments.csv` contains `customer_id,recency,frequency,monetary,segment_id,segment_name,is_premium`.

## Lapse cohort, features and leakage controls

The cohort includes customers with a completed order before `2025-09-01`. A customer is lapsed (positive class 1) when they have no completed order from `2025-09-01` through `2025-09-30`. The full dataset and split membership are stored in `.reports/lapse_cohort.csv`.

| Measure | Value |
| --- | --- |
| customers | 1156 |
| lapsed | 443 |
| lapse_pct | 38.3218 |
| training_rows | 924 |
| test_rows | 232 |
| train_class_counts | {'0': 570, '1': 354} |
| test_class_counts | {'0': 143, '1': 89} |

Features are recency, frequency, monetary, average discount percentage, cancellation rate, average delivery minutes, favourite product category, city, age, gender and tenure days. Behaviour uses only pre-cutoff orders; cancellation rate is cancelled orders divided by all completed/cancelled orders before cutoff. Favourite category is the mode of completed-order categories, resolving ties alphabetically. Demographics and signup dates come from the required clean customer table. The latest order entering feature aggregation is `2025-08-31 22:35:45`.

**No order from 2025-09-01 onwards enters the model feature aggregations, and whole-window segment labels and Premium flags are not model features.** September completed orders are used solely to define the target. Validation removes all September orders and asserts that every feature remains identical. Part A's prescribed age/delivery imputations are already present in the input tables; their scope is disclosed under Limitations.

The split is `train_test_split(test_size=0.2,stratify=y,random_state=42)`. Each tuned model has a ColumnTransformer/Pipeline with StandardScaler for numeric inputs and `OneHotEncoder(drop="first",handle_unknown="ignore")` for city, gender and favourite category. **No encoder, scaler or resampler saw test rows:** preprocessing is refitted inside each training CV fold and the selected pipeline is refitted on training rows only; oversampling is confined to training rows. There are no missing feature values.

The selected scaler's learned frequency mean is **8.114719**, versus **8.028547** for the full cohort. They differ because the scaler learns from the random training subset rather than all cohort customers. The persisted numeric scaler saw 924 rows, with 19 total transformed features after encoding.

## Baseline, tuning and held-out evaluation

Always predicting not lapsed gives test accuracy **0.6164**, positive-class precision **0.0000**, recall **0.0000** and F1 **0.0000**. Metrics use `zero_division=0`.

Both GridSearchCV runs use five-fold CV and `scoring="f1"` on training data. LogisticRegression uses `max_iter=1000` and C in `0.01,0.1,1,10,100` (25 CV fits plus refit). RandomForestClassifier uses 100 trees, seed 42, max_depth in `3,5,8` and min_samples_leaf in `5,10,20` (45 CV fits plus refit). Complete candidates, fold scores and ranks are in `.reports/logistic_regression_cv.csv` and `.reports/random_forest_cv.csv`.

Precision, recall and F1 below refer exclusively to the lapsed class. Test accuracy is included as context.

| model | best_parameters | cv_f1 | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | {'model__C': 100} | 0.6506 | 0.7284 | 0.6444 | 0.6517 | 0.648 |
| random_forest | {'model__max_depth': 8, 'model__min_samples_leaf': 10} | 0.6811 | 0.7371 | 0.6591 | 0.6517 | 0.6554 |

Confusion matrices use actual labels as rows and predictions as columns, label order not lapsed (0), lapsed (1):

**always_not_lapsed:**

| actual | Predicted not lapsed | Predicted lapsed |
| --- | --- | --- |
| Not lapsed | 143 | 0 |
| Lapsed | 89 | 0 |
**logistic_regression:**

| actual | Predicted not lapsed | Predicted lapsed |
| --- | --- | --- |
| Not lapsed | 111 | 32 |
| Lapsed | 31 | 58 |
**random_forest:**

| actual | Predicted not lapsed | Predicted lapsed |
| --- | --- | --- |
| Not lapsed | 113 | 30 |
| Lapsed | 31 | 58 |

We select **random forest** because it has the higher training cross-validated F1, fixing the selection before inspecting held-out scores. Its held-out lapsed F1 is **0.6554**, exceeding the required 0.55. Missing a truly lapsing customer can lose future revenue, while wrongly targeting a loyal customer spends campaign budget unnecessarily. F1 balances precision and recall for this demonstration; a production campaign should set its probability threshold using measured retention value and contact/incentive cost, rather than maximising test performance. Both tuned models are evaluated once after tuning, with results saved in `.reports/lapse_test_predictions.csv`.

## Training-only class balancing

The original training counts are 570 not lapsed and 354 lapsed. All original training rows are retained, and 216 lapsed training rows are sampled with replacement using seed 42 to reach 570 in each class. A fresh pipeline refits the best logistic C on that balanced training data and is scored on the original unchanged test set.

| training | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| Original logistic | 0.7284 | 0.6444 | 0.6517 | 0.648 |
| Oversampled logistic | 0.7155 | 0.5891 | 0.8539 | 0.6972 |

Changes (balanced minus original): precision **-0.0553**, recall **+0.2022**, F1 **+0.0492**. **Recall** moves most in absolute terms. Recall increases, finding more lapsing customers, while precision does not increase; the retention team must accommodate additional contacts and false positives. This experiment is reported separately and does not use held-out results to retune or replace the selected model.

## Interpretation, persistence and retention list

Top five random-forest importances:

| feature | importance | business_reading |
| --- | --- | --- |
| numeric__frequency | 0.2796 | Past order count captures the strength of the repeat-purchase habit. |
| numeric__monetary | 0.2712 | Past completed spend helps identify differing levels of customer value. |
| numeric__recency | 0.1151 | Time since the last completed purchase reflects how recently the customer engaged. |
| numeric__average_delivery_minutes | 0.0613 | Average delivery duration summarises the historical service experience. |
| numeric__cancellation_rate | 0.0523 | The share of cancelled orders captures disrupted purchasing. |

Importance measures relative predictive use, not an effect direction or causation. Logistic coefficients for three numeric features (scaled, so coefficients correspond to a one-standard-deviation change):

| feature | coefficient | sign | business_reading |
| --- | --- | --- | --- |
| numeric__recency | 0.0396 | positive | Higher recency is associated with higher lapse odds, holding other inputs fixed. |
| numeric__frequency | -1.6982 | negative | Higher frequency is associated with lower lapse odds, holding other inputs fixed. |
| numeric__monetary | 0.2846 | positive | Higher monetary is associated with higher lapse odds, holding other inputs fixed. |

The selected fitted preprocessing/model Pipeline is saved with `joblib.dump` to `outputs/lapse_model.joblib` and reloaded with `joblib.load`. Both its test predictions and predicted probabilities reproduce the originals exactly, asserted in `persist_model()` and checked against saved outputs by `validate_part_b.py`.

There are 40 Premium customers in the held-out set. `outputs/retention_list.csv` ranks the top 10 by descending selected-model lapse probability (customer ID breaks ties), with required columns `customer_id,lapse_probability,actually_lapsed`; **1** of them actually lapsed.

| customer_id | lapse_probability | actually_lapsed |
| --- | --- | --- |
| C0862 | 0.491 | 0 |
| C1112 | 0.4841 | 0 |
| C0060 | 0.4798 | 0 |
| C0802 | 0.4796 | 0 |
| C1008 | 0.4778 | 0 |
| C1000 | 0.4624 | 0 |
| C0525 | 0.4597 | 1 |
| C0133 | 0.4562 | 0 |
| C0267 | 0.4525 | 0 |
| C0167 | 0.4522 | 0 |

**The Premium flag is a post-hoc whole-window label that includes September, so the retention list is illustrative and is not a leakage-free evaluation.** It does not change the model's pre-cutoff feature restriction or its independent held-out metrics.

Part B acceptance checks cover the reference RFM, cluster and PCA tolerances, cohort/split counts, persisted schemas/profiles, train-only preprocessing, September feature isolation, full tuning grids, model F1, reloaded predictions and retention ordering. Successful validation prints `ALL PART B CHECKPOINTS PASSED`.


# Part C

Implemented by [part_c.py](<part_c.py>) and checked by [validate_part_c.py](<validate_part_c.py>). All calls use the unmodified offline `mock_llm.py`; no account, network call or API key is used.

## Input guards and privacy

Tickets are read with `keep_default_na=False` and processed in their original CSV order. Text that is empty or whitespace-only is marked `skipped_empty`; raw text above 400 `est_tokens` is `flagged_long`. Both guards precede model calls and budget checks: the four empty and four long tickets make no calls, including after a budget is exhausted. All four long tickets estimate to 734 tokens.

`redact_ticket()` applies the required patterns in this exact order:

| Type | Pattern | Replacement |
| --- | --- | --- |
| Email | `[\w.+-]+@[\w-]+\.[\w.]+` | `[EMAIL]` |
| Phone | `\b[6-9]\d{9}\b` | `[PHONE]` |
| Order ID | `ORD-\d{6}` | `[ORDER_ID]` |

Measured substitutions across all 164 tickets (counted once per run):

| Type | Substitutions |
| --- | --- |
| email | 55 |
| phone | 91 |
| order_id | 132 |
| total | 278 |

Reapplying all three patterns to the redacted texts finds **0 matches**. `redact_ticket()` and `call_with_retry()` assert privacy before any attempted call, including retries. Each request has exactly one system message and one user message; the user content is exactly the redacted ticket text, without ticket/customer metadata or labels. Supplementary `.reports/triage_redaction_audit.csv` records redacted text and substitution counts; the per-run `.reports/triage_<run>_audit.jsonl` records every attempt's redacted messages, preceding cumulative usage, reply and usage. Raw ticket text is never copied into these audits.

## Zero-shot and few-shot prompts

Both system prompts contain CLASSIFY, list all four categories and request JSON-only category/reason replies. The zero-shot prompt has no EXAMPLES marker; the few-shot prompt includes four labelled examples, including fraud and technical cases.

Zero-shot system prompt:

```text
CLASSIFY the ticket as billing, delivery, technical, or fraud.
Return JSON only with this schema: {"category": "...", "reason": "..."}.
Choose one allowed category and give a short reason.
```

Few-shot system prompt:

```text
CLASSIFY the ticket as billing, delivery, technical, or fraud.
Return JSON only with this schema: {"category": "...", "reason": "..."}.
Choose one allowed category and give a short reason.
EXAMPLES:
Ticket: Someone hacked my account and made an unauthorized payment.
Label: {"category": "fraud", "reason": "unauthorized account activity"}
Ticket: The app freezes with an error on the payment page.
Label: {"category": "technical", "reason": "application failure"}
Ticket: My invoice shows an extra charge.
Label: {"category": "billing", "reason": "incorrect billing"}
Ticket: My delivery is late and the rider is unreachable.
Label: {"category": "delivery", "reason": "delivery delay"}
```

## Retries and response validation

`mock_llm.reset()` runs immediately before each full batch. The retry wrapper allows one initial attempt plus at most three retries, with wait `0.05 * 2**n * random.uniform(0.9,1.15)` for retry numbers n=0,1,2. Jitter uses an isolated seed-42 generator, and the actual pipeline sleeps for the logged wait. Each retry produces one line with ticket ID, retry number and wait; log files are `.reports/triage_zero_shot_retry.log`, `.reports/triage_few_shot_retry.log`, `.reports/triage_budget_demo_retry.log` and `.reports/triage_final_retry.log`.

| run | attempts | successful_calls | transient_errors | retries | permanent_errors |
| --- | --- | --- | --- | --- | --- |
| zero_shot | 175 | 156 | 19 | 19 | 0 |
| few_shot | 175 | 156 | 19 | 19 | 0 |
| budget_demo | 103 | 92 | 11 | 11 | 0 |
| final | 175 | 156 | 19 | 19 | 0 |

All 19 transient failures in each full run recovered. Every actual retry had number 0 and a wait between 0.045 and 0.0575 seconds. Failed attempts have no response usage in this mock, so they add no tokens. If all three retries were ever exhausted, `flagged_model_error` would send the ticket to human review; a boundary test verifies the four-attempt stop, but no supplied-data ticket reaches it.

`validate_response()` accepts valid JSON only when its category is an allowed string. Only non-JSON replies are eligible for fallback extraction with `re.findall(r"[a-z]+", text.lower())`: exactly one distinct allowed category word yields `classified_keyword`, correctly recognising hyphenated wording such as billing-related. Valid JSON with an unknown category, missing/ill-typed category, multiple category words or no allowed word is `flagged_invalid_output`; it is never guessed. The 12 keyword recoveries are syntactic validation, not a guarantee of semantic correctness.

Status counts:

| status | zero_shot | few_shot | budget_demo | final |
| --- | --- | --- | --- | --- |
| classified_json | 135 | 135 | 80 | 135 |
| classified_keyword | 12 | 12 | 7 | 12 |
| flagged_invalid_output | 9 | 9 | 5 | 9 |
| skipped_empty | 4 | 4 | 4 | 4 |
| flagged_long | 4 | 4 | 4 | 4 |
| budget_exhausted | 0 | 0 | 64 | 0 |
| flagged_model_error | 0 | 0 | 0 | 0 |

## Prompt comparison and token cost

The first two runs have no token budget. Accuracy is measured only across classified tickets against `true_category`, which is never included in a prompt. Total usage sums every successful response's `usage.total_tokens`, including invalid outputs, and tokens per classified ticket divides that entire total by the number classified.

| prompt | classified | correct | accuracy | total_tokens | tokens_per_classified_ticket | system_prompt_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| zero_shot | 147 | 126 | 0.8571 | 10180 | 69.2517 | 37 |
| few_shot | 147 | 147 | 1 | 23128 | 157.3333 | 120 |

Zero-shot confusion table (actual rows, predicted columns; excluded/unclassified tickets omitted):

| actual | billing | delivery | technical | fraud |
| --- | --- | --- | --- | --- |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

Zero-shot correctly classifies 126/147 tickets (85.7%), consuming 10,180 tokens or 69.25 per classified ticket. Few-shot reaches 147/147 (100.0%), using 23,128 tokens or 157.33 per classified ticket. The extra 12,948 tokens eliminate 21 classification errors: 11 fraud cases and 10 technical cases that zero-shot sends to billing. That extra prompt cost is justified when missed fraud and incorrect support routing cost more than the additional 616.57 estimated tokens per corrected classification; a low-risk queue with a strict budget may prefer the cheaper prompt plus human checks.

These are the supplied mock's estimated tokens, not paid usage or evidence of real-model accuracy.

## Budget demonstration and final routing

The zero-shot budget demo uses **6,000 tokens**. Before every attempt, including retries, the wrapper checks whether cumulative usage is already at least the budget; after that point it makes no call and assigns `budget_exhausted`. The demo successfully calls 92 tickets, classifies 87 and leaves **64 budget-exhausted**. Its recorded total is **6,010**: the last call is allowed to cross the threshold because the brief specifies a pre-call check of actual usage, rather than a reservation for estimated future cost. Empty and long tickets still retain their own guardrail statuses. Results are in the required `outputs/triage_budget_demo.csv`.

The final run uses the few-shot prompt with budget **100,000** and consumes **23,128** tokens, with no exhausted tickets. `route_ticket()` reads the Part B `outputs/customer_segments.csv` and applies this precedence:

| Condition | Route | Priority |
| --- | --- | --- |
| Empty, long, invalid output, budget exhausted or unrecovered call failure | human_review | NORMAL |
| Otherwise category fraud | security_team | HIGH |
| Otherwise customer is Premium | premium_desk | HIGH |
| Otherwise (including customer absent from segment file) | support_queue | NORMAL |

Final route counts:

| Route | Tickets |
| --- | --- |
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

The required `outputs/triage_results.csv` contains all **164 rows**, in input order, with `ticket_id,customer_id,status,category,segment_name,priority,route,tokens_used`. Category is blank for unclassified cases; segment name is blank when no matching segment exists. A known segment is retained as metadata even for a guarded ticket; routing still sends that ticket to human review at NORMAL priority. A fraud classification overrides Premium routing.

**135 JSON replies and 12 keyword replies are rule-validated; the other 17 tickets go to humans (9 invalid outputs, 4 empty, 4 long).** The final classified subset has 100% accuracy on the synthetic mock, while the excluded tickets remain explicitly unclassified.

## Verification and supporting evidence

`validate_part_c.py` replays every run against the real offline mock with sleeping disabled and checks its outputs, exact prompts, privacy assertions, per-attempt budgets, usage sums, retry logs, reference counts, accuracies, confusion table and segment-based routes. Boundary checks cover the inclusive 400-token input limit, whitespace-only input, zero/exact budget, unsafe prompt rejection, valid JSON with unknown/ill-typed categories, ambiguous keyword output, retry exhaustion, unknown customers and fraud/Premium precedence. Successful validation prints `ALL PART C CHECKPOINTS PASSED`.

Required outputs stay in `outputs/`. Supplementary `.reports/part_c_report.json`, prompt-comparison/confusion CSVs, redaction audit and per-run audit/retry files are regenerated in the ignored `.reports/` folder; no charts or media are produced.


# Part D

Implemented by [policy_rag.py](<policy_rag.py>), [refund_agent.py](<refund_agent.py>) and [part_d.py](<part_d.py>); verified by [validate_part_d.py](<validate_part_d.py>). All model calls use the unmodified offline mock and all refunds are simulated local ledger entries.

## Policy chunks, retrieval and grounded answers

`chunk_documents()` reads the five policy text files in sorted filename order. Each chunk contains at most 40 whitespace-separated words, starts 30 words after the preceding one (10-word overlap), and retains its source filename and start/end word offsets. It stops after the first chunk reaching the end, avoiding a redundant trailing overlap-only chunk.

| Source | Chunks |
| --- | --- |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| refund_policy.txt | 4 |

Total: **15 chunks**. `PolicyIndex` fits the default `TfidfVectorizer()` on chunk texts, transforms each query using that same vectorizer and ranks with `cosine_similarity`; ties keep source/chunk order. The public `retrieve(query,k=3)` returns dictionaries with `source,score,text`. For `refund policy late delivery`, all three top chunks come from **refund_policy.txt**. The refund agent's `search_policy()` calls this exact retriever and reuses its cached default index.

TF measures how often a term occurs within a chunk, subject to the vectorizer's normalisation. IDF gives greater weight to terms that occur in fewer chunks across the policy collection. A limitation is lexical matching: a paraphrase using different words can miss a relevant passage even when its meaning matches.

`answer(question)` retrieves three chunks, uses a system message containing POLICY ASSISTANT and an instruction to answer only from context or state that information was not found. The user message has exactly this structure, with one source-prefixed chunk on each line:

```text
CONTEXT:
[source filename] first retrieved chunk
[source filename] second retrieved chunk
[source filename] third retrieved chunk
QUESTION:
the question
```

The returned dictionary contains the answer text and sources, plus retrieved chunks, messages and usage for inspection. No expected document or fact from the evaluation dataset is included in model inputs.

## Retrieval and grounding evaluation

All 14 supplied questions are evaluated: expected source in the top three for 12 answerable questions, case-sensitive `expected_fact` substring in the answer for those same 12, and an answer starting `I could not find` for two unanswerable questions.

| overlap_words | chunks | expected_source_in_top_three | answer_contains_expected_fact | unanswerable_fallback | tokens |
| --- | --- | --- | --- | --- | --- |
| 10 | 15 | 12/12 | 12/12 | 2/2 | 3094 |
| 0 | 13 | 12/12 | 10/12 | 2/2 | 2968 |

The zero-overlap comparison uses the same 40-word size, vectorizer, prompt and questions; only overlap changes to zero, producing 13 chunks. It retrieves the expected sources for all 12 answerable questions but fails these two fact checks:

- What delivery time makes a completed order eligible for a full refund?
- How soon is money returned when a payment fails but is deducted?

The first refund chunk ends with `more than 45 minutes, the customer`, while the next begins `receives a full refund of the order amount.` The mock selects that latter fragment without the eligibility threshold, so the expected phrase is absent. The payment-policy boundary similarly splits `the amount is returned automatically` from `within 48 hours.` Ten-word overlap keeps the full relevant sentences in a retrievable chunk, restoring both answers.

Detailed evidence is in supplementary `.reports/policy_eval_overlap_10.csv`, `.reports/policy_eval_overlap_0.csv` and `.reports/policy_chunks_overlap_10.json`/`.reports/policy_chunks_overlap_0.json`. These are benchmark substring/source checks, not a general proof that every possible answer is grounded.

## Tool definitions and refund rules

`TOOLS_SPEC` stores each tool's name, one-sentence description and JSON-style object parameter schema with required fields and no extra properties. `TOOL_MAP` dispatches model-selected names to Python functions; `_validate_call()` rejects unknown tools, missing/extra arguments and incorrect types.

| Tool | Required parameters | Implementation and result |
| --- | --- | --- |
| get_order | order_id: string | Read-only, parameterised SQLite query against `data/instamart.db`; returns order_id/status/order_amount/delivery_minutes, or order-not-found error. |
| search_policy | query: string | Calls the shared retriever; returns passages under results. |
| calculate_refund | order_amount: number, delivery_minutes: number or null | More than 45 minutes gives the full rounded amount; 45 or less, or unknown duration, gives zero. |
| issue_refund | order_id: string, refund_inr: number | Rechecks database eligibility and records automatic approval only for a positive eligible refund up to INR 1000. |
| escalate_to_human | order_id: string, refund_inr: number | Rechecks database eligibility and records a human-approval request only above INR 1000; no automatic payment is issued. |

Only completed orders are eligible; cancelled orders were never charged. The model sees the order status, and the action tools independently enforce that status, the delivery threshold, the full-order amount and approval threshold. Finite numeric values are required. All simulated requests are assumed raised within the policy's seven-day window; real request dates are not being checked.

The two action tools append a JSON line to required `outputs/refund_ledger.jsonl` with an aware UTC timestamp, tool name, order ID, refund amount, run ID and tool-call ID, returning `refund_inr`. A durable run/tool-call action key prevents a repeated ledger action when resuming across an action/checkpoint interruption. This is local, single-process simulation; concurrent workers would need transactional storage and locking.

## ReAct loop and seven scenarios

`run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10)` starts with an AGENT system message and the user request. Each response either ends the run with the prefix before the colon, or appends an assistant tool-call record, validates and executes the tool through TOOL_MAP, appends a JSON tool observation with its call ID, saves a checkpoint and continues. The result includes status, final answer, steps, model-call count and total tokens, plus run ID, refund amount and call-level audits.

Measured scenario results (refund amounts in INR):

| scenario_id | status | refund_inr | llm_calls | tokens_used |
| --- | --- | --- | --- | --- |
| S1 | REFUND | 128.25 | 5 | 914 |
| S2 | REFUND | 135 | 5 | 914 |
| S3 | ESCALATE | 1,999 | 5 | 901 |
| S4 | DENIED | 0 | 4 | 631 |
| S5 | DENIED | 0 | 3 | 371 |
| S6 | NOT_FOUND | 0 | 2 | 120 |
| S7 | FALLBACK_MAX_ITERATIONS | 0 | 6 | 2805 |

S1/S2 record refunds, S3 records escalation, S4 denies an on-time delivery, S5 denies a cancelled order and S6 returns NOT_FOUND. S7 intentionally keeps searching and reaches its iteration limit after six calls, consuming **2805**, below the default 6,000-token limit.

Required `outputs/agent_runs.jsonl` appends one line per completed logical run, containing scenario ID, status, final answer, calls, usage and every step's tool name, arguments and observation. The initial crash and resume are one logical run, with one final run record. Call-level audits also retain the exact request messages and replies for deterministic mock replay. Re-running `part_d.py` appends a fresh batch with unique run IDs; the supplementary scenario CSV and report describe the latest batch, while ledger/history remain append-only.

## Stop safely

- **Iteration:** at most max_iterations model calls across the whole logical run; no final reply after six calls produces FALLBACK_MAX_ITERATIONS.
- **Tokens:** usage is added after every model response. If the total is strictly greater than token_limit, stop immediately before executing the requested tool or accepting a final reply. An exactly-equal total is permitted.
- **Time:** at the top of each iteration, stop when `time.monotonic() - started >= time_limit_seconds`. Zero seconds stops before the first model call. Across a resume, accumulated active execution is retained and time spent stopped between processes is excluded.

Measured demonstrations:

| demo | status | llm_calls | tokens_used | completed_tools |
| --- | --- | --- | --- | --- |
| token_limit | FALLBACK_TOKEN_LIMIT | 3 | 388 | get_order, search_policy |
| timeout | FALLBACK_TIMEOUT | 0 | 0 | none |

The S1 200-token demonstration stops before the third response's requested `calculate_refund` tool; its completed lookup and policy search are retained. Every fallback returns its completed steps and a short human-handoff summary. Invalid/unsafe tool requests additionally produce FALLBACK_TOOL_ERROR and human handoff without executing the rejected action.

## Checkpoint, crash and resume

After each tool observation, `save_checkpoint()` writes `state/<run_id>.json` via a temporary file and atomic replacement. A short bounded retry handles temporary Windows file locks. State includes run/scenario IDs, the full message history, usage, call/step counts, executed tools, completed steps, pending tool calls, configured limits and active elapsed time. `resume_agent(run_id)` restores that state; it processes pending work before asking for another model response. Completed runs can be resumed again without duplicating their ledger or final run record. Run IDs are restricted to safe filename characters.

S1 was interrupted immediately after the second tool execution and checkpoint, then resumed from JSON:

| Stage | Tools executed |
| --- | --- |
| Before simulated crash | get_order → search_policy |
| After resume | calculate_refund → issue_refund |
| Across the logical run | get_order → search_policy → calculate_refund → issue_refund |

Exactly **four tool executions**, none repeated; final status **REFUND**, refund **INR 128.25**, total **5 model calls** and **914 tokens**. The first checkpoint held two calls; the resumed process performed the remaining three. This batch has four ledger actions: S1, S2, the S3 escalation and the resumed S1 demonstration. Limit and denial cases issue no refunds.

## ReAct versus plan-and-execute (at most 250 words)

The measured ReAct agent uses 5 calls/914 tokens for S1, 3 calls/371 tokens for cancelled S5, and 2 calls/120 tokens for missing-order S6. It asks the model after each observation, so later requests resend a growing history. This makes an unexpected NOT_FOUND easy to handle immediately, and a cancelled order ends after lookup and policy retrieval rather than progressing to payment.

A plan-and-execute design would request a step list once, execute validated tools, and replan only on an unexpected observation; an optional final summary could add a second model call. It should reduce routine model calls and repeated-context tokens, but its exact token savings are unmeasured here. The executor must stop on a missing or cancelled order, and reject an invalid planned action even if the initial plan requested it.

For this fixed refund workflow, I would choose plan-and-execute with deterministic eligibility checks, approval gates and the same durable checkpoints. The normal sequence is predictable and needs little model reasoning between tools. ReAct remains useful for less predictable support investigations where each observation changes the next step.

## Verification

`validate_part_d.py` checks exact chunk boundaries/counts, both policy evaluations, prompt structure, retrieval sources, every scenario outcome/amount, persisted message/step alignment and usage. It replays every recorded model request against the unmodified mock without repeating actions. Isolated temporary-file tests cover iteration/token/time boundaries, unknown tools, invalid values, ineligible actions, the INR 1000 threshold, path safety, ledger idempotency and a crash/resume with an independent execution spy proving no repeated tools. Successful validation prints `ALL PART D CHECKPOINTS PASSED`.

Required traces, ledger and checkpoints stay in `outputs/` and `state/`. Supplementary `.reports/agent_scenario_results.csv` and `.reports/part_d_report.json` and the policy-evaluation evidence are regenerated in the ignored `.reports/` folder. No real network call, API key or payment is used.


# Limitations and Next Steps

All data, policies and pricing are fictional. Category-median delivery imputation is required by the brief but is not evidence of an actual delivery duration; downstream refund demonstrations will use this cleaned database. Part A computes its required age/category-delivery medians on the whole cleaning dataset, so the strict pre-cutoff guarantee applies to Part B feature aggregation and fitted model preprocessing; a production temporal evaluation should fit missing-value statistics on historical training data only. IQR outliers are descriptive flags, not grounds for deletion. No real refund is issued.

The Premium flag is a post-hoc whole-window label that includes September, so the retention list is illustrative and is not a leakage-free evaluation. The synthetic holdout does not establish future campaign performance, causal effects or calibrated probabilities; campaign thresholds need real cost/benefit validation. Part C's exact regex guards cover the three specified identifier patterns, not every possible form of personal information, and its mock accuracy does not establish real-model performance. Part D uses lexical retrieval, synthetic policy checks and local single-process action logs; real deployment would need transactional actions, concurrent-worker coordination and real approval/payment integration. All simulated refund requests are assumed within the seven-day policy window. Public GitHub publication on `main` remains pending.
