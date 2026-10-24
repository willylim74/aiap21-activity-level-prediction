# Activity Level Prediction (AIAP Batch 21 Technical Assessment)

**Full name (as in NRIC):** `<FILL IN>`  
**Email (as in application form):** `<FILL IN>`

## Problem statement

ElderGuard Analytics wants a **non-invasive early-warning system** that can predict potential health emergencies or concerning situations for elderly residents living independently, using environmental sensor data and indoor air quality monitoring (CO2, CO and metal-oxide gas sensors, temperature, humidity, HVAC mode, lighting). The system should help identify when a resident may be experiencing distress, a medical episode or unsafe living conditions, supporting care coordination for family members and healthcare providers.

**The task as given (kept unchanged):** build and evaluate at least three models that classify a resident's `Activity Level` (Low / Moderate / High) from the sensor data, and identify which features contribute most. Activity level is the proxy signal for the resident's state; the pipeline is a scripted, configurable end-to-end pipeline (`main.py` + `src/`) with one executed notebook (`eda.ipynb`) covering the EDA, cleaning, baselines, tuning and the final test.

### Why activity level matters: illustrative health situations

Activity level is the only label in the data. The table lists situations that *could* be associated with each level; it is domain context to motivate the task, **not something derived from or validated by this dataset** (the data contain no emergency labels).

| Activity level | Normal reading | Concerning situations it might signal |
|---|---|---|
| **Low** | Resting, sleeping, quiet periods | Prolonged inactivity during the day: a fall with inability to get up, fainting or loss of consciousness, stroke, acute illness (infection, low blood pressure), severe fatigue or depression and social withdrawal; unusually low activity at night-time can also mean breathing problems |
| **Moderate** | Typical daily routine (cooking, moving between rooms) | Usually the expected baseline; the concern is a *departure from the resident's own routine* (for example a sustained shift from Moderate to Low) rather than the level itself |
| **High** | Active periods, housework, visitors | Agitation or restlessness, confusion or delirium, night-time wandering (e.g. dementia), pain or breathing distress causing pacing, or unusual activity at atypical hours |

Air-quality and climate readings also signal **unsafe living conditions** independently of activity: elevated carbon monoxide (poisoning risk), high CO2 from poor ventilation (drowsiness, headaches), and temperature extremes (heat stress, hypothermia). This project models activity level only; it does not separately detect these hazards.

### Assumptions and limitations of the problem framing

- **Recognition, not forecasting.** The model classifies the activity level from sensor readings recorded at the same time; it does not predict an emergency in advance.
- **Proxy label.** No health-event labels exist. The link between activity level and risk is an assumption supplied by domain reasoning (table above) and is untested here. A prediction is not a diagnosis.
- **Which level counts as an alert is not specified.** Low activity (inactivity) and High activity (agitation) can each be a concern at certain times; the baseline for one resident is not the baseline for another. The models treat the three classes equally (macro metrics); an operational alert rule would need an explicit choice of class, threshold and acceptable false-alarm rate.
- **Sessions, not residents.** `Session ID` identifies a data-collection session; the data do not say whether sessions map to residents or homes. Evaluation therefore splits by session, which approximates generalisation to unseen settings but not to unseen residents.
- **Modest signal.** The EDA shows small to moderate effects and a ceiling of about macro-F1 0.45-0.50; most of the association is between sessions rather than within them. The predictions are suitable for triage with human review, not for automated decisions.
- **Possibly synthetic or contaminated data.** Repairs (Kelvin temperatures, impossible humidity, label spellings) are assumptions documented in the EDA.
- **Privacy and consent.** Care monitoring involves personal data and needs consent, minimisation and access control in deployment.

> Note on format: built under a 2-hour limit and, by explicit instruction, the EDA, baselines, tuning and final evaluation were first run in a notebook (`eda.ipynb`). The scripts in `src/` + `main.py` are the same pipeline and **assert that they reproduce the notebook's numbers**.

## 1. Folder structure

```
.
├── eda.ipynb              # executed notebook: in-depth EDA (sections 1-12), cleaning/pipeline/baselines (13-23), tuning and final test (24-40)
├── main.py                # entry point (paths resolved relative to this file)
├── run.sh                 # runs main.py (does not install dependencies)
├── requirements.txt
├── src/
│   ├── config.yaml          # ALL tunable choices: cleaning rules, features, split, models, grids, tuning budget
│   ├── data_preparation.py  # loading, DataCleaner, FeatureEngineer, Winsorizer, Pipeline, session split, self-tests
│   └── model_training.py    # model registry, metrics, baselines, tuning + ledger, importance, final test
├── data/gas_monitoring.db # NOT committed (see .gitignore)
└── outputs/
    ├── (notebook artifacts) baseline_results.csv, tuned_results.csv/.json, best_params.json, search_spaces.json,
    │   cv_results/, feature_importance_tuned.csv, final_test_results.csv/.json, final_test_predictions.csv,
    │   split_ids.json, final_model.joblib, run_manifest.json, nb_ledger.json, figures/, README.md
    └── pipeline_run/      # artifacts written by main.py (same file names; checked against the notebook ones)
```

## 2. How to run

```bash
pip install -r requirements.txt
./run.sh                      # full pipeline: baselines, tuning, importance, one-time test (~3.5 minutes on 8 cores)
./run.sh --skip-tuning        # baselines only (~15 s)
./run.sh --self-test          # data_preparation self-tests
./run.sh --no-resume          # ignore checkpoints in outputs/pipeline_run and recompute everything
```

`main.py` compares its results with the notebook artifacts in `outputs/` and fails (assertion) if the baselines, the tuned parameters and validation scores, or the test metrics differ (relative tolerance 1e-6); the checks are skipped if those files do not exist.

**Changing the experiment:** edit `src/config.yaml` (or pass `--config other.yaml`).
- Preprocessing: `cleaning.*` (ranges, flags, ordinal maps), `features.engineered` (`[]` switches engineering off; `mos_pc1` is also available), `features.winsorize_quantiles`.
- Models and parameters: add or edit entries under `models:` (any scikit-learn classifier via `class:`; `params` = library defaults, `tuned_fixed` = settings used only when tuning, `grid` = search space, `search: grid|halving`).
- Split and metric: `split.*`, `metrics.primary`. Tuning budget: `tuning.*` (CV folds, halving rungs 450 -> 1,350 -> 4,050 rows, 27 candidates).

## 3. Pipeline flow

```
SQLite (data/gas_monitoring.db)
  -> normalise target labels, drop exact duplicates          (row-level, before splitting)  -> 9,840 rows
  -> split 70/15/15 BY SESSION (28/5/5 sessions)             (test locked until the end)
  -> Pipeline [clean -> engineer -> winsorise -> median impute -> scale | one-hot | missing flags]   (fitted on train only)
  -> baselines: 6 models at library defaults, train fit, validation score
  -> tuning: grouped 3-fold CV by session; grid (logistic) / successive halving on <=4,050-row subsamples (others);
             winner refit on the FULL training split; class weights only on the tuned clones
  -> select by highest validation macro PR AUC (rule fixed in advance) -> permutation importance (validation)
  -> refit on train+validation -> score test ONCE (+ bootstrap CIs, paired gain vs default twin and naive model)
```

## 4. Key EDA findings and resulting choices (details in `eda.ipynb`, sections 1-12)

| Finding | Choice in the pipeline |
|---|---|
| Target has 6 spellings of 3 classes (Low 57.9%, Moderate 31.2%, High 10.9%); 120 exact duplicates (160 after label normalisation) | Collapse to Low / Moderate / High; drop duplicates before the split; macro metrics, macro PR AUC primary |
| 804 Temperature values in Kelvin; 503 Humidity values outside 0-100; 22 negative CO2 infrared; HVAC has 23 spellings of 6 modes | K -> °C; impossible values -> NaN; lower-case HVAC (assumptions: unit/corruption errors) |
| Missingness (7-26% in 4 columns) is MCAR (unrelated to the target and to other columns) | Median imputation; missing flags kept (harmless, not needed) |
| **Session ID and row order leak under random CV** (Session ID alone: macro-F1 0.39 random vs 0.28 grouped; rows within a session are strongly autocorrelated) | Drop Session ID; split and tune **by whole sessions** |
| Strongest drivers: MOS Unit4, CO2 electrochemical, CO gas level, Temperature; **HVAC mode and Ambient Light have no association**; class order is not monotonic in the sensors (Moderate has the highest sensor values) | Non-linear learners; macro metrics; treat as 3-class, not ordinal |
| Heavy structure between sessions (session-mean Spearman up to 0.90 vs about 0.2 within sessions); two CO2 sensors negatively related (-0.80 within sessions) | Session-grouped evaluation; `co2_z_gap` engineered feature |
| Extreme readings are informative, not errors; skewness within +/-0.45 | Winsorise at 1%/99% (train-fitted); no log transform (the rule is learned in `fit` and selects no column) |
| Engineered features: only `temp_x_humidity` and `co2_z_gap` beat their parents | Used as configured; the sensitivity check (validation) shows they are not needed (dropping them changes PR AUC by +0.002, within noise) |
| Ceiling: grouped-CV macro-F1 about 0.45-0.50 for boosting, flat learning curve | Expect modest scores; report uncertainty |

## 5. Feature processing

| Feature | Processing |
|---|---|
| Temperature | Kelvin (>100) -> °C; winsorise 1/99%; median impute; standard scale |
| Humidity | outside 0-100 -> NaN; missing flag; winsorise; impute; scale |
| CO2_InfraredSensor | negative -> NaN; missing flag; winsorise; impute; scale |
| CO2_ElectroChemicalSensor, MetalOxideSensor_Unit3 | missing flag; winsorise; impute; scale |
| MetalOxideSensor_Unit1, 2, 4 | winsorise; impute; scale |
| CO_GasSensor, Ambient Light Level | ordinal codes (0-4); missing flag; winsorise; impute; scale |
| Time of Day | one-hot (unseen categories ignored) |
| HVAC Operation Mode | lower-case/strip; one-hot (6 modes) |
| `temp_x_humidity` (engineered) | Temperature x Humidity / 100 |
| `co2_z_gap` (engineered) | z(CO2 electrochemical) - z(CO2 infrared); z-score constants learned in `fit` on training rows |
| Session ID | **dropped** (leakage); only the grouping variable for splitting/CV |
| Activity Level (target) | label normalisation to Low / Moderate / High |

28 model inputs in total (10 numeric, 2 engineered, 4 + 6 one-hot, 6 missing flags).

## 6. Choice of models

Five classifiers plus a naive baseline, chosen to span bias/variance for a problem with modest signal, ~7k training rows, correlated sensors and non-monotonic effects:
- **Naive (prior)**: the floor any model must beat (macro PR AUC 0.333).
- **Logistic regression**: transparent linear baseline; very strong L2 regularisation handles the correlated MOS/CO2 sensors.
- **Decision tree**: interpretable non-linear baseline (high variance; overfits at defaults).
- **Random forest**: variance-reduced trees; robust to scale and outliers, captures interactions.
- **Histogram gradient boosting**: usually the strongest tabular learner; regularised through depth, leaves, learning rate and L2.
- **kNN**: local-similarity model; tests whether neighbourhood structure helps.

## 7. Evaluation

**Metrics.** The target is imbalanced (Low 58%), so accuracy misleads (the naive model gets 0.60 on validation, 0.55 on test). **Macro PR AUC** is the primary metric (per-class area under the precision-recall curve, averaged with equal class weight; threshold-free and sensitive to the rare High class; the naive model scores 0.333). **Macro-F1** is the thresholded companion; macro ROC AUC, macro precision/recall and accuracy are reported for reference. Selection and tuning use macro PR AUC; class weighting is used only on the tuned models.

**Validation (5 sessions): defaults vs tuned (macro PR AUC); session-grouped CV of the defaults on the training sessions in brackets**

| Model | Default | Tuned | Tuned parameters | Verdict |
|---|---|---|---|---|
| Histogram gradient boosting | 0.519 (0.505 +/- 0.007) | **0.539** | lr 0.05, max_iter 100, depth 3, 8 leaves, min leaf 20, L2 0 | improved (+0.020) |
| kNN | 0.471 (0.442 +/- 0.008) | 0.528 | k 60, distance weights, p = 1 | improved (+0.058) |
| Logistic regression | 0.523 (0.489 +/- 0.015) | 0.526 | C = 0.0003 | within noise (+0.004) |
| Random forest | 0.529 (0.507 +/- 0.009) | 0.526 | depth 8, min leaf 10, max_features sqrt | within noise (-0.004) |
| Decision tree | 0.372 (0.373 +/- 0.010) | 0.497 | depth 10, min leaf 40, gini | improved (+0.125) |
| Naive | 0.333 | n/a | | |

Default trees/forest/boosting reach about 1.0 train PR AUC vs about 0.52 validation (heavy overfitting). After tuning the top four models are within 0.014, so their order is not statistically supported; the selection (boosting, margin 0.011) follows the pre-declared rule.

**Test (5 sessions, evaluated once; tuned gradient boosting refit on train+validation):**

| Metric | Tuned HGB | Default HGB | Naive |
|---|---|---|---|
| Macro PR AUC | **0.525** (95% CI 0.505-0.549) | 0.504 | 0.333 |
| Macro F1 | **0.480** (0.455-0.504) | 0.430 | 0.236 |
| Macro ROC AUC | 0.742 | 0.731 | 0.500 |
| Accuracy | 0.532 | 0.606 | 0.548 |

Paired bootstrap: tuning beats its default twin by +0.021 PR AUC (CI 0.003-0.038, only just excluding 0) and +0.051 macro-F1 (CI 0.020-0.080); the improvement over the naive model is +0.194 (CI 0.172-0.216). The CIs resample rows and ignore that the test rows come from only 5 sessions (test PR AUC per session ranges from 0.41 to 0.51). **Tuning helped modestly**: it fixed the poorly regularised defaults (tree, kNN) and gave the selected model a small gain, but did not help random forest or logistic regression.

Per class on test: Low precision 0.84 / recall 0.63, Moderate 0.55 / 0.32, **High 0.21 / 0.67**. Class weighting makes the model call High on 39.8% of rows (actual 12.3%), so High alerts are mostly false and accuracy falls below the naive model; probabilities are miscalibrated and should be used for ranking only, or recalibrated.

**Important features** (permutation importance on validation, mean over the five tuned models; `outputs/feature_importance_tuned.csv`): MetalOxideSensor_Unit4 (clearly the most important, 0.016; 0.035 for boosting), then Temperature (0.009), MetalOxideSensor_Unit2 (0.007) and CO2 electrochemical (0.003). HVAC mode, Time of Day, Ambient Light, Humidity and MOS Unit1/Unit3 contribute essentially nothing. Values are small and noisy (5 validation sessions, correlated sensors).

**Sensitivity (validation only, tuned parameters fixed):** no preprocessing variant changes PR AUC by more than 0.005 (no engineered features +0.002, no missing flags +0.003, dropping HVAC/Time/Light -0.004, Session ID added +0.002), so the extra features and flags are not needed; removing class weights lowers macro-F1 from 0.507 to 0.469; early and late rows within sessions score alike (no drift).

## 8. Deployment considerations

- Ship the whole fitted Pipeline (cleaning included, `outputs/final_model.joblib`), not just the classifier, so raw sensor rows can be scored directly; unseen categories, Kelvin temperatures and impossible values are handled.
- **Generalisation to new homes is the real risk:** scores are for unseen *sessions*, but only 38 sessions exist and session-level scores vary widely (0.41-0.51 PR AUC); re-validate per deployment site and monitor input drift (sensor units, humidity range, new HVAC labels).
- The model is modest (macro PR AUC about 0.52, chance 0.33). Use it as a triage signal with human review. Class weights over-call High (precision 0.21); choose the alert class, threshold and acceptable false-alarm rate explicitly, and recalibrate probabilities before showing them to users (the notebook shows how High recall and precision trade off with a class-prior weight).
- A per-resident baseline (departure from the usual routine) is a more natural early-warning rule than an absolute activity class, but needs longitudinal per-resident data that this dataset does not identify.
- Care-related predictions involve personal data: apply consent, data minimisation and access control.

## 9. Deferred work and caveats

| Deferred item | Consequence |
|---|---|
| Repeated grouped CV for model comparison; larger validation/test sets (only 5 sessions each) | The ranking of the top four models rests on one split and is not reliable; CIs are too narrow |
| Larger searches and extended grids (boosting depth/leaves/iterations and kNN k sit at grid edges); subsample-based halving | Tuned parameters may be sub-optimal; differences below about 0.01 PR AUC are noise |
| Calibration, threshold selection, per-class cost analysis | Probabilities are uncalibrated (class weights); High alerts have 0.21 precision |
| Per-resident baselines, forecasting ahead in time, direct air-quality hazard detection | The model recognises current activity only; no emergency labels exist |
| Literature benchmark | Absolute performance level is not compared |

Other caveats: the Kelvin/humidity repairs are assumptions about possibly synthetic or contaminated data; the validation split chose the final model and the validation and test class mixes differ (High 9.8% vs 12.3%).
