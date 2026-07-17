# ICU Risk Score Prediction — Project Documentation

> ⚠️ **Status note:** This documentation describes the project as designed in the reviewed notebook. The modeling section (regression + classification) has not yet been executed with output captured, and the raw dataset (`icu_risk_score.csv`) was not available during this review. Sections marked **[PENDING EXECUTION]** need to be filled in with real numbers once the corrected pipeline (`fixed_pipeline.py`) is run against the actual data.

## 1. Project Overview

This project builds a machine learning pipeline to estimate ICU patient risk from a set of vital signs and laboratory values. It approaches the same underlying signal — `risk_score` — as two separate ML problems:

- **Regression**: predict the exact continuous `risk_score`.
- **Classification**: predict a discretized `risk_level` (Low / Medium / High), derived from `risk_score`.

Both tasks are benchmarked across a range of classical ML algorithms with cross-validation, held-out test evaluation, and an explicit "Overfit Gap" diagnostic.

## 2. Problem Statement

Clinicians in an ICU setting need a fast, data-driven way to triage patients by risk level using routinely collected vitals (heart rate, blood pressure, respiratory rate, temperature, SpO2) and labs (WBC count, hemoglobin, platelets, creatinine, blood urea, lactate). The goal is to produce a model that:

1. Predicts a continuous risk score for fine-grained ranking of patients.
2. Predicts a categorical risk tier (Low/Medium/High) for simpler triage workflows.
3. Generalizes to unseen patients rather than memorizing the training sample.

## 3. Dataset

- **Source:** `icu_risk_score.csv` (1,000 patient records, appears to be synthetically generated given the fully continuous, noise-free structure of the vitals).
- **Size:** 1,000 rows × 16 columns before feature exclusion.
- **Target(s):** `risk_score` (continuous), `risk_level` (derived, 3-class categorical).
- **Features:**
  - Demographic: `age`, `gender`
  - Vitals: `heart_rate`, `systolic_bp`, `diastolic_bp`, `respiratory_rate`, `temperature_c`, `spo2`
  - Labs: `wbc_count`, `hemoglobin`, `platelets`, `creatinine`, `blood_urea`, `lactate`
  - Identifier (excluded from modeling): `patient_id`

**Caveat:** Because the data has the statistical signature of a synthetic dataset (smooth continuous distributions, no missingness patterns typical of real EHR data), performance numbers from this project should not be assumed to transfer to real clinical deployments without validation on real patient data.

## 4. Workflow

1. **Load & deduplicate** — read CSV, drop duplicate `patient_id` rows (and full-row duplicates).
2. **Clean** — clip physiologically implausible values (SpO2 > 100%, temperature outside 25–45°C), convert negative lab values to missing, and (in the corrected pipeline) cap implausible upper-bound lab values too.
3. **Handle missing data** — impute **inside the model pipeline only**, fit separately per cross-validation fold (median for numeric, most-frequent for categorical), to avoid leaking test-set statistics into training.
4. **Feature typing** — cast `age` to integer, `gender` to categorical.
5. **Target engineering** — derive `risk_level` via tercile split of `risk_score`, with bin edges computed on the training set only and applied to test data.
6. **EDA** — missingness heatmap, target distributions, correlation heatmap, per-feature histograms and boxplots.
7. **Modeling** — build a `ColumnTransformer` + `Pipeline` (median imputation + scaling for numeric, most-frequent imputation + one-hot encoding for categorical gender), then benchmark multiple algorithms with 5-fold cross-validation on the training split and a final held-out test evaluation.
8. **Model selection** — rank models by cross-validation score first, confirm with test score, and check the Overfit Gap (Train score − CV score) to flag overfitting.
9. **Reporting** — comparison tables/plots, confusion matrix and classification report for the best classifier.

## 5. Models Used

**Regression:** Dummy (mean baseline), Linear Regression, Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Regression.

**Classification:** Dummy (most-frequent baseline), Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Classifier, K-Nearest Neighbors.

All models are wrapped in the same preprocessing pipeline and evaluated with identical CV splitting strategy (`KFold` for regression, `StratifiedKFold` for classification) for a fair comparison.

## 6. Results & Metrics **[PENDING EXECUTION]**

*Populate this table by running `fixed_pipeline.py` against the actual `icu_risk_score.csv`.*

| Model | CV Score (mean ± std) | Test Score | Overfit Gap |
|---|---|---|---|
| _(regression rows)_ | | | |

| Model | CV Macro F1 (mean ± std) | Test Macro F1 | Test Accuracy | Overfit Gap |
|---|---|---|---|---|
| _(classification rows)_ | | | | |

Include: winning model per task, confusion matrix, per-class precision/recall/F1, and a short written interpretation (which features drove predictions, whether the model clears the dummy baseline by a meaningful margin, whether the Overfit Gap is acceptable).

## 7. Installation & Requirements

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
```

Place `icu_risk_score.csv` in the project root (or update `DATA_PATH` in `fixed_pipeline.py`), then run:

```bash
python fixed_pipeline.py
```

## 8. Project Structure

**Current (as reviewed):**
```
├── Nti_project_اخر_تعديل.ipynb   # single monolithic notebook (EDA + modeling)
```

**Recommended structure:**
```
├── data/
│   └── icu_risk_score.csv
├── notebooks/
│   └── eda.ipynb                 # exploration only, no modeling logic
├── src/
│   ├── data.py                   # load_and_clean(), make_risk_level_labels()
│   ├── pipeline.py                # make_preprocessor(), model registries
│   ├── train.py                   # run_regression(), run_classification()
│   └── evaluate.py                # metrics, plots, reports
├── models/
│   └── best_model.joblib          # serialized final model
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── README.md
└── PROJECT_DOCUMENTATION.md
```

## 9. Strengths & Weaknesses

**Strengths**
- Correct high-level ML hygiene: separate train/test split, 5-fold CV, dummy baselines, an explicit Overfit Gap metric, and model selection ordered by CV score rather than test score.
- Sensible use of `StratifiedKFold` for the imbalanced-ish 3-class problem and `Macro F1` as the primary classification metric.
- Preprocessing correctly scoped inside a `ColumnTransformer`/`Pipeline` (in principle — see leakage issue below).
- Good breadth of models compared (linear, tree, ensemble, kernel, instance-based) for both tasks.

**Weaknesses**
- Two data-leakage bugs found: global median imputation and global quantile-based label creation, both computed before the train/test split (see `ISSUES_AND_FIXES.md`).
- No hyperparameter tuning — all model configs are hand-picked constants.
- No interpretability (feature importance/SHAP) despite a clinical use case where "why" matters.
- Entirely procedural notebook code with no functions, tests, or packaging — hard to reuse or productionize.
- No results have actually been generated/verified yet.

## 10. Future Improvements

- Run nested cross-validation with `GridSearchCV`/`RandomizedSearchCV` for honest hyperparameter tuning.
- Add feature importance / SHAP analysis for the winning model in each task.
- Add probability calibration (`CalibratedClassifierCV`) and a calibration curve, since a "risk score" implies calibrated probabilities matter clinically.
- Expand data cleaning to include upper-bound clipping for all lab values, not just SpO2/temperature.
- Refactor into a proper `src/` package with unit tests.
- Serialize the final model with `joblib` and add a minimal inference script/API.
- Validate results on a real (non-synthetic) ICU dataset before drawing clinical conclusions.

## 11. Conclusion

The project demonstrates a structurally sound comparative ML benchmarking approach — good instincts around cross-validation, baselines, and overfitting checks. However, two leakage bugs undermine the validity of any results produced by the original notebook, and the modeling section has not actually been run to completion. Once the leakage is fixed (patch provided in `fixed_pipeline.py`) and the pipeline is executed against real data, the same evaluation framework should produce trustworthy, publishable results.
