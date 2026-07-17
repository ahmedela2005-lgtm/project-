# ICU Risk Score Prediction

Predicting ICU patient risk from vitals and lab values — as both a continuous **regression** target (`risk_score`) and a 3-class **classification** target (`risk_level`: Low / Medium / High).

> **Status:** Pipeline reviewed and leakage-fixed (see `fixed_pipeline.py`). Results below are placeholders pending execution against the real dataset — run the pipeline and paste your numbers in before publishing this README.

## Overview

This project benchmarks classical ML algorithms (linear, tree-based, ensemble, kernel, instance-based) on ICU patient data, using a leakage-safe pipeline: preprocessing (imputation, scaling, encoding) is fit *only* on training folds, and classification labels are derived from training-set quantiles, not the full dataset.

## Dataset

- 1,000 patient records
- Features: demographics (`age`, `gender`), vitals (`heart_rate`, `systolic_bp`, `diastolic_bp`, `respiratory_rate`, `temperature_c`, `spo2`), labs (`wbc_count`, `hemoglobin`, `platelets`, `creatinine`, `blood_urea`, `lactate`)
- Targets: `risk_score` (continuous), `risk_level` (derived tercile classes)

## Installation

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place `icu_risk_score.csv` in the project root, then:

```bash
python fixed_pipeline.py
```

## Models Benchmarked

| Task | Models |
|---|---|
| Regression | Dummy, Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, SVR |
| Classification | Dummy, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVC, KNN |

Each model is evaluated with 5-fold cross-validation on the training set and a single held-out test set, reporting an **Overfit Gap** (Train score − CV score) to flag overfitting.

## Results

*(Fill in after running `fixed_pipeline.py` on the real dataset.)*

| Model | CV R² | Test R² | Overfit Gap |
|---|---|---|---|
| — | — | — | — |

| Model | CV Macro F1 | Test Macro F1 | Test Accuracy | Overfit Gap |
|---|---|---|---|---|
| — | — | — | — | — |

## Project Structure

```
├── data/icu_risk_score.csv
├── src/ (or fixed_pipeline.py for the single-file version)
├── notebooks/eda.ipynb
├── requirements.txt
└── README.md
```

## Known Limitations

- Dataset appears synthetic; validate on real ICU data before clinical use.
- No hyperparameter tuning grid has been finalized — current search spaces in `fixed_pipeline.py` are a starting point, not exhaustive.
- No interpretability (SHAP/feature importance) output yet.

## License

Add your license here.
