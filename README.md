# Fraud-Shield-ML
**End-to-end credit card fraud detection system** — from raw transaction data to a deployed interactive dashboard. Built on the IEEE-CIS Fraud Detection dataset (590,540 transactions, Vesta Corporation).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-shield-ml.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-PR--AUC%200.824-brightgreen)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Live Demo

**→ [fraud-shield-ml.streamlit.app](https://fraud-shield-ml.streamlit.app)**

Four interactive tabs: model performance overview · EDA explorer · live transaction scoring · SHAP + LIME explainability deep dive.

---

## Results

| Model | PR-AUC | ROC-AUC | F2 Score | Recall | Precision |
|-------|--------|---------|----------|--------|-----------|
| Logistic Regression (baseline) | 0.3062 | 0.8372 | 0.3958 | 0.5437 | 0.1896 |
| LightGBM + Optuna | 0.7861 | 0.9622 | 0.7364 | 0.7687 | 0.6305 |
| **XGBoost + Optuna** ✅ | **0.8242** | **0.9670** | **0.7669** | **0.7866** | **0.6970** |

XGBoost catches **78.7% of all fraud** at **69.7% precision** — a **169% PR-AUC improvement** over the logistic regression baseline.

Validated on 118,108 held-out transactions (4,133 fraud).

---

## What This Project Covers

```
Phase 1  →  Project scaffold, modular structure, centralised config + logging
Phase 2  →  Business framing (F2 vs accuracy, cost asymmetry), data ingestion
Phase 3  →  EDA: 8 publication-quality figures, temporal patterns, missing audit
Phase 4  →  Imbalance: SMOTE vs ADASYN vs class_weight vs threshold_tune on PR-AUC
Phase 5  →  Model training: LR → XGBoost → LightGBM, Optuna HPO, threshold analysis
Phase 6  →  XAI: SHAP global beeswarm, SHAP waterfall, LIME false positive
Phase 7  →  Internal DS report: 6-section PDF memo to Head of Risk Analytics
Phase 8  →  Streamlit dashboard, surrogate model, Streamlit Cloud deployment
Phase 9  →  This README
```

---

## Business Framing

This is an **adversarial, rare-event classification problem** with severe class imbalance (3.50% fraud rate, 27.6:1 ratio).

**Why F2, not accuracy:**
A model predicting every transaction as legitimate scores **96.5% accuracy** while catching zero fraud. Accuracy is useless here. F2 score (β=2) weights recall twice over precision, reflecting the real cost asymmetry: a missed fraud costs the full transaction value plus chargeback fees; a false alarm costs recoverable customer friction.

**Two operating thresholds:**
- **0.446**: F2-optimal, routes to analyst review queue
- **0.990**: high-precision hard-block, routes to automatic decline

---

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle, 2019 — Vesta Corporation)

| Property | Value |
|----------|-------|
| Transactions | 590,540 |
| Fraud rate | 3.50% (20,663 fraud) |
| Raw features | 394 (transaction + identity tables) |
| Features after preprocessing | 221 |
| Date range | December 2017 – May 2018 |

Download: [kaggle.com/competitions/ieee-fraud-detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

---

## Key Technical Decisions

**Left join, not inner join** — only 24.4% of transactions have identity data. Dropping the other 75.6% would discard the majority of the dataset. Missingness in identity columns is itself a behavioural signal the model learns.

**No SMOTE on full dataset** — SMOTE on 590K rows with 221 features requires ~12GB RAM and 45+ minutes. Strategy comparison was done on a 50K stratified subsample. XGBoost uses `scale_pos_weight=27.6` instead.

**PR-AUC over ROC-AUC for model selection** — ROC-AUC is optimistic under class imbalance because it accounts for true negatives (trivially easy when 96.5% of transactions are legitimate). PR-AUC focuses entirely on the minority class.

**`select_dtypes(exclude='number')` for dtype detection** — pandas 2.3 introduced `future.infer_string=True` which silently converts object columns to `StringDtype` on copy/slice operations. Hard-coded column lists break. Data-driven detection using pandas' own API is immune to version changes.

**Module-level wrapper classes for joblib** — Python's pickle cannot serialise classes defined inside functions. `XGBWrapper` and `LGBMWrapper` must be at module scope for `joblib.dump/load` to work correctly across different execution contexts.

---

## Project Structure

```
Fraud-Shield-ML/
├── config/
│   └── config.yaml              # all params, paths, thresholds — single source of truth
├── data/
│   ├── raw/                     # CSVs from Kaggle (git-ignored, ~680MB)
│   └── processed/               # merged.parquet (git-ignored, ~55MB)
├── docs/
│   └── business_framing.md      # problem definition, cost matrix, metric rationale
├── src/
│   ├── ingestion/
│   │   └── load_data.py         # merge, timestamp engineering, missing audit → parquet
│   ├── eda/
│   │   ├── plot_style.py        # shared matplotlib theme, colour palette
│   │   └── run_eda.py           # 8-figure EDA pipeline
│   ├── imbalance/
│   │   ├── preprocessor.py      # ColumnTransformer: impute + encode (pandas 2.3-safe)
│   │   └── compare_strategies.py # SMOTE vs ADASYN vs class_weight vs threshold_tune
│   ├── models/
│   │   ├── evaluate.py          # metric suite, F2 threshold search, PR/ROC curves
│   │   └── train_models.py      # LR → XGBoost (Optuna) → LightGBM (Optuna)
│   ├── explainability/
│   │   ├── shap_analysis.py     # SHAP TreeExplainer, beeswarm, waterfall
│   │   ├── lime_analysis.py     # LIME false positive explanation
│   │   └── run_xai.py           # SHAP + LIME orchestrator
│   ├── reporting/
│   │   ├── generate_report.py   # 6-section ReportLab PDF memo
│   │   ├── generate_feature_stats.py  # precompute medians/modes for Streamlit
│   │   └── train_surrogate_model.py   # lightweight model for Streamlit Cloud
│   └── utils/
│       ├── logger.py            # loguru rotating logger, coloured stdout
│       └── config_loader.py     # dot-accessible YAML config (SimpleNamespace)
├── outputs/
│   ├── figures/                 # 19 PNG figures (EDA, evaluation, XAI)
│   ├── models/                  # JSON configs, feature stats, surrogate model
│   └── reports/                 # PDF memo (git-ignored, regenerated from source)
├── streamlit_app/
│   └── app.py                   # 4-tab interactive dashboard
├── tests/                       # 215-test pytest suite across all 7 phases
├── .streamlit/
│   └── config.toml              # navy/red theme, server settings
├── requirements.txt             # full local development dependencies
└── requirements_streamlit.txt   # minimal Streamlit Cloud dependencies
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/maniharshith68/Fraud-Shield-ML.git
cd Fraud-Shield-ML

# 2. Install
pip3 install -r requirements.txt
pip3 install -e .

# 3. Download data (requires Kaggle account + competition rules acceptance)
# Place train_transaction.csv and train_identity.csv in data/raw/
# https://www.kaggle.com/competitions/ieee-fraud-detection/data

# 4. Run the full pipeline
python3 src/ingestion/load_data.py
python3 src/eda/run_eda.py
python3 src/imbalance/compare_strategies.py
python3 src/models/train_models.py
python3 src/explainability/run_xai.py
python3 src/reporting/generate_report.py
python3 src/reporting/generate_feature_stats.py

# 5. Launch dashboard
streamlit run streamlit_app/app.py

# 6. Run test suite
pytest tests/ -v   # 215 tests across all phases
```

---

## EDA Highlights

**Class imbalance:** 27.6:1 ratio — justifies every metric choice in this project.

**Peak fraud hour: 07:00** — early morning spike consistent with overnight batch fraud exploiting card-holder sleep windows.

**Avg fraud amount $149 vs legit $134** — amount alone is a weak signal. The model's primary drivers are Vesta's V-features (behavioural velocity signals), not the transaction amount.

**200 of 221 features have missing values** — structured missingness in V-columns correlates with product type. Missingness is informative, not noise.

---

## Explainability

**SHAP (global):** Top features driving fraud predictions are Vesta V-features capturing transaction velocity and behavioural fingerprints. TransactionAmt ranks lower than V12, V87, V258 — confirming the model learned behavioural patterns, not just amount thresholds.

**SHAP (per-transaction):** The waterfall for the highest-confidence fraud (probability=1.000) shows multiple V-features compounding in the same direction — a convergence of behavioural anomalies, not a single trigger.

**LIME (false positive):** The selected false positive sits at fraud probability 0.595 against a threshold of 0.446 — close enough to the boundary to have a genuinely instructive local explanation. Red features reveal which transaction characteristics triggered the flag on a legitimate transaction.

---

## Limitations

- **Temporal validity** — trained on 2017–2018 data. Fraud tactics evolve; retraining should be triggered when live PR-AUC drops >3pp below this baseline.
- **V-feature opacity** — Vesta's V-columns are proprietary. Their exact definitions are not publicly documented, limiting feature engineering without Vesta's documentation.
- **No real-time velocity features** — features like "transactions on this card in the last 5 minutes" are among the strongest production fraud signals but are not in this static dataset.
- **Subgroup fairness not evaluated** — performance was not disaggregated by geography, card issuer, or demographic proxies.
- **Surrogate on Streamlit Cloud** — the live prediction tab uses a lightweight surrogate model (PR-AUC ≈ 0.76) due to GitHub's file size constraints. The full model runs locally.

---

## Tech Stack

| Area | Tools |
|------|-------|
| Data | pandas, numpy, pyarrow |
| ML | scikit-learn, XGBoost, LightGBM |
| Imbalance | imbalanced-learn (SMOTE, ADASYN) |
| Tuning | Optuna (TPE sampler) |
| Explainability | SHAP, LIME |
| Visualisation | matplotlib, seaborn |
| Reporting | ReportLab (Platypus) |
| Dashboard | Streamlit |
| Logging | loguru |
| Testing | pytest (215 tests) |
| Deployment | Streamlit Community Cloud |

---

## Test Suite

```
pytest tests/ -v
```

215 tests across 7 phases covering project structure, config loading, data ingestion logic, EDA plot functions, imbalance strategy metrics, model evaluation utilities, XAI computation, and PDF report generation. Unit tests use synthetic data and run without the raw dataset. Integration tests activate automatically when pipeline outputs exist.

---

## Internal DS Report

A 6-section PDF memo addressed to the Head of Risk Analytics is generated by `src/reporting/generate_report.py`. Sections: Executive Summary · Methodology · Results · Model Card · Explainability · Limitations & Recommendations. Available for download directly from the Streamlit dashboard.

---

