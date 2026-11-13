# Vehicle Fuel Efficiency Prediction

Machine Learning Algorithms, endterm course project.

## Team and roles

| Member | Role in the endterm | Notebook stages |
|---|---|---|
| Iskander | Features, leakage protection, clustering, reproducibility | 2a (leakage audit, `is_hybrid`, `Pipeline`, split design), 3 (K-Means), 7 (reproducibility) |
| Asset | Supervised models and neural network | 2b (tuning), 2c (time-based check), 5 (ANN), 6a (final comparison table) |
| Bigali | Problem statement, PCA, conclusions, presentation | 1 (final problem statement), 4 (PCA), 6b (error analysis, limitations, ethics, conclusions), presentation |

## Project question

Can the combined fuel consumption of a new passenger car be predicted from characteristics known before it is measured: model year, engine size, cylinders, vehicle class, transmission, fuel type and a hybrid flag derived from the model name?

- **Task type:** regression
- **Target:** `Combined (L/100 km)`
- **Metrics:** MAE (main), RMSE, R²
- **Success criterion:** the MAE of the best model on test is at most half the MAE of the mean baseline, and R² ≥ 0.8 (fixed in the midterm, not changed after seeing the test results).

## Dataset

- **Source:** Natural Resources Canada, Fuel Consumption Ratings (Open Canada): https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64
- **Licence:** Open Government Licence - Canada
- **Loaded from:** a copy of the official data in `bgl96395/fuel-efficiency-dataset`, file `fuel_consumption_2014_2025.csv`:
  https://raw.githubusercontent.com/bgl96395/fuel-efficiency-dataset/refs/heads/main/fuel_consumption_2014_2025.csv
  The years 2015-2024 were checked against the official file (rows per year, fuel types, target statistics, missing values); 2014 and 2025 were not checked separately.
- **Size:** 11829 rows, 15 columns, model years 2014-2025; 11828 rows after cleaning (one row with fuel type `N` removed).
- **Known issues:** no vehicle weight or power; `CO2 rating` and `Smog rating` are missing for 2014-2016; six columns (`City`, `Highway`, `Combined (mpg)`, `CO2 emissions`, `CO2 rating`, `Smog rating`) are the target in another form and are excluded as leakage.

## Setup

Python 3.13 and an internet connection (the notebook reads the CSV from the URL above).

```bash
pip install -r requirements.txt
jupyter notebook notebooks/endterm.ipynb
```

Package versions (`requirements.txt`): numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, seaborn 0.13.2, torch 2.15.0 (CPU), jupyter.

## Execution order

Open `notebooks/endterm.ipynb` and run **Kernel → Restart & Run All**. One run takes about 5-15 minutes on a CPU, depending on the machine; most of it is the ANN. PyTorch results can differ slightly between machines even with the same seed and versions; all other numbers are identical.

The notebook runs top to bottom:

1. Midterm part: problem, data loading, cleaning, EDA, feature engineering, group split, baseline and models (Linear Regression, Decision Tree, KNN), cross-validation, test, error analysis.
2. Endterm part: 1 final problem statement → 2a leakage audit, `is_hybrid`, preprocessing `Pipeline` → 2b tuning → 2c time-based check → 3 K-Means → 4 PCA → 5 ANN → 6a final comparison → 6b error analysis and conclusions → 7 reproducibility check.

`notebooks/midterm.ipynb` is the midterm submission, kept for reference; it is not needed to run the endterm notebook.

All random steps use one `SEED = 42` (split, CV, models, K-Means, PCA, `torch.manual_seed`). The ANN is trained on CPU with deterministic algorithms.

## Method

- **Leakage protection:** the six target-derived columns are excluded (2a.1 shows the formulas behind them); all preprocessing (median imputation, scaling, one-hot) is inside a `Pipeline` and is fitted on train only.
- **Split:** group split by `Make + Model`, train 70% / validation 15% / test 15%, so the same car model never appears in two parts. A random row split would make the error look about 15% lower (2a.4). No stratification: the parts already have a similar target distribution (checked in 2a.4).
- **Tuning:** `GridSearchCV` with `GroupKFold(5)` on train; validation as an independent check; test used once after refitting on train + validation.
- **Clustering:** K-Means on engine size, cylinders, gears and `is_hybrid` (scaled on train), k = 4 chosen by elbow, silhouette and stability.
- **PCA:** on the 31 scaled model features; used for visualization and interpretation.
- **ANN:** PyTorch MLP 31 → 128 → 64 → 32 → 1, ReLU, MSE loss, Adam (lr 0.003), early stopping on validation; evaluated with the same CV and test protocol as the other models.

## Final results (test, 1751 cars)

| Model | Settings | CV MAE (train) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| Mean baseline | mean of train | 2.284 ± 0.070 | 2.211 | 2.835 | 0.000 |
| Linear Regression | OLS | 0.815 ± 0.022 | 0.724 | 0.948 | 0.888 |
| **Decision Tree** | max_depth=20, min_samples_leaf=1 | **0.653 ± 0.031** | **0.598** | 0.842 | 0.912 |
| KNN | scaled, no `Model year`, k=50, distance | 0.667 ± 0.034 | 0.612 | 0.859 | 0.908 |
| ANN | MLP (128, 64, 32), 93 epochs | 0.666 ± 0.018 | 0.604 | 0.844 | 0.911 |

- **Final model:** Decision Tree. Its test MAE is 3.70 times lower than the baseline, R² = 0.912, so the success criterion is met. Midterm tree on the same test: MAE 0.615.
- The tree, the ANN and KNN are practically tied (difference ≤ 0.014 MAE, less than one CV std); the ANN is second on every test metric, trains about 85 times longer than the tree and is harder to explain.
- `is_hybrid` fixed the main midterm error: the hybrid bias on test fell from +1.00 to +0.31 L/100 km.
- **Clusters:** hybrids (6.8 L/100 km), small engines with older gearboxes (9.7), mid engines with 8+ speed automatics (10.7), V8 and larger (14.7); the cluster label alone explains 51% of the consumption variance.
- **PCA:** PC1 = engine size (31% of variance), PC2 = generation (year and gears, 17%), PC3 = hybrid flag (12.5%); 10 components keep 90% of the variance.
- **Time check:** trained on 2014-2023 and tested on 2024-2025, the tree has MAE 0.665; for brand-new car models in future years the error is about 1.2 L/100 km.

**Error analysis and limitations (6b):**
- The largest errors are high-performance versions (Chevrolet Camaro ZL1, Mitsubishi Lancer Evolution), underestimated by 3-4 L/100 km, and 8-cylinder cars (bias −0.27); the data has no power or weight, so no model can see the difference.
- The test set is one group split with only 48 hybrids and very few vans; the target is the laboratory rating, not real-world consumption.
- The model is an estimate and must not replace the official NRCan label in purchase, tax or regulatory decisions.

## Contribution record

| Stage | Author | Commits |
|---|---|---|
| Midterm: structure, data loading, cleaning, split, models, CV, test (stages 2, 3, 7) | Iskander | `isk0nka-dot` |
| Midterm: dataset, feature engineering, EDA, models, notebook finish, README (stages 4, 5, 6) | Bigali | `Bigali` |
| Endterm 2a, 3, 7 | Iskander | `isk0nka-dot` |
| Endterm 2b, 2c, 5, 6a | Asset | `aaset` |
| Endterm 1, 4, 6b, README, presentation | Bigali | `Bigali` |

The full history is in the git log of the repository. In the notebook the table at the start of the endterm part lists the author of every stage.

## Repository

https://github.com/isk0nka-dot/Vehicle-Fuel-Efficiency
