# Vehicle Fuel Efficiency Prediction

## Project question

Can the combined fuel consumption of a new passenger car be predicted from its basic characteristics (model year, engine size, number of cylinders, vehicle class, transmission type, fuel type)?

- **Task type:** regression
- **Target:** `Combined (L/100 km)`
- **Metrics:** MAE (main), RMSE, R²
- **Success criterion:** the MAE of the best model on test is at least 2 times lower than the MAE of the baseline (predicting the train mean), and R² ≥ 0.8.

## Dataset

- **Source:** Natural Resources Canada, Fuel Consumption Ratings (Open Canada): https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64

- **Loaded from:** a copy of the data in the GitHub repository `bgl96395/fuel-efficiency-dataset`, 

    * file `fuel_consumption_2014_2025.csv` 
    * link: https://raw.githubusercontent.com/bgl96395/fuel-efficiency-dataset/refs/heads/main/fuel_consumption_2014_2025.csv

- **License:** Open Government Licence - Canada

- **Size:** 11829 rows, 15 columns, model years 2014-2025 (11828 rows after cleaning)

- **Known issues:**
  - no vehicle weight or power in the data (engine size, cylinders, and vehicle class are used instead);
  - `CO2 rating` and `Smog rating` are missing for the first years (2014-2016);
  - several columns directly express the target (`City`, `Highway`, `Combined (mpg)`, `CO2 emissions`, `CO2 rating`, `Smog rating`) and are excluded as data leakage;
  - one row with `Fuel type = N` was removed.

## Approach

1. **Data cleaning:** duplicates, missing values, categories, ranges, model name normalization, leakage columns.
2. **EDA:** 5 plots with interpretation (target distribution, engine size vs. consumption, correlation matrix, cylinders and fuel type, years and vehicle class).
3. **Feature engineering:** transmission split into `trans_type` and `gears`, new feature `disp_per_cyl`, median imputation, one-hot encoding.
4. **Split:** group split by `Make` + `Model` (train 70% / validation 15% / test 15%), so that the same car model never appears in both train and test.
5. **Models:** mean baseline, Linear Regression, Decision Tree Regressor, KNN Regressor.
6. **Evaluation:** validation for model selection, `GroupKFold` (5 folds) cross-validation on train, test used once at the end, error analysis.

## Current results (test set)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Mean baseline | 2.211 | 2.835 | -0.000 |
| Linear Regression | 0.780 | 1.028 | 0.868 |
| Decision Tree | 0.628 | 0.920 | 0.895 |
| KNN | 0.715 | 0.973 | 0.882 |

Cross-validation on train (`GroupKFold`, 5 folds), MAE mean ± std: Decision Tree 0.736 ± 0.040, KNN 0.816 ± 0.019, Linear Regression 0.880 ± 0.049.

The best model is **Decision Tree**: its MAE is about 3.5 times lower than the baseline, so the success criterion is met.

Main error analysis findings: hybrids are overestimated (no hybrid feature in the data), and powerful "supercharged" versions (e.g. Camaro ZL1, Lancer Evolution) are underestimated.

## Setup

Python 3.10+ and Jupyter are required.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook midterm.ipynb
```

Run all cells in order. The data is loaded directly from the `CSV_URL` link.

## Repository

https://github.com/isk0nka-dot/Vehicle-Fuel-Efficiency

## Next steps

**Open problems:**
- Decision Tree with `max_depth=None` is overfitted, and a depth restriction has not been tested yet.
- There is no "hybrid" feature, so consumption for hybrids is overestimated.
- "Supercharged" versions are underestimated because the data has no power.
- E85 and the rare groups (5 and 16 cylinders, Van: Cargo) are evaluated on too few examples.
- KNN without scaling is hurt by `Model year`.
- Generalization over time has not been checked.

**Plan for the final:**
- Tune the tree hyperparameters via GridSearchCV with GroupKFold (`max_depth`, `min_samples_leaf`).
- Add an `is_hybrid` feature based on the model name and re-evaluate hybrids.
- Check over time: train on 2014-2023, test on 2024-2025.
- Discuss E85 and "supercharged" versions separately; test KNN without `Model year`.
