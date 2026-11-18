# Vehicle Fuel Efficiency Prediction

Machine Learning Algorithms, final course project.

Predicting the combined fuel consumption (L/100 km) of new cars from their specifications, using the official Natural Resources Canada fuel consumption ratings for model years 2014-2025.

## Team and roles

| Member | Role | Notebook sections |
|---|---|---|
| Bigali | Problem statement, EDA, PCA, error analysis and conclusions; written report | 1 problem statement, 4 EDA, 5.1 and 5.4 features, 6.1-6.3 split, 10 PCA, 13 error analysis, 15 conclusions |
| Iskander | Data, leakage protection, first models, clustering, reproducibility | 2 data loading, 3 cleaning, 5.2 `is_hybrid`, 5.3 preprocessing, 6.4-6.6 split design and leakage audit, 7 baseline and first models, 9 K-Means, 14 reproducibility check |
| Asset | Model tuning and neural network | 8 tuning, test evaluation and time-based check, 11 neural network, 12 final comparison |

## Project question

Can the combined fuel consumption of a new passenger car be predicted from characteristics known before it is measured: model year, engine size, cylinders, vehicle class, transmission, fuel type and a hybrid flag derived from the model name?

- **Task type:** regression
- **Target:** `Combined (L/100 km)`
- **Metrics:** MAE (main), RMSE, R²
- **Success criterion:** the test MAE of the final model is at most half the MAE of the mean baseline, and R² ≥ 0.8. The criterion was fixed before any model was trained.

## Dataset

- **Source:** Natural Resources Canada, Fuel Consumption Ratings (Open Canada): https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64
- **Licence:** Open Government Licence - Canada
- **Loaded from:** a copy of the official data in `bgl96395/fuel-efficiency-dataset`, file `fuel_consumption_2014_2025.csv`:
  https://raw.githubusercontent.com/bgl96395/fuel-efficiency-dataset/refs/heads/main/fuel_consumption_2014_2025.csv
  The years 2015-2024 were checked against the official file (rows per year, fuel types, target statistics, missing values); 2014 and 2025 were not checked separately.
- **Size:** 11829 rows, 15 columns, model years 2014-2025; 11828 rows after cleaning (one row with fuel type `N` removed).
- **Known issues:** no vehicle weight or power; `CO2 rating` and `Smog rating` are missing for 2014-2016; six columns (`City`, `Highway`, `Combined (mpg)`, `CO2 emissions`, `CO2 rating`, `Smog rating`) are the target in another form and are excluded as leakage.

## Setup

Python 3.13 (the package versions below need Python 3.11 or newer; tested with 3.13.1) and an internet connection: the notebook reads the CSV from the URL above.

**Package versions:**

| Package | Version |
|---|---|
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| torch | 2.15.0 (CPU) |
| jupyter | any recent version |

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS / Linux: source .venv/bin/activate
pip install numpy==2.5.3 pandas==3.0.5 scikit-learn==1.9.1 scipy==1.18.1 matplotlib==3.11.2 seaborn==0.13.2 torch==2.15.0 jupyter
jupyter notebook notebooks/Vehicle_Fuel_Efficiency_prediction.ipynb
```

In VS Code, open the notebook and choose the `.venv` kernel (Select Kernel → Python Environments → `.venv`). The first code cell prints the Python and package versions, so you can check that they match the table above.

## Execution order

Open `notebooks/Vehicle_Fuel_Efficiency_prediction.ipynb` and run **Kernel → Restart & Run All**. One run takes about 5 minutes on a laptop CPU (longer on slower machines); most of it is the grid searches and the neural network.

The notebook is one study and runs top to bottom:

1. Problem statement
2. Data: source, loading and first look
3. Data cleaning
4. Exploratory data analysis
5. Feature engineering (`trans_type`, `gears`, `disp_per_cyl`, `is_hybrid`, preprocessing)
6. Train / validation / test split and leakage protection
7. Baseline and first supervised models (Linear Regression, Decision Tree, KNN), cross-validation
8. Hyperparameter tuning, evaluation on the test set, time-based check
9. K-Means clustering
10. PCA
11. Feedforward neural network (MLP)
12. Final comparison
13. Error analysis and discussion
14. Reproducibility check
15. Conclusions

All imports are in the first code cell. Every random step uses one `SEED = 42` (split, CV, models, K-Means, PCA, `torch.manual_seed`), and the network is trained on CPU with deterministic algorithms. With the versions above every table is reproduced; PyTorch results can differ slightly on other machines.

## Method

- **Leakage protection:** the six target-derived columns are excluded (6.5 shows the formulas behind them); all preprocessing (median imputation, scaling, one-hot encoding) is inside a `Pipeline` and is fitted on train only (6.6).
- **Split:** group split by `Make + Model`, train 70% / validation 15% / test 15%, so the same car model never appears in two parts. A random row split would make the error look about 15% lower (6.4). No stratification: the parts already have a similar target distribution, and stratified folds give the same result (6.4).
- **Models and tuning:** mean baseline, Linear Regression, Decision Tree and KNN; `GridSearchCV` with `GroupKFold(5)` on train, validation as an independent check, test used once after refitting on train + validation (8).
- **Model selection:** the final model is chosen by CV MAE on train, not by test; test only reports the final estimate (12).
- **Clustering:** K-Means on engine size, cylinders, gears and `is_hybrid` (scaled on train); k = 4 chosen by the elbow, silhouette and stability across seeds (9).
- **PCA:** on the 31 scaled model features, fitted on train; used for visualization and interpretation (10).
- **Neural network:** PyTorch MLP 31 → 128 → 64 → 32 → 1, ReLU, MSE loss, Adam (lr 0.003), early stopping on validation; evaluated with the same CV and test protocol as the other models (11).

## Final results (test, 1751 cars)

| Model | Settings | CV MAE (train) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| Mean baseline | mean of train | 2.284 ± 0.070 | 2.211 | 2.835 | 0.000 |
| Linear Regression | OLS | 0.815 ± 0.022 | 0.724 | 0.948 | 0.888 |
| **Decision Tree** | max_depth=20, min_samples_leaf=1 | **0.653 ± 0.031** | **0.598** | **0.842** | **0.912** |
| KNN | scaled, no `Model year`, k=50, distance | 0.667 ± 0.034 | 0.612 | 0.859 | 0.908 |
| Neural network | MLP (128, 64, 32), 93 epochs | 0.666 ± 0.018 | 0.604 | 0.844 | 0.911 |

- **Final model: Decision Tree,** chosen by the lowest CV MAE on train (0.653, vs 0.666 for the network and 0.667 for KNN). The differences are within one CV standard deviation, so among practically tied models the simpler, interpretable tree is preferred; the single test run confirms the choice.
- **Success criterion met:** test MAE 0.598 is 3.70 times lower than the baseline (2.211), R² = 0.912.
- **Effect of the hybrid flag and tuning:** the first models (basic features, one parameter chosen on validation) reach test MAE 0.780 (Linear Regression), 0.615 (tree) and 0.715 (KNN). On the 48 test hybrids the tree's error fell from MAE 1.07 (bias +1.00) to 0.52 (bias +0.31).
- **Clusters:** hybrids (6.8 L/100 km), small engines with older gearboxes (9.7), mid engines with 8+ speed automatics (10.7), V8 and larger (14.7); the cluster label alone explains 51% of the consumption variance.
- **PCA:** PC1 = engine size (31% of the variance), PC2 = generation of the car (year and gears, 17%), PC3 = hybrid flag (12.5%); 2 components (48.8%) are kept for the plots, 10 components keep 90%.
- **Time-based check:** trained on 2014-2023 and tested on 2024-2025, the tree has MAE 0.665; for brand-new car models in future years the error is about 1.2 L/100 km.

**Error analysis and limitations (13):**
- The largest errors are high-performance versions (Chevrolet Camaro ZL1, Mitsubishi Lancer Evolution), underestimated by 3-4 L/100 km, and 8-cylinder cars (bias −0.27); the data has no power or weight, so no model can see the difference.
- The test set is one group split with only 48 hybrids and very few vans; the target is the laboratory rating, not real-world consumption.
- The model is an estimate for buyers, dealers and analysts and must not replace the official NRCan label in purchase, tax or regulatory decisions.

## Contribution record

| Work | Author | Git user |
|---|---|---|
| Notebook sections 1 problem statement, 4 EDA, 5.1 and 5.4 features, 6.1-6.3 split | Bigali | `Bigali` |
| Notebook sections 10 PCA, 13 error analysis, 15 conclusions; written report | Bigali | `Bigali` |
| Notebook sections 2 data loading, 3 cleaning, 7 baseline and first models | Iskander | `isk0nka-dot` |
| Notebook sections 5.2 `is_hybrid`, 5.3 preprocessing, 6.4-6.6 split design and leakage audit, 9 K-Means, 14 reproducibility check | Iskander | `isk0nka-dot` |
| Notebook sections 8 tuning, test evaluation and time-based check, 11 neural network, 12 final comparison | Asset | `aaset` |

The full history is in the git log of the repository.

## Repository

https://github.com/isk0nka-dot/Vehicle-Fuel-Efficiency
