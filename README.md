# Housing Price Regression: Data Exploration, Preprocessing & Regularized Linear Models

An end-to-end, leak-free machine learning investigation analyzing property characteristics associated with housing prices and comparing **Linear Regression**, **Ridge**, **Lasso**, and **Elastic Net** models on `Housing.xlsx` / `Housing.csv`.

## Repository Structure
```text
├── housing_price_regression.ipynb            # Fully executed Jupyter Notebook (EDA, pipeline, tuning, evaluation, experiments)
├── Housing.xlsx                              # Original Excel dataset (545 rows x 13 columns)
├── Housing.csv                               # CSV export of the dataset for cross-platform compatibility
├── Housing_Price_Regression_Report.pdf       # Concise analytical report (PDF)
├── report.md                                 # Markdown version of the short analytical report
├── Housing_Price_Regression_Presentation.pptx# 8-slide summary presentation deck
├── figures/                                  # High-resolution EDA, residual, and coefficient plots
├── requirements.txt                          # Pinned Python dependencies
└── README.md                                 # Project overview and reproduction instructions
```

## Key Findings Summary
- **Data Audit:** 545 properties across 13 variables with zero missing values, zero exact duplicates, and positive skewness in `price` (`1.21`) and `area` (`1.32`). Dataset units for `price`, `area`, and `prefarea` are not specified in the source file and are preserved without external assumptions.
- **Primary Price Associations:** Across linear models, `airconditioning` (`+791,427` OLS weight), `hotwaterheating` (`+684,650`), `prefarea` (`+629,891`), `bathrooms` (`+521,879` per 1 SD), and `area` (`+519,552` per 1 SD) show the strongest positive associations with `price`, while `unfurnished` status (`-413,645` vs. `furnished`) shows the largest negative shift.
- **Model Comparison (5-Fold CV & 20% Held-Out Test Set):**
  - All four models outperform a constant-mean `DummyRegressor` (`Test R² = -0.0179`) by a wide margin (`Test R² ≈ 0.647–0.653`).
  - **Ridge (`alpha = 6.2605`)** achieves the lowest 5-fold CV RMSE (`1,021,858`), while unregularized **Linear Regression** slightly leads on the held-out test set (`MAE = 970,043`, `RMSE = 1,324,507`, `R² = 0.6529`) because all 13 encoded predictors carry genuine signal with low multicollinearity (`max r = 0.41`).
- **Best Experiment (`log1p(price)` Target Transform):** Modeling multiplicative price scaling via `TransformedTargetRegressor(log1p / expm1)` improves every metric simultaneously (`CV RMSE = 1,005,443`, `Test MAE = 960,136`, `Test RMSE = 1,314,663`, `Test R² = 0.6581`).

## How to Reproduce
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd housing-price-regression
   ```
2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook end-to-end:**
   ```bash
   jupyter notebook housing_price_regression.ipynb
   ```
   Select **Kernel -> Restart & Run All** to reproduce every audit table, visualization, cross-validation search, and evaluation metric deterministically (`RANDOM_STATE = 42`).
