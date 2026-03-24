# AI & Data Science Job Market Analysis

A machine learning solution for analysing the AI & Data Science job market.
Two Gradient Boosting models are trained end-to-end:

| Task | Model | Target |
|---|---|---|
| Salary prediction | `GradientBoostingRegressor` | `salary` |
| Hiring urgency | `GradientBoostingClassifier` | `hiring_urgency` |

---

## Project Structure

```
.
├── data_preprocessing.py   # Data loading, cleaning, encoding, scaling & splitting
├── models.py               # SalaryPredictor and HiringUrgencyPredictor classes
├── model_evaluation.py     # Metrics, cross-validation and visualisation helpers
├── exploratory_analysis.py # EDA functions and plotting utilities
├── train.py                # Main orchestration script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Dataset

The expected CSV file should contain the following columns:

### Features

| Column | Type | Description |
|---|---|---|
| `experience_level` | categorical | Entry / Mid / Senior / Executive |
| `years_experience` | numeric | Years of professional experience |
| `education_level` | categorical | Bachelor's / Master's / PhD / etc. |
| `hiring_urgency` | categorical/numeric | Target for classification |
| `num_openings` | numeric | Number of open positions |
| `job_posting_year` | numeric | Year the job was posted |
| `job_title` | categorical | Job title string |
| `company_industry` | categorical | Industry sector |
| `country` | categorical | Country of the role |
| `company_size` | categorical | Small / Medium / Large |
| `requires_python` | binary (0/1) | Python skill required |
| `requires_sql` | binary (0/1) | SQL skill required |
| `requires_ml` | binary (0/1) | Machine Learning skill required |
| `requires_deep_learning` | binary (0/1) | Deep Learning skill required |
| `requires_cloud` | binary (0/1) | Cloud skill required |

### Targets

| Column | Type | Description |
|---|---|---|
| `salary` | numeric | Annual salary (regression target) |
| `hiring_urgency` | categorical/numeric | Urgency level (classification target) |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/thintz123/AI-Data-Science-Job-Market-Analysis.git
cd AI-Data-Science-Job-Market-Analysis

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Quick start

```bash
python train.py --data path/to/job_market_data.csv
```

### All options

```
usage: train.py [-h] [--data DATA] [--output-dir OUTPUT_DIR]
                [--test-size TEST_SIZE] [--cv-folds CV_FOLDS] [--skip-eda]

optional arguments:
  --data          Path to the CSV dataset  (default: data/job_market_data.csv)
  --output-dir    Output directory for models and plots  (default: output)
  --test-size     Test set fraction  (default: 0.2)
  --cv-folds      Cross-validation folds  (default: 5)
  --skip-eda      Skip exploratory data analysis plots
```

### Output

After training, the `output/` directory will contain:

```
output/
├── models/
│   ├── salary_model.joblib
│   ├── hiring_urgency_model.joblib
│   ├── encoders.joblib
│   └── scaler.joblib
└── plots/
    ├── eda/
    │   ├── numeric_distributions.png
    │   ├── categorical_distributions.png
    │   ├── correlation_heatmap.png
    │   ├── salary_by_experience_level.png
    │   └── hiring_urgency_distribution.png
    ├── salary/
    │   ├── salary_regression.png
    │   └── salary_feature_importance.png
    └── hiring_urgency/
        ├── hiring_urgency_confusion.png
        └── hiring_urgency_feature_importance.png
```

### Using individual modules

```python
from data_preprocessing import preprocess
from models import SalaryPredictor, HiringUrgencyPredictor
from model_evaluation import evaluate_regression, evaluate_classification
from exploratory_analysis import run_eda
import pandas as pd

# EDA
df = pd.read_csv("data/job_market_data.csv")
run_eda(df, output_dir="output/plots/eda")

# Preprocessing
data = preprocess("data/job_market_data.csv")

# Train salary model
salary_model = SalaryPredictor()
salary_model.train(data["X_train"], data["y_salary_train"])
preds = salary_model.predict(data["X_test"])
metrics = evaluate_regression(data["y_salary_test"], preds)

# Train hiring urgency model
urgency_model = HiringUrgencyPredictor()
urgency_model.train(data["X_train"], data["y_urgency_train"])
preds_u = urgency_model.predict(data["X_test"])
metrics_u = evaluate_classification(data["y_urgency_test"], preds_u)
```

---

## Model Details

Both models use scikit-learn's `GradientBoosting` implementations with the following
default hyperparameters (configurable via the `params` argument):

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 |
| `learning_rate` | 0.05 |
| `max_depth` | 5 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 5 |
| `subsample` | 0.8 |
| `max_features` | `"sqrt"` |

---

## Expected Model Performance

Performance will vary depending on the specific dataset, but typical ranges are:

| Metric | Salary Regression | Hiring Urgency Classification |
|---|---|---|
| R² / Accuracy | 0.70 – 0.90 | 0.75 – 0.92 |
| RMSE / F1 | dataset-dependent | 0.75 – 0.92 |

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list.

- Python ≥ 3.9
- scikit-learn ≥ 1.3
- pandas ≥ 2.0
- numpy ≥ 1.24
- matplotlib ≥ 3.7
- seaborn ≥ 0.12
- joblib ≥ 1.3