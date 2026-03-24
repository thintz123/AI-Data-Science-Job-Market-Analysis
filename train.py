"""
Main training script for the AI & Data Science Job Market Analysis.

Orchestrates the full ML pipeline:
  1. Load & preprocess data
  2. Run exploratory data analysis
  3. Train salary (regression) and hiring urgency (classification) models
  4. Evaluate and visualise performance
  5. Save trained models to disk
"""

import argparse
import logging
import os
import sys

import joblib

from data_preprocessing import preprocess
from exploratory_analysis import run_eda, load_data as eda_load
from model_evaluation import (
    cross_validate_model,
    full_classification_report,
    full_regression_report,
)
from models import HiringUrgencyPredictor, SalaryPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train salary regression and hiring urgency classification models."
    )
    parser.add_argument(
        "--data",
        default="data/job_market_data.csv",
        help="Path to the raw CSV dataset (default: data/job_market_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for saved models and plots (default: output)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing (default: 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip exploratory data analysis plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.data):
        logger.error("Dataset not found at '%s'. Please provide a valid path.", args.data)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Exploratory data analysis (optional)
    # ------------------------------------------------------------------
    if not args.skip_eda:
        logger.info("=== Exploratory Data Analysis ===")
        raw_df = eda_load(args.data)
        run_eda(raw_df, output_dir=os.path.join(plots_dir, "eda"))

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    logger.info("=== Data Preprocessing ===")
    data = preprocess(args.data, test_size=args.test_size)

    X_train = data["X_train"]
    X_test = data["X_test"]
    feature_names = data["feature_names"]
    encoders = data["encoders"]
    scaler = data["scaler"]

    # ------------------------------------------------------------------
    # 3. Salary regression
    # ------------------------------------------------------------------
    if "y_salary_train" in data:
        logger.info("=== Salary Regression ===")
        y_salary_train = data["y_salary_train"]
        y_salary_test = data["y_salary_test"]

        salary_model = SalaryPredictor()
        salary_model.train(X_train, y_salary_train, feature_names=feature_names)

        y_salary_pred = salary_model.predict(X_test)

        salary_metrics = full_regression_report(
            y_salary_test,
            y_salary_pred,
            importances=salary_model.get_feature_importances(),
            feature_names=feature_names,
            output_dir=os.path.join(plots_dir, "salary"),
        )

        # Cross-validation
        cv_salary = cross_validate_model(
            salary_model.model,
            X_train,
            y_salary_train,
            cv=args.cv_folds,
            scoring="r2",
            label="SalaryPredictor",
        )
        logger.info("Salary CV R² – mean: %.4f ± %.4f", cv_salary["mean"], cv_salary["std"])

        # Persist model
        salary_model_path = os.path.join(models_dir, "salary_model.joblib")
        joblib.dump(salary_model, salary_model_path)
        logger.info("Saved SalaryPredictor to %s", salary_model_path)
    else:
        logger.warning("No salary target found – skipping salary model training")
        salary_metrics = {}

    # ------------------------------------------------------------------
    # 4. Hiring urgency classification
    # ------------------------------------------------------------------
    if "y_urgency_train" in data:
        logger.info("=== Hiring Urgency Classification ===")
        y_urgency_train = data["y_urgency_train"]
        y_urgency_test = data["y_urgency_test"]

        urgency_model = HiringUrgencyPredictor()
        urgency_model.train(X_train, y_urgency_train, feature_names=feature_names)

        y_urgency_pred = urgency_model.predict(X_test)

        class_labels = (
            [str(c) for c in urgency_model.classes_]
            if urgency_model.classes_ is not None
            else None
        )

        urgency_metrics = full_classification_report(
            y_urgency_test,
            y_urgency_pred,
            class_labels=class_labels,
            importances=urgency_model.get_feature_importances(),
            feature_names=feature_names,
            output_dir=os.path.join(plots_dir, "hiring_urgency"),
        )

        # Cross-validation
        cv_urgency = cross_validate_model(
            urgency_model.model,
            X_train,
            y_urgency_train,
            cv=args.cv_folds,
            scoring="accuracy",
            label="HiringUrgencyPredictor",
        )
        logger.info(
            "Hiring urgency CV accuracy – mean: %.4f ± %.4f",
            cv_urgency["mean"],
            cv_urgency["std"],
        )

        # Persist model
        urgency_model_path = os.path.join(models_dir, "hiring_urgency_model.joblib")
        joblib.dump(urgency_model, urgency_model_path)
        logger.info("Saved HiringUrgencyPredictor to %s", urgency_model_path)
    else:
        logger.warning("No hiring urgency target found – skipping urgency model training")
        urgency_metrics = {}

    # ------------------------------------------------------------------
    # 5. Persist preprocessing artefacts
    # ------------------------------------------------------------------
    joblib.dump(encoders, os.path.join(models_dir, "encoders.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    logger.info("Saved preprocessing artefacts to %s", models_dir)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    logger.info("=== Training Complete ===")
    if salary_metrics:
        logger.info(
            "Salary  – MAE: %.2f | RMSE: %.2f | R²: %.4f",
            salary_metrics["mae"],
            salary_metrics["rmse"],
            salary_metrics["r2"],
        )
    if urgency_metrics:
        logger.info(
            "Urgency – Accuracy: %.4f | F1: %.4f",
            urgency_metrics["accuracy"],
            urgency_metrics["f1"],
        )
    logger.info("All outputs saved to '%s'", args.output_dir)


if __name__ == "__main__":
    main()
