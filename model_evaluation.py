"""
Model evaluation module for the AI & Data Science Job Market Analysis.

Provides functions to compute regression and classification metrics,
run cross-validation, and generate performance visualisation plots.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------


def evaluate_regression(
    y_true: Any,
    y_pred: Any,
    label: str = "Salary",
) -> Dict[str, float]:
    """Compute common regression metrics.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.
    label:
        Human-readable label used in logging.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys ``mae``, ``mse``, ``rmse``, ``r2``.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
    logger.info(
        "%s regression – MAE: %.4f | MSE: %.4f | RMSE: %.4f | R²: %.4f",
        label, mae, mse, rmse, r2,
    )
    return metrics


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------


def evaluate_classification(
    y_true: Any,
    y_pred: Any,
    label: str = "Hiring Urgency",
) -> Dict[str, Any]:
    """Compute common classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth class labels.
    y_pred:
        Predicted class labels.
    label:
        Human-readable label used in logging.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``confusion_matrix``, ``classification_report``.
    """
    avg = "weighted"
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    logger.info(
        "%s classification – Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
        label, accuracy, precision, recall, f1,
    )
    logger.info("\n%s", report)
    return metrics


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_model(
    model: Any,
    X: Any,
    y: Any,
    cv: int = 5,
    scoring: str = "r2",
    label: str = "Model",
) -> Dict[str, float]:
    """Run k-fold cross-validation and return summary statistics.

    Parameters
    ----------
    model:
        Unfitted or fitted scikit-learn estimator.
    X:
        Full feature matrix.
    y:
        Target vector.
    cv:
        Number of folds.
    scoring:
        Scoring metric (e.g. ``'r2'``, ``'accuracy'``).
    label:
        Human-readable model name for logging.

    Returns
    -------
    Dict[str, float]
        Keys ``mean``, ``std``, ``min``, ``max`` of CV scores.
    """
    logger.info("Running %d-fold CV on %s with scoring='%s' …", cv, label, scoring)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    summary = {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
    logger.info(
        "%s CV %s – mean: %.4f ± %.4f (min: %.4f, max: %.4f)",
        label, scoring, summary["mean"], summary["std"], summary["min"], summary["max"],
    )
    return summary


# ---------------------------------------------------------------------------
# Feature importance visualisation
# ---------------------------------------------------------------------------


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Importances",
    top_n: int = 20,
    output_path: Optional[str] = None,
) -> None:
    """Bar chart of the top-N most important features.

    Parameters
    ----------
    importances:
        Array of feature importance values.
    feature_names:
        Corresponding feature names.
    title:
        Plot title.
    top_n:
        Number of features to show.
    output_path:
        If provided, the figure is saved to this path.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n // 2)))
    ax.barh(top_names[::-1], top_vals[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=120)
        logger.info("Saved feature importance plot to %s", output_path)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Regression performance plots
# ---------------------------------------------------------------------------


def plot_regression_results(
    y_true: Any,
    y_pred: Any,
    title: str = "Actual vs Predicted Salary",
    output_path: Optional[str] = None,
) -> None:
    """Scatter plot of actual vs predicted values for regression.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.
    title:
        Plot title.
    output_path:
        If provided, the figure is saved to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs. predicted
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="none", color="steelblue")
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()

    # Residuals
    residuals = np.array(y_true) - np.array(y_pred)
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals, alpha=0.4, edgecolors="none", color="darkorange")
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual Plot")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=120)
        logger.info("Saved regression results plot to %s", output_path)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    class_labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix – Hiring Urgency",
    output_path: Optional[str] = None,
) -> None:
    """Heatmap of the confusion matrix for a classification model.

    Parameters
    ----------
    y_true:
        Ground-truth class labels.
    y_pred:
        Predicted class labels.
    class_labels:
        Optional list of human-readable class names.
    title:
        Plot title.
    output_path:
        If provided, the figure is saved to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels or "auto",
        yticklabels=class_labels or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=120)
        logger.info("Saved confusion matrix plot to %s", output_path)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined evaluation report
# ---------------------------------------------------------------------------


def full_regression_report(
    y_true: Any,
    y_pred: Any,
    importances: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a full regression evaluation and optionally save plots.

    Parameters
    ----------
    y_true:
        Ground-truth salary values.
    y_pred:
        Predicted salary values.
    importances:
        Feature importance values from the trained model.
    feature_names:
        Feature names corresponding to ``importances``.
    output_dir:
        Directory to save plots. If None, plots are shown interactively.

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary from :func:`evaluate_regression`.
    """
    metrics = evaluate_regression(y_true, y_pred)

    reg_path = os.path.join(output_dir, "salary_regression.png") if output_dir else None
    plot_regression_results(y_true, y_pred, output_path=reg_path)

    if importances is not None and feature_names is not None:
        fi_path = os.path.join(output_dir, "salary_feature_importance.png") if output_dir else None
        plot_feature_importance(
            importances,
            feature_names,
            title="Salary Model – Feature Importances",
            output_path=fi_path,
        )

    return metrics


def full_classification_report(
    y_true: Any,
    y_pred: Any,
    class_labels: Optional[List[str]] = None,
    importances: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a full classification evaluation and optionally save plots.

    Parameters
    ----------
    y_true:
        Ground-truth hiring urgency labels.
    y_pred:
        Predicted hiring urgency labels.
    class_labels:
        Human-readable class names.
    importances:
        Feature importance values from the trained model.
    feature_names:
        Feature names corresponding to ``importances``.
    output_dir:
        Directory to save plots. If None, plots are shown interactively.

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary from :func:`evaluate_classification`.
    """
    metrics = evaluate_classification(y_true, y_pred)

    cm_path = os.path.join(output_dir, "hiring_urgency_confusion.png") if output_dir else None
    plot_confusion_matrix(y_true, y_pred, class_labels=class_labels, output_path=cm_path)

    if importances is not None and feature_names is not None:
        fi_path = (
            os.path.join(output_dir, "hiring_urgency_feature_importance.png")
            if output_dir
            else None
        )
        plot_feature_importance(
            importances,
            feature_names,
            title="Hiring Urgency Model – Feature Importances",
            output_path=fi_path,
        )

    return metrics
