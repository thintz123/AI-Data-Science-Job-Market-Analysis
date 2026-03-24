"""
Exploratory Data Analysis (EDA) module for the AI & Data Science Job Market Analysis.

Provides functions to summarise data, analyse distributions and correlations,
and produce visualisations that aid understanding before model training.
"""

import logging
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def print_data_summary(df: pd.DataFrame) -> None:
    """Log basic dataset statistics.

    Parameters
    ----------
    df:
        The raw or cleaned dataframe to summarise.
    """
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Column dtypes:\n%s", df.dtypes.to_string())
    logger.info("Missing values per column:\n%s", df.isnull().sum().to_string())
    logger.info("Descriptive statistics:\n%s", df.describe(include="all").to_string())


def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe summarising missing values.

    Parameters
    ----------
    df:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Columns: ``missing_count``, ``missing_pct``.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    report = pd.DataFrame(
        {"missing_count": missing, "missing_pct": missing_pct}
    ).sort_values("missing_pct", ascending=False)
    return report[report["missing_count"] > 0]


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------


def plot_numeric_distributions(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> None:
    """Plot histograms with KDE for numeric columns.

    Parameters
    ----------
    df:
        Dataframe containing the columns to plot.
    numeric_cols:
        Explicit list of numeric columns. Defaults to all numeric columns.
    output_path:
        If provided, the figure is saved to this path.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        logger.warning("No numeric columns to plot")
        return

    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(col)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric Feature Distributions", fontsize=14, y=1.02)
    plt.tight_layout()

    _save_or_show(fig, output_path, "numeric distributions")


def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    max_categories: int = 20,
    output_path: Optional[str] = None,
) -> None:
    """Plot bar charts of value counts for categorical columns.

    Parameters
    ----------
    df:
        Dataframe containing the columns to plot.
    categorical_cols:
        Explicit list of categorical columns. Defaults to all object columns.
    max_categories:
        Maximum number of categories to display per column.
    output_path:
        If provided, the figure is saved to this path.
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

    if not categorical_cols:
        logger.warning("No categorical columns to plot")
        return

    n_cols = 2
    n_rows = int(np.ceil(len(categorical_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols // 2, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(categorical_cols):
        counts = df[col].value_counts().head(max_categories)
        axes[i].bar(counts.index.astype(str), counts.values, color="steelblue")
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Categorical Feature Distributions", fontsize=14, y=1.02)
    plt.tight_layout()

    _save_or_show(fig, output_path, "categorical distributions")


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    title: str = "Feature Correlation Heatmap",
    output_path: Optional[str] = None,
) -> None:
    """Heatmap of Pearson correlations for numeric columns.

    Parameters
    ----------
    df:
        Dataframe to analyse.
    numeric_cols:
        Columns to include. Defaults to all numeric columns.
    title:
        Plot title.
    output_path:
        If provided, the figure is saved to this path.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        logger.warning("Need at least 2 numeric columns for a correlation heatmap")
        return

    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) - 2)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    plt.tight_layout()

    _save_or_show(fig, output_path, "correlation heatmap")


def get_top_correlations(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 10,
    numeric_cols: Optional[List[str]] = None,
) -> pd.Series:
    """Return the top-N features most correlated with ``target_col``.

    Parameters
    ----------
    df:
        Dataframe to analyse.
    target_col:
        Column to compare against.
    top_n:
        Number of features to return.
    numeric_cols:
        Columns to include. Defaults to all numeric columns.

    Returns
    -------
    pd.Series
        Absolute correlations sorted descending.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if target_col not in df.columns:
        logger.warning("Target column '%s' not found", target_col)
        return pd.Series(dtype=float)

    corr = df[numeric_cols].corr()[target_col].drop(target_col, errors="ignore")
    return corr.abs().sort_values(ascending=False).head(top_n)


# ---------------------------------------------------------------------------
# Salary & hiring urgency specific plots
# ---------------------------------------------------------------------------


def plot_salary_by_category(
    df: pd.DataFrame,
    category_col: str,
    salary_col: str = "salary",
    max_categories: int = 15,
    output_path: Optional[str] = None,
) -> None:
    """Box plot of salary distribution by a categorical variable.

    Parameters
    ----------
    df:
        Dataframe with salary and category columns.
    category_col:
        Name of the categorical column to group by.
    salary_col:
        Name of the salary column.
    max_categories:
        Maximum number of categories to display.
    output_path:
        If provided, the figure is saved to this path.
    """
    if salary_col not in df.columns or category_col not in df.columns:
        logger.warning("Required columns not found for salary-by-category plot")
        return

    top_cats = df[category_col].value_counts().head(max_categories).index
    subset = df[df[category_col].isin(top_cats)]

    fig, ax = plt.subplots(figsize=(12, 6))
    subset.boxplot(column=salary_col, by=category_col, ax=ax, grid=False)
    ax.set_title(f"Salary Distribution by {category_col}")
    plt.suptitle("")
    ax.set_xlabel(category_col)
    ax.set_ylabel(salary_col)
    plt.xticks(rotation=45)
    plt.tight_layout()

    _save_or_show(fig, output_path, f"salary by {category_col}")


def plot_hiring_urgency_distribution(
    df: pd.DataFrame,
    urgency_col: str = "hiring_urgency",
    output_path: Optional[str] = None,
) -> None:
    """Count plot showing the distribution of hiring urgency classes.

    Parameters
    ----------
    df:
        Dataframe with the hiring urgency column.
    urgency_col:
        Name of the hiring urgency column.
    output_path:
        If provided, the figure is saved to this path.
    """
    if urgency_col not in df.columns:
        logger.warning("Hiring urgency column '%s' not found", urgency_col)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[urgency_col].value_counts()
    ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.set_title("Hiring Urgency Distribution")
    ax.set_xlabel(urgency_col)
    ax.set_ylabel("Count")
    plt.tight_layout()

    _save_or_show(fig, output_path, "hiring urgency distribution")


# ---------------------------------------------------------------------------
# Full EDA runner
# ---------------------------------------------------------------------------


def run_eda(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """Run a comprehensive EDA and optionally save all plots.

    Parameters
    ----------
    df:
        The raw dataset.
    output_dir:
        Directory to save all generated plots. If None, plots are shown
        interactively.
    """
    print_data_summary(df)

    missing_report = get_missing_value_report(df)
    if not missing_report.empty:
        logger.info("Missing value report:\n%s", missing_report.to_string())

    _make_dir(output_dir)

    plot_numeric_distributions(
        df,
        output_path=os.path.join(output_dir, "numeric_distributions.png") if output_dir else None,
    )

    plot_categorical_distributions(
        df,
        output_path=(
            os.path.join(output_dir, "categorical_distributions.png") if output_dir else None
        ),
    )

    plot_correlation_heatmap(
        df,
        output_path=os.path.join(output_dir, "correlation_heatmap.png") if output_dir else None,
    )

    if "salary" in df.columns:
        plot_salary_by_category(
            df,
            "experience_level",
            output_path=(
                os.path.join(output_dir, "salary_by_experience_level.png")
                if output_dir
                else None
            ),
        )

    if "hiring_urgency" in df.columns:
        plot_hiring_urgency_distribution(
            df,
            output_path=(
                os.path.join(output_dir, "hiring_urgency_distribution.png")
                if output_dir
                else None
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_or_show(fig: plt.Figure, output_path: Optional[str], label: str) -> None:
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("Saved %s plot to %s", label, output_path)
    else:
        plt.show()
    plt.close(fig)


def _make_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)
