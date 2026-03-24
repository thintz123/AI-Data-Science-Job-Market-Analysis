"""
Data preprocessing module for the AI & Data Science Job Market Analysis.

Handles loading, cleaning, encoding, scaling, and splitting the dataset
for both salary regression and hiring urgency classification tasks.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column configuration
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: List[str] = [
    "years_experience",
    "num_openings",
    "job_posting_year",
]

BINARY_SKILL_FEATURES: List[str] = [
    "requires_python",
    "requires_sql",
    "requires_ml",
    "requires_deep_learning",
    "requires_cloud",
]

CATEGORICAL_FEATURES: List[str] = [
    "experience_level",
    "education_level",
    "job_title",
    "company_industry",
    "country",
    "company_size",
]

SALARY_TARGET: str = "salary"
HIRING_URGENCY_TARGET: str = "hiring_urgency"


def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Parameters
    ----------
    filepath:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning steps.

    - Remove duplicate rows.
    - Strip whitespace from string columns.
    - Replace common missing-value placeholders with NaN.

    Parameters
    ----------
    df:
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    logger.info("Cleaning data …")
    df = df.copy()

    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("Removed %d duplicate rows", before - len(df))

    # Normalise string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Replace common NA placeholders
    df.replace(["N/A", "n/a", "NA", "na", "None", "none", "null", ""], np.nan, inplace=True)

    return df


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> pd.DataFrame:
    """Impute missing values in numeric and categorical columns.

    Parameters
    ----------
    df:
        Cleaned dataframe.
    numeric_strategy:
        Imputation strategy for numeric columns (default: 'median').
    categorical_strategy:
        Imputation strategy for categorical columns (default: 'most_frequent').

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values imputed.
    """
    logger.info("Handling missing values …")
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if num_cols:
        num_imputer = SimpleImputer(strategy=numeric_strategy)
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy=categorical_strategy)
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode categorical columns.

    Parameters
    ----------
    df:
        Dataframe after missing-value handling.
    categorical_cols:
        List of column names to encode. Defaults to ``CATEGORICAL_FEATURES``.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, LabelEncoder]]
        Encoded dataframe and a mapping of column name → fitted LabelEncoder.
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES

    logger.info("Encoding categorical features: %s", categorical_cols)
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for col in categorical_cols:
        if col not in df.columns:
            logger.warning("Column '%s' not found – skipping encoding", col)
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def scale_numeric_features(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standard-scale numeric feature columns (zero mean, unit variance).

    Parameters
    ----------
    df:
        Dataframe with encoded features.
    numeric_cols:
        Columns to scale. Defaults to ``NUMERIC_FEATURES``.

    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler]
        Scaled dataframe and the fitted scaler.
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_FEATURES

    cols_present = [c for c in numeric_cols if c in df.columns]
    if not cols_present:
        logger.warning("No numeric columns to scale – returning unchanged dataframe")
        return df, StandardScaler()

    logger.info("Scaling numeric features: %s", cols_present)
    df = df.copy()
    scaler = StandardScaler()
    df[cols_present] = scaler.fit_transform(df[cols_present])
    return df, scaler


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Select the feature columns used for modelling.

    Parameters
    ----------
    df:
        Preprocessed dataframe.
    feature_cols:
        Explicit list of feature columns. Defaults to all numeric, binary
        skill, and categorical feature columns.

    Returns
    -------
    pd.DataFrame
        Feature matrix.
    """
    if feature_cols is None:
        feature_cols = NUMERIC_FEATURES + BINARY_SKILL_FEATURES + CATEGORICAL_FEATURES

    cols_present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("Feature columns not found in dataframe: %s", missing)

    return df[cols_present]


def preprocess(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """End-to-end preprocessing pipeline.

    Loads data, cleans it, handles missing values, encodes and scales
    features, and splits into train/test sets for both tasks.

    Parameters
    ----------
    filepath:
        Path to the raw CSV dataset.
    test_size:
        Fraction of data reserved for testing.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``X_train``, ``X_test``,
        ``y_salary_train``, ``y_salary_test``,
        ``y_urgency_train``, ``y_urgency_test``,
        ``encoders``, ``scaler``, ``feature_names``
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = handle_missing_values(df)
    df, encoders = encode_categorical_features(df)
    df, scaler = scale_numeric_features(df)

    X = build_feature_matrix(df)
    feature_names = X.columns.tolist()

    results: Dict = {
        "encoders": encoders,
        "scaler": scaler,
        "feature_names": feature_names,
    }

    # --- Salary regression ---
    if SALARY_TARGET in df.columns:
        y_salary = df[SALARY_TARGET]
        X_tr, X_te, ys_tr, ys_te = train_test_split(
            X, y_salary, test_size=test_size, random_state=random_state
        )
        results.update(
            {
                "X_train": X_tr,
                "X_test": X_te,
                "y_salary_train": ys_tr,
                "y_salary_test": ys_te,
            }
        )
        logger.info(
            "Salary split – train: %d, test: %d", len(ys_tr), len(ys_te)
        )
    else:
        logger.warning("Salary column '%s' not found", SALARY_TARGET)

    # --- Hiring urgency classification ---
    if HIRING_URGENCY_TARGET in df.columns:
        y_urgency = df[HIRING_URGENCY_TARGET]
        X_tr2, X_te2, yu_tr, yu_te = train_test_split(
            X, y_urgency, test_size=test_size, random_state=random_state
        )
        if "X_train" not in results:
            results["X_train"] = X_tr2
            results["X_test"] = X_te2
        results.update(
            {
                "y_urgency_train": yu_tr,
                "y_urgency_test": yu_te,
            }
        )
        logger.info(
            "Hiring urgency split – train: %d, test: %d", len(yu_tr), len(yu_te)
        )
    else:
        logger.warning("Hiring urgency column '%s' not found", HIRING_URGENCY_TARGET)

    return results
