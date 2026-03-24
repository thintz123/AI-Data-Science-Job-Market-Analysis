"""
Model definitions for the AI & Data Science Job Market Analysis.

Provides:
- ``SalaryPredictor``  – Gradient Boosting Regressor for salary prediction.
- ``HiringUrgencyPredictor`` – Gradient Boosting Classifier for hiring urgency.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_REGRESSOR_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "subsample": 0.8,
    "max_features": "sqrt",
    "random_state": 42,
    "loss": "squared_error",
}

DEFAULT_CLASSIFIER_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "subsample": 0.8,
    "max_features": "sqrt",
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Salary predictor (regression)
# ---------------------------------------------------------------------------


class SalaryPredictor:
    """Gradient Boosting Regressor that predicts salary.

    Parameters
    ----------
    params:
        Optional dict of hyperparameters to override the defaults.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        model_params = {**DEFAULT_REGRESSOR_PARAMS, **(params or {})}
        self.model = GradientBoostingRegressor(**model_params)
        self.is_trained: bool = False
        self.feature_names: Optional[list] = None

    def train(self, X: Any, y: Any, feature_names: Optional[list] = None) -> "SalaryPredictor":
        """Fit the model on training data.

        Parameters
        ----------
        X:
            Feature matrix (array-like or DataFrame).
        y:
            Target salary values.
        feature_names:
            Optional list of feature names for later inspection.

        Returns
        -------
        SalaryPredictor
            Self, to allow method chaining.
        """
        logger.info("Training SalaryPredictor on %d samples …", len(y))
        self.feature_names = feature_names
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("SalaryPredictor training complete")
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict salary for the given feature matrix.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted salary values.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict()")
        return self.model.predict(X)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances from the trained model."""
        if not self.is_trained:
            return None
        return self.model.feature_importances_

    @property
    def params(self) -> Dict[str, Any]:
        """Return the hyperparameter configuration."""
        return self.model.get_params()


# ---------------------------------------------------------------------------
# Hiring urgency predictor (classification)
# ---------------------------------------------------------------------------


class HiringUrgencyPredictor:
    """Gradient Boosting Classifier that predicts hiring urgency.

    Parameters
    ----------
    params:
        Optional dict of hyperparameters to override the defaults.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        model_params = {**DEFAULT_CLASSIFIER_PARAMS, **(params or {})}
        self.model = GradientBoostingClassifier(**model_params)
        self.is_trained: bool = False
        self.feature_names: Optional[list] = None
        self.classes_: Optional[np.ndarray] = None

    def train(
        self, X: Any, y: Any, feature_names: Optional[list] = None
    ) -> "HiringUrgencyPredictor":
        """Fit the model on training data.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target hiring-urgency labels.
        feature_names:
            Optional list of feature names.

        Returns
        -------
        HiringUrgencyPredictor
            Self, to allow method chaining.
        """
        logger.info("Training HiringUrgencyPredictor on %d samples …", len(y))
        self.feature_names = feature_names
        self.model.fit(X, y)
        self.is_trained = True
        self.classes_ = self.model.classes_
        logger.info(
            "HiringUrgencyPredictor training complete – classes: %s", self.classes_
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict hiring urgency class for the given feature matrix.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict()")
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities for the given feature matrix.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            Class probability matrix.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict_proba()")
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances from the trained model."""
        if not self.is_trained:
            return None
        return self.model.feature_importances_

    @property
    def params(self) -> Dict[str, Any]:
        """Return the hyperparameter configuration."""
        return self.model.get_params()
