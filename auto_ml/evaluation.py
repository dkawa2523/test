"""Model evaluation utilities for the AutoML library.

This module defines functions to evaluate multiple preprocessing and model
combinations using cross‑validation. It abstracts away the selection of
appropriate scoring metrics and cross‑validation splitters based on the
problem type (classification vs. regression) and returns a tidy
DataFrame summarizing the results.

The evaluation is designed to work with small datasets where hold‑out
test sets may not be meaningful. K‑fold cross‑validation is the default;
for classification tasks the splitter is stratified to preserve class
distributions. Users can configure the number of folds and shuffling via
the ``CVConfig`` object loaded from the configuration file.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from .config import CVConfig
from .model_factory import ModelInstance


def _get_cv_splitter(
    problem_type: str, n_samples: int, cv_config: CVConfig, y: Iterable | None
) -> object:
    """Construct a cross‑validation splitter.

    Parameters
    ----------
    problem_type : str
        'classification' or 'regression'. Determines whether to use
        StratifiedKFold.
    n_samples : int
        Number of samples in the dataset; used to choose a default number
        of folds when ``cv_config.n_folds`` is None.
    cv_config : CVConfig
        Configuration specifying the number of folds, whether to shuffle and
        the random seed.
    y : iterable, optional
        Target values. Required for stratified splitting.

    Returns
    -------
    object
        A scikit‑learn cross‑validator.
    """
    problem_type = problem_type.lower()
    n_folds: int
    if cv_config.n_folds is None:
        # Use leave‑one‑out when n_samples is very small; otherwise min(5, n_samples)
        if n_samples < 3:
            n_folds = n_samples
        else:
            n_folds = min(5, n_samples)
    else:
        n_folds = cv_config.n_folds
    if problem_type == "classification":
        if y is None:
            raise ValueError("Target values are required for stratified cross‑validation")
        return StratifiedKFold(
            n_splits=n_folds, shuffle=cv_config.shuffle, random_state=cv_config.random_seed
        )
    else:
        return KFold(
            n_splits=n_folds, shuffle=cv_config.shuffle, random_state=cv_config.random_seed
        )


def _get_scoring(problem_type: str, metrics: Iterable[str]) -> Dict[str, str]:
    """Create a scoring dictionary based on requested metrics.

    Parameters
    ----------
    problem_type : str
        'regression' or 'classification'. Determines available scoring keys.
    metrics : iterable of str
        Names of metrics requested by the user. Supported regression metrics:
        'mae', 'mse', 'rmse', 'r2'; classification metrics: 'accuracy',
        'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr'.

    Returns
    -------
    dict
        Mapping from internal metric names to scikit‑learn scoring strings.
    """
    problem_type = problem_type.lower()
    scoring: Dict[str, str] = {}
    if problem_type == "regression":
        for m in metrics:
            if m == "mae":
                scoring[m] = "neg_mean_absolute_error"
            elif m == "mse" or m == "rmse":
                scoring["mse"] = "neg_mean_squared_error"
            elif m == "r2":
                scoring[m] = "r2"
            else:
                raise ValueError(f"Unsupported regression metric: {m}")
    else:
        for m in metrics:
            if m in {"accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"}:
                scoring[m] = m
            else:
                raise ValueError(f"Unsupported classification metric: {m}")
    return scoring


def evaluate_model_combinations(
    preprocessors: List[Tuple[str, object]],
    models: List[ModelInstance],
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    cv_config: CVConfig,
    problem_type: str,
    metrics: Iterable[str],
) -> pd.DataFrame:
    """Evaluate all (preprocessor, model) combinations using cross‑validation.

    Parameters
    ----------
    preprocessors : list of (str, transformer)
        Each tuple contains a descriptive name and a scikit‑learn transformer
        (typically a ColumnTransformer) that prepares the features.
    models : list of ModelInstance
        Instantiated models with hyperparameter settings from the factory.
    X : array‑like of shape (n_samples, n_features)
        Feature matrix for training.
    y : array‑like of shape (n_samples,)
        Target values.
    cv_config : CVConfig
        Cross‑validation configuration loaded from YAML.
    problem_type : str
        'regression' or 'classification'. Determines scoring metrics and
        splitting strategy.

    Returns
    -------
    pandas.DataFrame
        A tidy table with one row per combination and mean scores for each
        metric. Negative scores are negated so that larger is better for
        all metrics.
    """
    scoring = _get_scoring(problem_type, metrics)
    # Construct cross‑validator once to reuse splits for all combinations.
    cv = _get_cv_splitter(problem_type, len(y), cv_config, y)
    records: List[Dict[str, Any]] = []
    for preproc_name, transformer in preprocessors:
        for model in models:
            # Build a pipeline combining the preprocessor and estimator.
            pipe = Pipeline([
                ("preprocessor", transformer),
                ("model", model.estimator),
            ])
            try:
                cv_results = cross_validate(
                    pipe,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    error_score="raise",
                    n_jobs=None,
                )
            except Exception as e:
                # If the model cannot be evaluated on this data (e.g., target
                # type mismatch), record the failure and continue.
                record = {
                    "preprocessor": preproc_name,
                    "model": model.name,
                    "params": model.params,
                    "error": str(e),
                }
                records.append(record)
                continue
            result = {
                "preprocessor": preproc_name,
                "model": model.name,
                "params": model.params,
            }
            # Aggregate scores; for negative losses, negate the mean
            for metric_name, scores in cv_results.items():
                if not metric_name.startswith("test_"):
                    continue
                simple_name = metric_name.replace("test_", "")
                mean_score = np.mean(scores)
                # Convert sign for error metrics where higher is worse
                if simple_name in {"mae", "mse"}:
                    mean_score = -mean_score
                result[simple_name] = mean_score
            # If RMSE requested explicitly, compute from MSE
            if "rmse" in metrics and "mse" in result:
                result["rmse"] = float(np.sqrt(result["mse"]))
            records.append(result)
    df = pd.DataFrame(records)
    return df