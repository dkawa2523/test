"""Preprocessing pipeline generation.

This module defines functions to build a collection of preprocessing pipelines
based on user configuration. Each pipeline is a scikit‑learn ``Pipeline`` or
``ColumnTransformer`` that transforms raw feature columns into formats suitable
for model training. The pipelines handle missing value imputation, scaling,
categorical encoding and optional polynomial feature expansion.
"""

from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer

from ..config import PreprocessingConfig

_ONE_HOT_ENCODER_KWARGS: Dict[str, object] = {"handle_unknown": "ignore"}
if "sparse_output" in inspect.signature(OneHotEncoder.__init__).parameters:
    _ONE_HOT_ENCODER_KWARGS["sparse_output"] = False
else:
    _ONE_HOT_ENCODER_KWARGS["sparse"] = False


def _to_numpy_array(data):
    return np.asarray(data)


def _build_numeric_pipeline(
    numeric_cols: List[str],
    imputation: Optional[str],
    scaling: Optional[str],
    polynomial_degree: Optional[int],
) -> Tuple[str, Pipeline]:
    """Create a numeric preprocessing pipeline.

    Parameters
    ----------
    numeric_cols : list of str
        Names of numeric columns.
    imputation : str or None
        Strategy for imputing missing values. One of 'mean', 'median',
        'most_frequent' or None.
    scaling : str or None
        Scaling strategy. One of 'standard', 'minmax', 'robust' or None.
    polynomial_degree : int or None
        Degree for PolynomialFeatures. If None or False, polynomial
        expansion is disabled.

    Returns
    -------
    tuple
        (name, pipeline) where name describes the pipeline components and
        pipeline is a scikit‑learn Pipeline object.
    """
    steps: List[Tuple[str, object]] = []
    name_parts: List[str] = []
    # Imputation
    if imputation:
        steps.append(("imputer", SimpleImputer(strategy=imputation)))
        name_parts.append(f"impute_{imputation}")
    # Scaling
    if scaling:
        scaler: object
        if scaling == "standard":
            scaler = StandardScaler()
        elif scaling == "minmax":
            scaler = MinMaxScaler()
        elif scaling == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling strategy: {scaling}")
        steps.append(("scaler", scaler))
        name_parts.append(f"scale_{scaling}")
    # Polynomial features
    if polynomial_degree and isinstance(polynomial_degree, int) and polynomial_degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=polynomial_degree, include_bias=False)))
        name_parts.append(f"poly_{polynomial_degree}")

    if not steps:
        # If no transformation specified, use 'passthrough'
        return "passthrough", "passthrough"
    return "_".join(name_parts), Pipeline(steps)


def _build_categorical_pipeline(
    categorical_cols: List[str],
    imputation: Optional[str],
    encoding: Optional[str],
) -> Tuple[str, Pipeline]:
    """Create a categorical preprocessing pipeline.

    Parameters
    ----------
    categorical_cols : list of str
        Names of categorical columns.
    imputation : str or None
        Strategy for imputing missing values ('most_frequent' or None).
    encoding : str or None
        Encoding strategy ('onehot', 'ordinal' or None).

    Returns
    -------
    tuple
        (name, pipeline) where name describes the pipeline and pipeline is a
        scikit‑learn Pipeline object.
    """
    steps: List[Tuple[str, object]] = []
    name_parts: List[str] = []

    if imputation:
        steps.append(("imputer", SimpleImputer(strategy=imputation)))
        name_parts.append(f"impute_{imputation}")

    if encoding:
        encoder: object
        if encoding == "onehot":
            encoder = OneHotEncoder(**_ONE_HOT_ENCODER_KWARGS)
        elif encoding == "ordinal":
            encoder = OrdinalEncoder()
        else:
            raise ValueError(f"Unknown categorical encoding: {encoding}")
        steps.append(("encoder", encoder))
        name_parts.append(f"encode_{encoding}")

    if not steps:
        return "passthrough", "passthrough"
    return "_".join(name_parts), Pipeline(steps)


def generate_preprocessors(
    config: PreprocessingConfig,
    feature_types: Dict[str, List[str]],
) -> List[Tuple[str, ColumnTransformer]]:
    """Generate a list of preprocessing ColumnTransformers based on config.

    Parameters
    ----------
    config : PreprocessingConfig
        The preprocessing configuration containing lists of strategies to try.
    feature_types : dict
        Dictionary with keys 'numeric' and 'categorical' listing column names.

    Returns
    -------
    list of tuples
        Each element is a (name, ColumnTransformer) pair representing a unique
        preprocessing pipeline.
    """
    numeric_cols = feature_types.get("numeric", [])
    cat_cols = feature_types.get("categorical", [])
    preprocessors: List[Tuple[str, ColumnTransformer]] = []

    # If there are no numeric or categorical columns, we still return at least one pipeline
    if not numeric_cols:
        numeric_imputations = [None]
        scalings = [None]
        degrees = [config.polynomial_degree if isinstance(config.polynomial_degree, int) else None]
    else:
        numeric_imputations = config.numeric_imputation or [None]
        scalings = config.scaling or [None]
        degrees: List[Optional[int]]
        if config.polynomial_degree and isinstance(config.polynomial_degree, int) and config.polynomial_degree > 1:
            degrees = [None, int(config.polynomial_degree)]
        else:
            degrees = [None]

    if not cat_cols:
        categorical_imputations = [None]
        encodings = [None]
    else:
        categorical_imputations = config.categorical_imputation or [None]
        encodings = config.categorical_encoding or [None]

    for num_imp in numeric_imputations:
        for scale in scalings:
            for degree in degrees:
                # Build numeric pipeline
                numeric_name, numeric_pipeline = _build_numeric_pipeline(
                    numeric_cols, num_imp, scale, degree
                )
                for cat_imp in categorical_imputations:
                    for enc in encodings:
                        cat_name, cat_pipeline = _build_categorical_pipeline(
                            cat_cols, cat_imp, enc
                        )
                        # Combine into ColumnTransformer
                        transformers = []
                        if numeric_cols:
                            transformers.append(("numeric", numeric_pipeline, numeric_cols))
                        if cat_cols:
                            transformers.append(("categorical", cat_pipeline, cat_cols))
                        if transformers:
                            ct = ColumnTransformer(
                                transformers=transformers,
                                remainder="drop",
                                verbose_feature_names_out=False,
                            )
                            name_parts = []
                            if numeric_name != "passthrough":
                                name_parts.append(numeric_name)
                            if cat_name != "passthrough":
                                name_parts.append(cat_name)
                            name = "|".join(name_parts) or "no_preproc"
                            preprocessors.append((name, ct))
                        else:
                            # No transformation; use identity transformer that ensures numpy output
                            preprocessors.append((
                                "no_preproc",
                                FunctionTransformer(_to_numpy_array, validate=False),
                            ))
    # Remove duplicates by names (rare but possible when both numeric and categorical are passthrough)
    unique = {}
    for name, ct in preprocessors:
        if name not in unique:
            unique[name] = ct
    return [(n, ct) for n, ct in unique.items()]
