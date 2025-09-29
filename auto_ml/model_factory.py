"""Model factory for the AutoML library.

This module maps human‑readable model names from the configuration file to
their corresponding estimator classes and parameter grids. It supports
regression and classification tasks and includes optional third‑party
models when the required libraries are available. The factory functions
return a list of instantiated estimators along with their parameter
combinations so that each configuration can be evaluated independently.

The design emphasizes extensibility: to add a new model, define its
mapping in the appropriate dictionary and, if necessary, import it
dynamically in ``_optional_imports``. Each model specification in the
configuration is expanded into all combinations of provided hyperparameters
using scikit‑learn's ``ParameterGrid``.

Notes
-----
* Scikit‑learn serves as the backbone for most estimators. Additional
  libraries like LightGBM, XGBoost, CatBoost, TabNet and TabPFN are
  imported conditionally so that the code still runs when they are
  absent. Users can install these packages to enable the extra models.
* The ``problem_type`` determines whether to instantiate regressor or
  classifier variants when both are available (e.g. RandomForestRegressor
  vs RandomForestClassifier). Classification names are normalized to
  lowercase to avoid mismatches.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np

import warnings
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from .config import ModelSpec
from .tabpfn_utils import build_fallback_regressor_spec


def _optional_imports() -> Dict[str, Any]:
    """Attempt to import optional libraries and return a mapping.

    Returns
    -------
    dict
        A dictionary where keys are library identifiers and values are the
        imported modules or ``None`` if the import failed. The keys used
        here correspond to the names used in model mappings below.
    """
    mods: Dict[str, Any] = {}
    # LightGBM
    try:
        import lightgbm as lgb  # type: ignore
        mods["lightgbm"] = lgb
    except Exception:
        mods["lightgbm"] = None
    # XGBoost
    try:
        import xgboost as xgb  # type: ignore
        mods["xgboost"] = xgb
    except Exception:
        mods["xgboost"] = None
    # CatBoost
    try:
        import catboost  # type: ignore
        mods["catboost"] = catboost
    except Exception:
        mods["catboost"] = None
    # PyTorch TabNet
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier  # type: ignore

        class _TabNetMixin:
            """Common helpers to use tiny tabular datasets safely."""

            @staticmethod
            def _n_samples(X_train: Any) -> int:
                if hasattr(X_train, "shape") and X_train.shape is not None and len(X_train.shape) > 0:
                    return int(X_train.shape[0])
                return len(X_train)

            @classmethod
            def _normalize_train_kwargs(cls, n_samples: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
                params = dict(kwargs)
                batch_size = params.get("batch_size")
                if batch_size is None:
                    suggested = max(8, n_samples // 2)
                    batch_size = min(256, suggested)
                batch_size = max(1, min(batch_size, n_samples))
                params["batch_size"] = batch_size

                virtual_bs = params.get("virtual_batch_size")
                if virtual_bs is None:
                    half = batch_size // 2 if batch_size > 1 else 1
                    virtual_bs = max(1, min(32, half))
                virtual_bs = max(1, min(virtual_bs, batch_size))
                params["virtual_batch_size"] = virtual_bs

                drop_last = params.get("drop_last", True)
                if n_samples <= batch_size:
                    drop_last = False
                params["drop_last"] = drop_last
                return params

        class _PatchedTabNetRegressor(_TabNetMixin, TabNetRegressor):  # type: ignore
            """Ensure TabNet handles 1D targets and tiny batches."""

            def fit(self, X_train, y_train, **kwargs):  # type: ignore[override]
                y_array = np.asarray(y_train)
                if y_array.ndim == 1:
                    y_array = y_array.reshape(-1, 1)
                params = self._normalize_train_kwargs(self._n_samples(X_train), kwargs)
                return super().fit(X_train, y_array, **params)

            def predict(self, X):  # type: ignore[override]
                preds = super().predict(X)
                if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] == 1:
                    return preds.ravel()
                return preds

        class _PatchedTabNetClassifier(_TabNetMixin, TabNetClassifier):  # type: ignore
            """Adjust training kwargs for small datasets."""

            def fit(self, X_train, y_train, **kwargs):  # type: ignore[override]
                y_array = np.asarray(y_train)
                params = self._normalize_train_kwargs(self._n_samples(X_train), kwargs)
                return super().fit(X_train, y_array, **params)

        mods["tabnet_regressor"] = _PatchedTabNetRegressor
        mods["tabnet_classifier"] = _PatchedTabNetClassifier
    except Exception:
        mods["tabnet_regressor"] = None
        mods["tabnet_classifier"] = None
    # TabPFN
    try:
        home = os.environ.get("TABPFN_HOME")
        if not home:
            default_home = Path(__file__).resolve().parent.parent / ".tabpfn_home"
            default_home.mkdir(parents=True, exist_ok=True)
            os.environ["TABPFN_HOME"] = str(default_home)
            os.environ.setdefault("TABPFN_STATE_DIR", str(default_home))
            os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(default_home / "model_cache"))
        else:
            Path(home).mkdir(parents=True, exist_ok=True)
            if not os.environ.get("TABPFN_STATE_DIR"):
                os.environ["TABPFN_STATE_DIR"] = home
            os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(Path(home) / "model_cache"))
        from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore
        mods["tabpfn_classifier"] = TabPFNClassifier
        mods["tabpfn_regressor"] = TabPFNRegressor
    except Exception:
        mods["tabpfn_classifier"] = None
        mods["tabpfn_regressor"] = None
    return mods


_OPTIONAL_MODULES: Dict[str, Any] = _optional_imports()


_TABPFN_DEFAULT_MODEL_FILES = {
    "classification": "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
    "regression": "tabpfn-v2-regressor.ckpt",
}

_TABPFN_MODEL_URLS = {
    "classification": "https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/main/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
    "regression": "https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt",
}


def _get_tabpfn_cached_model_path(problem_type: str) -> Optional[Path]:
    """Return local path to the default TabPFN weight file if it exists."""

    filename = _TABPFN_DEFAULT_MODEL_FILES.get(problem_type.lower())
    if not filename:
        return None
    cache_dir_str = os.environ.get("TABPFN_MODEL_CACHE_DIR")
    if cache_dir_str:
        cache_dir = Path(cache_dir_str)
    else:
        cache_dir = Path.home() / ".cache" / "tabpfn"
    candidate = cache_dir / filename
    return candidate if candidate.exists() else None


def prepare_tabpfn_params(problem_type: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize TabPFN parameters and verify that weights are available.

    Returns the updated parameter dict or ``None`` when the model should be
    skipped (e.g. missing weights).
    """

    normalized = dict(params)
    normalized.setdefault("device", "cpu")
    normalized.setdefault("n_jobs", 1)
    problem_key = "classification" if problem_type.lower() == "classification" else "regression"
    model_path_val = normalized.get("model_path")
    if model_path_val is None or (isinstance(model_path_val, str) and model_path_val.lower() == "auto"):
        cached_path = _get_tabpfn_cached_model_path(problem_key)
        if cached_path is None:
            if problem_key == "regression":
                spec = build_fallback_regressor_spec()
                if spec is not None:
                    normalized.pop("model_path", None)
                    normalized.setdefault("ignore_pretraining_limits", True)
                    normalized.setdefault("random_state", 0)
                    normalized["use_fallback_tabpfn"] = True
                    return normalized
                url = _TABPFN_MODEL_URLS.get(problem_key, "")
                location = os.environ.get("TABPFN_MODEL_CACHE_DIR", "TABPFN_MODEL_CACHE_DIR")
                warnings.warn(
                    "Skipping TabPFN because pretrained weights were not found locally. "
                    f"Download the model from {url} and place it under {location} or specify "
                    "'model_path' in the configuration."
                )
                return None
            url = _TABPFN_MODEL_URLS.get(problem_key, "")
            location = os.environ.get("TABPFN_MODEL_CACHE_DIR", "TABPFN_MODEL_CACHE_DIR")
            warnings.warn(
                "Skipping TabPFN because pretrained weights were not found locally. "
                f"Download the model from {url} and place it under {location} or specify "
                "'model_path' in the configuration."
            )
            return None
        normalized["model_path"] = str(cached_path)
    elif isinstance(model_path_val, str):
        candidate = Path(model_path_val).expanduser()
        if not candidate.exists():
            warnings.warn(
                f"Skipping TabPFN because model_path '{model_path_val}' does not exist."
            )
            return None
    return normalized


def _get_model_class(name: str, problem_type: str):
    """Resolve a model name to an estimator class based on the problem type.

    Parameters
    ----------
    name : str
        Name of the model as specified in the configuration. The comparison
        is case‑insensitive.
    problem_type : str
        Either ``'regression'`` or ``'classification'``.

    Returns
    -------
    type
        The estimator class corresponding to the name and problem type.

    Raises
    ------
    KeyError
        If the name is unknown or unsupported for the given problem type.
    """
    name_lower = name.lower()
    problem_type = problem_type.lower()

    # Base scikit‑learn models for regression and classification.
    if problem_type == "regression":
        mapping: Dict[str, str] = {
            "linearregression": "sklearn.linear_model.LinearRegression",
            "ridge": "sklearn.linear_model.Ridge",
            "lasso": "sklearn.linear_model.Lasso",
            "elasticnet": "sklearn.linear_model.ElasticNet",
            "svr": "sklearn.svm.SVR",
            "knn": "sklearn.neighbors.KNeighborsRegressor",
            "kneighbors": "sklearn.neighbors.KNeighborsRegressor",
            "randomforest": "sklearn.ensemble.RandomForestRegressor",
            "extratrees": "sklearn.ensemble.ExtraTreesRegressor",
            "gradientboosting": "sklearn.ensemble.GradientBoostingRegressor",
            "gaussianprocess": "sklearn.gaussian_process.GaussianProcessRegressor",
            "mlp": "sklearn.neural_network.MLPRegressor",
        }
        # Optional models – only included if the library is available
        if _OPTIONAL_MODULES.get("lightgbm") is not None:
            mapping["lightgbm"] = "lightgbm.LGBMRegressor"
        if _OPTIONAL_MODULES.get("xgboost") is not None:
            mapping["xgboost"] = "xgboost.XGBRegressor"
        if _OPTIONAL_MODULES.get("catboost") is not None:
            mapping["catboost"] = "catboost.CatBoostRegressor"
        if _OPTIONAL_MODULES.get("tabnet_regressor") is not None:
            mapping["tabnet"] = "tabnet_regressor"
        if _OPTIONAL_MODULES.get("tabpfn_regressor") is not None:
            mapping["tabpfn"] = "tabpfn_regressor"
    else:  # classification
        mapping = {
            "logisticregression": "sklearn.linear_model.LogisticRegression",
            "ridge": "sklearn.linear_model.RidgeClassifier",
            "svm": "sklearn.svm.SVC",
            "svc": "sklearn.svm.SVC",
            "knn": "sklearn.neighbors.KNeighborsClassifier",
            "kneighbors": "sklearn.neighbors.KNeighborsClassifier",
            "randomforest": "sklearn.ensemble.RandomForestClassifier",
            "extratrees": "sklearn.ensemble.ExtraTreesClassifier",
            "gradientboosting": "sklearn.ensemble.GradientBoostingClassifier",
            "gaussianprocess": "sklearn.gaussian_process.GaussianProcessClassifier",
            "mlp": "sklearn.neural_network.MLPClassifier",
        }
        if _OPTIONAL_MODULES.get("lightgbm") is not None:
            mapping["lightgbm"] = "lightgbm.LGBMClassifier"
        if _OPTIONAL_MODULES.get("xgboost") is not None:
            mapping["xgboost"] = "xgboost.XGBClassifier"
        if _OPTIONAL_MODULES.get("catboost") is not None:
            mapping["catboost"] = "catboost.CatBoostClassifier"
        if _OPTIONAL_MODULES.get("tabnet_classifier") is not None:
            mapping["tabnet"] = "tabnet_classifier"
        if _OPTIONAL_MODULES.get("tabpfn_classifier") is not None:
            mapping["tabpfn"] = "tabpfn_classifier"

    key = name_lower.replace("-", "").replace(" ", "")
    if key not in mapping:
        raise KeyError(f"Model '{name}' is not supported for problem type '{problem_type}'")

    # The mapping values are import strings or sentinel names for optional modules.
    qualname = mapping[key]
    # If the value points to one of the optional modules imported earlier, return that directly.
    if qualname in {"tabnet_regressor", "tabnet_classifier", "tabpfn_classifier", "tabpfn_regressor"}:
        cls = _OPTIONAL_MODULES.get(qualname)
        if cls is None:
            raise KeyError(f"Optional model '{name}' is unavailable because the required library is not installed")
        return cls
    # Otherwise import from fully qualified module string.
    module_name, class_name = qualname.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _expand_param_grid(params: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Expand hyperparameter definitions into a grid of parameter dictionaries.

    This helper interprets each value in ``params``: if the value is a list,
    it is treated as a discrete set of options; otherwise it is treated as a
    fixed value. The Cartesian product of all options is produced. If
    ``params`` is empty, yields an empty dict.

    Parameters
    ----------
    params : dict
        Mapping of parameter names to either single values or lists of values.

    Returns
    -------
    iterable of dict
        Each dict corresponds to a unique combination of hyperparameters.
    """
    if not params:
        yield {}
        return
    # Convert single values to singleton lists
    param_grid: Dict[str, List[Any]] = {}
    for k, v in params.items():
        if isinstance(v, list):
            param_grid[k] = v
        else:
            param_grid[k] = [v]
    # Use sklearn's ParameterGrid to expand combinations
    for combo in ParameterGrid(param_grid):
        yield combo


@dataclass
class ModelInstance:
    """Container for a specific model instantiation.

    Attributes
    ----------
    name : str
        The base name of the model (e.g. 'Ridge').
    params : dict
        Hyperparameters used to construct this instance.
    estimator : object
        The estimator object itself.
    """

    name: str
    params: Dict[str, Any]
    estimator: Any


def generate_model_instances(
    model_specs: List[ModelSpec], problem_type: str
) -> List[ModelInstance]:
    """Generate instantiated estimators from model specifications.

    Parameters
    ----------
    model_specs : list of ModelSpec
        Configurations for each model to include in the search.
    problem_type : str
        'regression' or 'classification'.

    Returns
    -------
    list of ModelInstance
        Each entry contains the base name, hyperparameters, and estimator.
    """
    instances: List[ModelInstance] = []
    for spec in model_specs:
        try:
            cls = _get_model_class(spec.name, problem_type)
        except KeyError as exc:
            warnings.warn(str(exc))
            continue
        # Expand grid of hyperparameter combinations
        for param_combo in _expand_param_grid(spec.params or {}):
            # Normalize parameter values: parse string representations of tuples or
            # lists into Python objects where possible. Without this, YAML strings
            # like "(64,)" are treated as literal strings, causing instantiation
            # errors for estimators expecting tuples.
            init_params: Dict[str, Any] = {}
            for key, value in param_combo.items():
                val = value
                if isinstance(val, str):
                    import ast
                    try:
                        val = ast.literal_eval(val)
                    except Exception:
                        # If parsing fails, leave as string
                        pass
                # Convert hidden_layer_sizes lists to tuples when appropriate
                if key.lower() == "hidden_layer_sizes" and isinstance(val, (list, tuple)):
                    try:
                        val = tuple(val)
                    except Exception:
                        pass
                init_params[key] = val
            module_name_lower = cls.__module__.lower()
            # CatBoost: disable verbose logging by default
            if "catboost" in module_name_lower:
                if "verbose" not in init_params:
                    init_params["verbose"] = 0
            # LightGBM: set verbose level to -1 if unspecified (suppress warnings)
            if "lightgbm" in module_name_lower:
                # LGBMRegressor/LGBMClassifier support "verbose" or "verbosity"
                if "verbose" not in init_params and "verbosity" not in init_params:
                    init_params["verbose"] = -1
            # MLP defaults: encourage convergence on small datasets
            if spec.name.lower() == "mlp":
                init_params.setdefault("random_state", 0)
                if problem_type == "regression":
                    init_params.setdefault("max_iter", 2000)
                    init_params.setdefault("early_stopping", True)
                    init_params.setdefault("n_iter_no_change", 20)
                    init_params.setdefault("validation_fraction", 0.1)
            # Special handling for GaussianProcess kernels: convert string names to kernel objects
            model_name_lower = spec.name.lower()
            if model_name_lower in {"gaussianprocess", "gaussianprocessregressor", "gaussianprocessclassifier"}:
                if "kernel" in init_params and isinstance(init_params["kernel"], str):
                    kernel_str = init_params["kernel"]
                    # Import gaussian process kernels dynamically
                    try:
                        from sklearn.gaussian_process import kernels as gpkernels  # type: ignore
                        kernel_cls = getattr(gpkernels, kernel_str)
                        init_params["kernel"] = kernel_cls()
                    except Exception:
                        # Leave as is if conversion fails
                        pass
            if spec.name.lower() == "tabpfn":
                tabpfn_params = prepare_tabpfn_params(problem_type, init_params)
                if tabpfn_params is None:
                    continue
                init_params = tabpfn_params
            try:
                estimator = cls(**init_params)
                model_name_lower = spec.name.lower()
                if model_name_lower in {"tabnet", "mlp"} and problem_type == "regression":
                    estimator = TransformedTargetRegressor(
                        regressor=estimator,
                        transformer=StandardScaler(),
                        check_inverse=False,
                    )
            except TypeError:
                # Some third‑party estimators require specific names or may not
                # accept unknown parameters. Warn and skip those combos.
                warnings.warn(
                    f"Failed to instantiate {spec.name} with params {init_params}; skipping."
                )
                continue
            instances.append(ModelInstance(name=spec.name, params=init_params, estimator=estimator))
    return instances
