"""Configuration loader for the AutoML library.

This module defines a set of data classes that mirror the structure of the
configuration YAML file. Users can modify the YAML file without touching
Python code. The loader parses the YAML into typed objects for easier
consumption elsewhere in the codebase.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)


@dataclass
class DataConfig:
    csv_path: str
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    problem_type: Optional[str] = None
    test_size: float = 0.0
    random_seed: int = 42


@dataclass
class PreprocessingConfig:
    numeric_imputation: List[Optional[str]] = field(default_factory=lambda: ['mean'])
    categorical_imputation: List[Optional[str]] = field(default_factory=lambda: ['most_frequent'])
    scaling: List[Optional[str]] = field(default_factory=lambda: ['standard'])
    categorical_encoding: List[Optional[str]] = field(default_factory=lambda: ['onehot'])
    polynomial_degree: Union[bool, int] = False
    target_standardize: bool = False


@dataclass
class ModelSpec:
    """Specification for a single model.

    Attributes
    ----------
    name : str
        The human‑readable name of the model (e.g. 'Ridge').
    enable : bool
        Whether this model should be included in the AutoML search. Allows
        disabling a model without removing its entry from the YAML.
    params : dict
        Hyperparameters and their candidate values. Values can be lists
        to denote grids.
    """

    name: str
    enable: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    enable: bool = False
    estimators: List[str] = field(default_factory=list)
    final_estimator: Optional[str] = None
    voting: Optional[str] = None


@dataclass
class EnsembleGroup:
    stacking: EnsembleConfig = field(default_factory=EnsembleConfig)
    voting: EnsembleConfig = field(default_factory=EnsembleConfig)


@dataclass
class CVConfig:
    n_folds: Optional[int] = None
    shuffle: bool = True
    random_seed: int = 42


@dataclass
class OutputConfig:
    output_dir: str = "auto_ml_output"
    save_models: bool = True
    generate_plots: bool = True
    results_csv: str = "results_summary.csv"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics.

    Attributes
    ----------
    regression_metrics : list of str
        Metrics to compute for regression tasks. Supported: 'mae', 'mse',
        'rmse', 'r2'.
    classification_metrics : list of str
        Metrics for classification tasks. Supported: 'accuracy',
        'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr'.
    primary_metric : Optional[str]
        If provided, determines which metric to use when selecting the best
        model. Otherwise defaults to r2 (regression) or accuracy (classification).
    """

    regression_metrics: List[str] = field(default_factory=lambda: ["mae", "rmse", "r2"])
    classification_metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1_macro", "roc_auc_ovr"])
    primary_metric: Optional[str] = None


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter search methods.

    Attributes
    ----------
    method : str
        'grid', 'random' or 'bayesian'.
    n_iter : int
        Number of iterations for random or bayesian search. Ignored for grid.
    """

    method: str = "grid"
    n_iter: int = 10


@dataclass
class InterpretationConfig:
    """Configuration for feature importance and SHAP analysis.

    Attributes
    ----------
    compute_feature_importance : bool
        Whether to compute and plot feature importances for models that
        support it (e.g., tree‑based models).
    compute_shap : bool
        Whether to compute SHAP values and produce summary plots for
        supported models (requires the 'shap' library).
    """

    compute_feature_importance: bool = False
    compute_shap: bool = False


@dataclass
class VisualizationConfig:
    """Configuration for per‑model visualisations.

    Attributes
    ----------
    predicted_vs_actual : bool
        If True, generate scatter plots of actual vs. predicted values for each
        model.
    residual_scatter : bool
        If True, generate residuals vs. predicted scatter plots for each model.
    residual_hist : bool
        If True, generate residual (error) histograms for each model.
    feature_importance : bool
        If True, compute and plot feature importances for models that support it.
    shap_summary : bool
        If True, compute SHAP values (if shap library is installed) and output
        summary plots for each model.
    """

    predicted_vs_actual: bool = True
    residual_scatter: bool = False
    residual_hist: bool = False
    feature_importance: bool = False
    shap_summary: bool = False

    # Whether to generate a comparative heatmap of models vs metrics when
    # comparing many models. The heatmap shows how each model performs on all
    # metrics. If False, only bar charts are generated.
    comparative_heatmap: bool = False


@dataclass
class Config:
    data: DataConfig
    preprocessing: PreprocessingConfig
    models: List[ModelSpec]
    ensembles: EnsembleGroup
    cross_validation: CVConfig
    output: OutputConfig
    evaluation: EvaluationConfig
    optimization: OptimizationConfig
    interpretation: InterpretationConfig
    visualizations: VisualizationConfig

    @staticmethod
    def load_from_file(path: Union[str, Path]) -> "Config":
        """Load a configuration YAML file and return a Config instance.

        Parameters
        ----------
        path: str or Path
            Path to the YAML configuration file.

        Returns
        -------
        Config
            A populated configuration object.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file '{path}' does not exist")

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Parse data section
        data_cfg = raw.get("data", {})
        data = DataConfig(
            csv_path=data_cfg.get("csv_path"),
            target_column=data_cfg.get("target_column"),
            feature_columns=data_cfg.get("feature_columns"),
            problem_type=data_cfg.get("problem_type"),
            test_size=float(data_cfg.get("test_size", 0.0)),
            random_seed=int(data_cfg.get("random_seed", 42)),
        )

        # Preprocessing
        pre_cfg = raw.get("preprocessing", {})
        preprocessing = PreprocessingConfig(
            numeric_imputation=pre_cfg.get("numeric_imputation", ['mean']),
            categorical_imputation=pre_cfg.get("categorical_imputation", ['most_frequent']),
            scaling=pre_cfg.get("scaling", ['standard']),
            categorical_encoding=pre_cfg.get("categorical_encoding", ['onehot']),
            polynomial_degree=pre_cfg.get("polynomial_degree", False),
            target_standardize=_coerce_bool(pre_cfg.get("target_standardize", False)),
        )

        # Models
        models_cfg = raw.get("models", [])
        models: List[ModelSpec] = []
        for item in models_cfg:
            name = item.get("name")
            if not name:
                continue
            enable = bool(item.get("enable", True))
            params = item.get("params", {}) or {}
            # Normalize hidden_layer_sizes to avoid YAML splitting parentheses
            if "hidden_layer_sizes" in params:
                normalized_hls = _normalize_hidden_layer_sizes(params["hidden_layer_sizes"])
                params["hidden_layer_sizes"] = normalized_hls
            models.append(ModelSpec(name=name, enable=enable, params=params))

        # Ensembles
        ensembles_cfg = raw.get("ensembles", {})
        stacking_cfg = ensembles_cfg.get("stacking", {})
        voting_cfg = ensembles_cfg.get("voting", {})
        ensembles = EnsembleGroup(
            stacking=EnsembleConfig(
                enable=bool(stacking_cfg.get("enable", False)),
                estimators=stacking_cfg.get("estimators", []),
                final_estimator=stacking_cfg.get("final_estimator"),
            ),
            voting=EnsembleConfig(
                enable=bool(voting_cfg.get("enable", False)),
                estimators=voting_cfg.get("estimators", []),
                voting=voting_cfg.get("voting"),
            ),
        )

        # Cross‑validation
        cv_cfg = raw.get("cross_validation", {})
        cross_validation = CVConfig(
            n_folds=cv_cfg.get("n_folds"),
            shuffle=bool(cv_cfg.get("shuffle", True)),
            random_seed=int(cv_cfg.get("random_seed", 42)),
        )

        # Output settings
        out_cfg = raw.get("output", {})
        output = OutputConfig(
            output_dir=out_cfg.get("output_dir", "auto_ml_output"),
            save_models=bool(out_cfg.get("save_models", True)),
            generate_plots=bool(out_cfg.get("generate_plots", True)),
            results_csv=out_cfg.get("results_csv", "results_summary.csv"),
        )

        # Evaluation settings
        eval_cfg = raw.get("evaluation", {})
        evaluation = EvaluationConfig(
            regression_metrics=eval_cfg.get("regression_metrics", ["mae", "rmse", "r2"]),
            classification_metrics=eval_cfg.get("classification_metrics", ["accuracy", "f1_macro", "roc_auc_ovr"]),
            primary_metric=eval_cfg.get("primary_metric"),
        )

        # Optimization settings
        opt_cfg = raw.get("optimization", {})
        optimization = OptimizationConfig(
            method=opt_cfg.get("method", "grid"),
            n_iter=int(opt_cfg.get("n_iter", 10)),
        )

        # Interpretation settings
        interp_cfg = raw.get("interpretation", {})
        interpretation = InterpretationConfig(
            compute_feature_importance=bool(interp_cfg.get("compute_feature_importance", False)),
            compute_shap=bool(interp_cfg.get("compute_shap", False)),
        )

        # Visualization settings
        visual_cfg = raw.get("visualizations", {})
        visualizations = VisualizationConfig(
            predicted_vs_actual=bool(visual_cfg.get("predicted_vs_actual", True)),
            residual_scatter=bool(visual_cfg.get("residual_scatter", False)),
            residual_hist=bool(visual_cfg.get("residual_hist", False)),
            feature_importance=bool(visual_cfg.get("feature_importance", False)),
            shap_summary=bool(visual_cfg.get("shap_summary", False)),
            comparative_heatmap=bool(visual_cfg.get("comparative_heatmap", False)),
        )

        return Config(
            data=data,
            preprocessing=preprocessing,
            models=models,
            ensembles=ensembles,
            cross_validation=cross_validation,
            output=output,
            evaluation=evaluation,
            optimization=optimization,
            interpretation=interpretation,
            visualizations=visualizations,
        )

    def __repr__(self) -> str:
        return (
            f"Config(data={self.data}, preprocessing={self.preprocessing}, "
            f"models={self.models}, ensembles={self.ensembles}, "
            f"cross_validation={self.cross_validation}, output={self.output})"
        )


def _coerce_hidden_layer_tuple(value: Any) -> Tuple[int, ...]:
    """Convert a raw hidden_layer_sizes entry into a tuple of ints.

    Accepts integers, floats representing whole numbers, iterables of
    numbers, or string representations such as "(64,)" or "128 64".
    Raises ValueError if the value cannot be interpreted.
    """

    if isinstance(value, tuple):
        items = value
    elif isinstance(value, list):
        items = tuple(value)
    elif isinstance(value, int) and not isinstance(value, bool):
        return (int(value),)
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"hidden_layer_sizes entries must be integers, got {value!r}")
        return (int(value),)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("hidden_layer_sizes string entries cannot be empty")
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            digits = re.findall(r"-?\d+", stripped)
            if not digits:
                raise ValueError(f"Could not parse hidden_layer_sizes value '{value}'")
            return _coerce_hidden_layer_tuple(tuple(int(d) for d in digits))
        return _coerce_hidden_layer_tuple(parsed)
    else:
        raise ValueError(f"Unsupported hidden_layer_sizes value type: {type(value)}")

    if not items:
        raise ValueError("hidden_layer_sizes tuples cannot be empty")

    coerced: List[int] = []
    for item in items:
        if isinstance(item, float):
            if not item.is_integer():
                raise ValueError(f"hidden_layer_sizes entries must be integers, got {item!r}")
            coerced.append(int(item))
        elif isinstance(item, int) and not isinstance(item, bool):
            coerced.append(int(item))
        elif isinstance(item, str):
            if not item.strip():
                raise ValueError("hidden_layer_sizes string entries cannot be empty")
            try:
                coerced.append(int(item))
            except ValueError as exc:
                raise ValueError(f"Could not parse hidden_layer_sizes value '{item}'") from exc
        else:
            raise ValueError(f"Unsupported hidden_layer_sizes element type: {type(item)}")
    return tuple(coerced)


def _merge_tokenized_hidden_layer_values(tokens: List[str]) -> List[Any]:
    """Reconstruct tuple strings from YAML-tokenized hidden_layer_sizes entries."""

    pattern = re.compile(r"\([^()]*\)|-?\d+")
    merged: List[Any] = []
    text = " ".join(tokens)
    for segment in pattern.findall(text):
        if segment.startswith("(") and ")" in segment:
            digits = re.findall(r"-?\d+", segment)
            if not digits:
                continue
            if len(digits) == 1:
                merged.append(f"({digits[0]},)")
            else:
                merged.append("(" + ", ".join(digits) + ")")
        else:
            merged.append(segment)
    return merged


def _normalize_hidden_layer_sizes(raw: Any) -> List[Tuple[int, ...]]:
    """Normalize hidden_layer_sizes config values into tuples of ints."""

    if raw is None:
        return []

    candidates: List[Tuple[int, ...]] = []

    if isinstance(raw, list):
        values: List[Any]
        if raw and all(isinstance(elem, str) for elem in raw) and any(
            "(" in elem or ")" in elem for elem in raw
        ):
            merged = _merge_tokenized_hidden_layer_values(raw)
            if merged:
                values = merged
            else:
                values = list(raw)
        else:
            values = list(raw)
    else:
        values = [raw]

    for value in values:
        try:
            tuple_value = _coerce_hidden_layer_tuple(value)
        except ValueError:
            continue
        if tuple_value not in candidates:
            candidates.append(tuple_value)

    if not candidates:
        raise ValueError("No valid hidden_layer_sizes values could be parsed from configuration")

    return candidates
