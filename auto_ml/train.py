"""Entry point for running the custom AutoML pipeline.

This script coordinates the end‑to‑end workflow: reading the configuration
file, loading the data, generating preprocessing pipelines and model
instances, evaluating all combinations via cross‑validation, writing
results to disk, and producing visualizations. It can be executed as a
stand‑alone Python module or imported and called from other code.

Usage
-----
From the command line:

```
python -m auto_ml.train --config path/to/config.yaml
```

or equivalently from the project root:

```
python train.py --config path/to/config.yaml
```

The configuration YAML specifies all behavior including data paths,
preprocessing options, models to evaluate, cross‑validation settings and
output preferences. See ``config.yaml`` for a template.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if __package__ in {None, ""}:
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
tabpfn_home = PROJECT_ROOT / ".tabpfn_home"
tabpfn_home.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TABPFN_HOME", str(tabpfn_home))
os.environ.setdefault("TABPFN_STATE_DIR", str(tabpfn_home))
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(tabpfn_home / "model_cache"))

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from auto_ml.config import Config
from auto_ml.data_loader import load_dataset, infer_problem_type, split_data, get_feature_types
from auto_ml.preprocessing.preprocessors import generate_preprocessors
from auto_ml.model_factory import ModelInstance, prepare_tabpfn_params
from auto_ml.tabpfn_utils import OfflineTabPFNRegressor
from auto_ml.evaluation import evaluate_model_combinations, _get_scoring, _get_cv_splitter
from auto_ml.visualization import (
    plot_bar_comparison,
    plot_predicted_vs_actual,
    plot_residual_scatter,
    plot_residual_hist,
    plot_metric_heatmap,
)
from auto_ml.search import generate_param_combinations
from auto_ml.ensemble import build_stacking, build_voting
from auto_ml.interpretation import extract_feature_importance, plot_feature_importance, plot_shap_summary


def _tune_lightgbm_params(params: Dict[str, Any], train_size: int, problem_type: str) -> Dict[str, Any]:
    """Ensure LightGBM receives sensible defaults for small datasets."""

    tuned = dict(params)
    tuned.setdefault("force_row_wise", True)
    if problem_type == "regression":
        tuned.setdefault("objective", "regression_l2")
    if "min_child_samples" not in tuned and "min_data_in_leaf" not in tuned:
        if train_size > 0:
            candidate = max(1, train_size // 5)
            tuned["min_child_samples"] = candidate
    return tuned


def _instantiate_estimator(
    model_name: str,
    estimator_cls,
    init_params: Dict[str, Any],
):
    """Instantiate an estimator, handling TabPFN fallbacks when necessary."""

    params_for_init = dict(init_params)
    fallback = params_for_init.pop("use_fallback_tabpfn", False)
    if fallback:
        return OfflineTabPFNRegressor(**params_for_init)
    return estimator_cls(**params_for_init)


def _maybe_wrap_with_target_scaler(
    estimator: Any,
    cfg: Config,
    problem_type: str,
):
    """Optionally wrap estimator with target standardization for regression."""

    if problem_type.lower() != "regression":
        return estimator
    if not getattr(cfg.preprocessing, "target_standardize", False):
        return estimator
    if isinstance(estimator, TransformedTargetRegressor):
        return estimator
    return TransformedTargetRegressor(
        regressor=estimator,
        transformer=StandardScaler(),
        check_inverse=False,
    )


def _build_estimator_with_defaults(
    model_name: str,
    estimator_cls,
    init_params: Dict[str, Any] | None,
    problem_type: str,
    cfg: Config,
    train_size: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Instantiate an estimator while applying evaluation-time defaults."""

    params: Dict[str, Any] = dict(init_params or {})
    name_lower = model_name.lower()
    module_name_lower = estimator_cls.__module__.lower()

    if name_lower in {"gaussianprocess", "gaussianprocessregressor", "gaussianprocessclassifier"}:
        if "kernel" in params and isinstance(params["kernel"], str):
            kernel_str = params["kernel"]
            try:
                from sklearn.gaussian_process import kernels as gpkernels  # type: ignore

                kernel_cls = getattr(gpkernels, kernel_str)
                params["kernel"] = kernel_cls()
            except Exception:
                pass
        if problem_type == "regression":
            if "kernel" not in params:
                from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel  # type: ignore

                params["kernel"] = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                    length_scale=1.0,
                    length_scale_bounds=(1e-2, 1e3),
                ) + WhiteKernel(
                    noise_level=1.0,
                    noise_level_bounds=(1e-5, 1e5),
                )
            params.setdefault("alpha", 1e-2)
            params.setdefault("normalize_y", True)
            params.setdefault("n_restarts_optimizer", 10)

    if "catboost" in module_name_lower:
        params.setdefault("verbose", 0)
        params.setdefault("random_seed", cfg.data.random_seed)
        params.setdefault("allow_writing_files", False)

    if "lightgbm" in module_name_lower:
        params = _tune_lightgbm_params(params, train_size, problem_type)
        if "verbose" not in params and "verbosity" not in params:
            params["verbose"] = -1
        params.setdefault("random_state", cfg.data.random_seed)

    if "xgboost" in module_name_lower:
        params.setdefault("random_state", cfg.data.random_seed)
        params.setdefault("n_jobs", -1)

    if "pytorch_tabnet" in module_name_lower:
        params.setdefault("device_name", "cpu")
        params.setdefault("verbose", 0)

    if name_lower == "mlp":
        params.setdefault("random_state", cfg.data.random_seed)
        if problem_type == "regression":
            params.setdefault("max_iter", 2000)
            params.setdefault("early_stopping", True)
            params.setdefault("n_iter_no_change", 20)
            params.setdefault("validation_fraction", 0.1)

    if name_lower == "tabpfn":
        tabpfn_params = prepare_tabpfn_params(problem_type, params)
        if tabpfn_params is None:
            raise ValueError("TabPFN weights are unavailable")
        params = tabpfn_params

    if "gaussian_process" in module_name_lower and problem_type == "regression":
        if "kernel" not in params:
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel  # type: ignore

            params["kernel"] = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e3),
            ) + WhiteKernel(
                noise_level=1.0,
                noise_level_bounds=(1e-5, 1e5),
            )
        params.setdefault("alpha", 1e-2)
        params.setdefault("normalize_y", True)
        params.setdefault("n_restarts_optimizer", 10)

    estimator = _instantiate_estimator(model_name, estimator_cls, params)

    if name_lower in {"gaussianprocess", "gaussianprocessregressor"} and problem_type == "regression":
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
            check_inverse=False,
        )

    if name_lower in {"tabnet", "mlp"} and problem_type == "regression":
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
            check_inverse=False,
        )

    estimator = _maybe_wrap_with_target_scaler(estimator, cfg, problem_type)

    return estimator, params

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    message="The optimal value found for dimension 0 of parameter length_scale is close to the specified lower bound 1e-05",
)
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    message="Stochastic Optimizer: Maximum iterations",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Could not find the number of physical cores",
)

def run_automl(config_path: Path) -> None:
    """Execute the AutoML pipeline with the given configuration.

    This function orchestrates data loading, preprocessing, hyperparameter
    search, model evaluation, ensemble construction, result visualization,
    and interpretability analysis. It reads the configuration, applies
    various search strategies (grid, random, bayesian), evaluates models
    using cross‑validation, generates plots, selects the best model, and
    optionally computes feature importances and SHAP values.
    """
    cfg = Config.load_from_file(config_path)
    # Load data
    X, y = load_dataset(cfg.data)
    problem_type = infer_problem_type(y, cfg.data.problem_type)
    # Split data into training and optional hold‑out test set
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
        shuffle=cfg.cross_validation.shuffle,
    )
    # Determine feature types and generate preprocessors
    feature_types = get_feature_types(X_train)
    preprocessors = generate_preprocessors(cfg.preprocessing, feature_types)
    # Determine metrics list based on problem type
    metrics = cfg.evaluation.regression_metrics if problem_type == "regression" else cfg.evaluation.classification_metrics
    # Hyperparameter search and model instantiation
    model_instances: List[ModelInstance] = []
    # Local import for model class resolution
    from auto_ml.model_factory import _get_model_class
    # --- Model instantiation -------------------------------------------------
    # Iterate over each model specification and generate parameter combinations
    # using the configured search strategy. For each combination we parse the
    # parameter values (e.g. convert strings like "(64,)" to tuples) and
    # instantiate the corresponding estimator. Any instantiation errors are
    # gracefully skipped to allow the rest of the pipeline to continue.
    import ast
    for spec in cfg.models:
        # Skip models that are explicitly disabled
        if hasattr(spec, "enable") and not spec.enable:
            continue
        combos = generate_param_combinations(
            spec=spec,
            problem_type=problem_type,
            optimization_config=cfg.optimization,
            preprocessors=preprocessors,
            X=X_train,
            y=y_train,
            cv_config=cfg.cross_validation,
            metrics=metrics,
            target_standardize=cfg.preprocessing.target_standardize if problem_type == "regression" else False,
        )
        for params in combos:
            try:
                cls = _get_model_class(spec.name, problem_type)
            except Exception as exc:
                print(f"Warning: {exc}")
                continue
            # Normalize parameter values
            init_params: Dict[str, Any] = {}
            if params:
                for key, value in params.items():
                    val = value
                    # Attempt to parse string literals (e.g. "(64,)")
                    if isinstance(val, str):
                        try:
                            val = ast.literal_eval(val)
                        except Exception:
                            # Leave as string if parsing fails
                            pass
                    # Convert lists for hidden_layer_sizes into tuples
                    if key.lower() == "hidden_layer_sizes":
                        # If parsed value is a list containing a single element, unwrap it
                        if isinstance(val, list):
                            # Handle lists like [(64,)] or [[64]]
                            if len(val) == 1:
                                inner = val[0]
                                # If the inner element is itself a list or tuple of ints, convert directly
                                if isinstance(inner, (list, tuple)):
                                    val = tuple(inner)
                                else:
                                    # Fallback: treat list as the sequence of layer sizes
                                    val = tuple(val)
                            else:
                                # List of ints -> tuple
                                val = tuple(val)
                        elif isinstance(val, tuple):
                            # Already tuple, keep as is
                            val = val
                        init_params[key] = val
                        continue
                    init_params[key] = val
            try:
                estimator, applied_params = _build_estimator_with_defaults(
                    spec.name,
                    cls,
                    init_params,
                    problem_type,
                    cfg,
                    len(y_train),
                )
            except ValueError as exc:
                print(f"Warning: could not instantiate {spec.name}: {exc}")
                continue
            except Exception as exc:
                print(f"Warning: could not instantiate {spec.name} with {init_params}: {exc}")
                continue
            model_instances.append(ModelInstance(name=spec.name, params=applied_params, estimator=estimator))
    if not model_instances:
        raise RuntimeError("No valid models were instantiated. Check your configuration.")
    # Evaluate all base model combinations
    results_df = evaluate_model_combinations(
        preprocessors,
        model_instances,
        X_train,
        y_train,
        cfg.cross_validation,
        problem_type,
        metrics,
    )
    # Evaluate ensembles if configured
    ensemble_records = []
    # Stacking ensembles
    if cfg.ensembles.stacking.enable:
        base_names = cfg.ensembles.stacking.estimators
        final_name = cfg.ensembles.stacking.final_estimator
        for preproc_name, transformer in preprocessors:
            estimators_list: List[Tuple[str, object]] = []
            for bn in base_names:
                try:
                    cls = _get_model_class(bn, problem_type)
                    base_est = cls()
                except Exception:
                    continue
                estimators_list.append((bn, base_est))
            if not estimators_list:
                continue
            # Final estimator
            if final_name:
                try:
                    final_cls = _get_model_class(final_name, problem_type)
                    final_est = final_cls()
                except Exception:
                    final_est = None
            else:
                final_est = None
            if final_est is None:
                try:
                    if problem_type == "regression":
                        from sklearn.linear_model import LinearRegression
                        final_est = LinearRegression()
                    else:
                        from sklearn.linear_model import LogisticRegression
                        final_est = LogisticRegression(max_iter=1000)
                except Exception:
                    continue
            stack_pipeline = build_stacking(
                preprocessor=transformer,
                estimators=estimators_list,
                final_estimator=final_est,
                problem_type=problem_type,
            )
            stack_for_eval = stack_pipeline
            if problem_type == "regression":
                stack_for_eval = _maybe_wrap_with_target_scaler(stack_pipeline, cfg, problem_type)
            try:
                from sklearn.model_selection import cross_validate
                cv = _get_cv_splitter(problem_type, len(y_train), cfg.cross_validation, y_train)
                scoring = _get_scoring(problem_type, metrics)
                cv_res = cross_validate(
                    stack_for_eval,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    error_score="raise",
                )
                record = {
                    "preprocessor": preproc_name,
                    "model": f"Stacking({' + '.join(base_names)})",
                    "params": {},
                }
                for metric_name, scores in cv_res.items():
                    if not metric_name.startswith("test_"):
                        continue
                    simple = metric_name.replace("test_", "")
                    mean_score = np.mean(scores)
                    if simple in {"mae", "mse"}:
                        mean_score = -mean_score
                    record[simple] = mean_score
                if "rmse" in metrics and "mse" in record:
                    record["rmse"] = float(np.sqrt(record["mse"]))
                ensemble_records.append(record)
            except Exception as exc:
                print(f"Warning: stacking ensemble failed for {preproc_name}: {exc}")
    # Voting ensembles
    if cfg.ensembles.voting.enable:
        base_names = cfg.ensembles.voting.estimators
        voting_scheme = cfg.ensembles.voting.voting or ("hard" if problem_type == "classification" else "soft")
        for preproc_name, transformer in preprocessors:
            est_list: List[Tuple[str, object]] = []
            for bn in base_names:
                try:
                    cls = _get_model_class(bn, problem_type)
                    est = cls()
                except Exception:
                    continue
                est_list.append((bn, est))
            if not est_list:
                continue
            vote_pipeline = build_voting(
                preprocessor=transformer,
                estimators=est_list,
                voting=voting_scheme,
                problem_type=problem_type,
            )
            vote_for_eval = vote_pipeline
            if problem_type == "regression":
                vote_for_eval = _maybe_wrap_with_target_scaler(vote_pipeline, cfg, problem_type)
            try:
                from sklearn.model_selection import cross_validate
                cv = _get_cv_splitter(problem_type, len(y_train), cfg.cross_validation, y_train)
                scoring = _get_scoring(problem_type, metrics)
                cv_res = cross_validate(
                    vote_for_eval,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    error_score="raise",
                )
                rec = {
                    "preprocessor": preproc_name,
                    "model": f"Voting({' + '.join(base_names)})",
                    "params": {},
                }
                for metric_name, scores in cv_res.items():
                    if not metric_name.startswith("test_"):
                        continue
                    simple = metric_name.replace("test_", "")
                    mean_score = np.mean(scores)
                    if simple in {"mae", "mse"}:
                        mean_score = -mean_score
                    rec[simple] = mean_score
                if "rmse" in metrics and "mse" in rec:
                    rec["rmse"] = float(np.sqrt(rec["mse"]))
                ensemble_records.append(rec)
            except Exception as exc:
                print(f"Warning: voting ensemble failed for {preproc_name}: {exc}")
    if ensemble_records:
        results_df = pd.concat([results_df, pd.DataFrame(ensemble_records)], ignore_index=True)
    # Prepare output directory
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save only the best result per model. Determine primary metric to rank within each model
    primary_metric_model = cfg.evaluation.primary_metric or ("r2" if problem_type == "regression" else "accuracy")
    if primary_metric_model not in results_df.columns:
        available_cols = [c for c in results_df.columns if c not in {"preprocessor", "model", "params", "error"}]
        primary_metric_model = available_cols[0] if available_cols else None
    # Identify best row for each model (including ensembles) based on primary metric
    best_rows: Dict[str, pd.Series] = {}
    if primary_metric_model:
        for _, row in results_df.iterrows():
            if pd.isna(row.get(primary_metric_model)):
                continue
            model_label = row["model"]
            score = row[primary_metric_model]
            if model_label not in best_rows or score > best_rows[model_label][primary_metric_model]:
                best_rows[model_label] = row
    else:
        for _, row in results_df.iterrows():
            model_label = row["model"]
            best_rows.setdefault(model_label, row)
    # Construct DataFrame of best results
    if best_rows:
        results_df_best = pd.DataFrame(list(best_rows.values()))
    else:
        results_df_best = results_df.copy()
    # Save filtered results (best per model) to CSV and JSON
    results_path = output_dir / cfg.output.results_csv
    results_df_best.to_csv(results_path, index=False)
    results_df_best.to_json(results_path.with_suffix(".json"), orient="records", indent=2)
    # Generate visualizations if enabled
    if cfg.output.generate_plots:
        # Determine metric columns for the filtered DataFrame
        metric_columns = [c for c in results_df_best.columns if c not in {"preprocessor", "model", "params", "error"}]
        # Generate bar charts for each metric using top performers
        for metric in metric_columns:
            plot_path = output_dir / f"top_{metric}.png"
            df_metric = results_df_best.dropna(subset=[metric])
            if df_metric.empty:
                continue
            top_n = len(df_metric) if len(df_metric) <= 20 else 20
            df_sorted = df_metric.sort_values(by=metric, ascending=False).head(top_n)
            try:
                plot_bar_comparison(
                    results_df=df_sorted,
                    metric=metric,
                    top_n=top_n,
                    output_path=plot_path,
                    title=f"Top models by {metric}",
                )
            except Exception as e:
                print(f"Warning: failed to create bar plot for {metric}: {e}")
            try:
                df_sorted.to_csv(plot_path.with_suffix(".csv"), index=False)
            except Exception as e:
                print(f"Warning: failed to write CSV for {metric}: {e}")
        # Comparative heatmap for all models and metrics
        if cfg.visualizations.comparative_heatmap:
            try:
                heatmap_path = output_dir / "model_metric_heatmap.png"
                plot_metric_heatmap(
                    results_df=results_df_best,
                    metrics=metric_columns,
                    output_path=heatmap_path,
                    primary_metric=primary_metric_model,
                    title="Model Comparison Heatmap",
                )
            except Exception as e:
                print(f"Warning: failed to create comparative heatmap: {e}")
    # Choose the overall best model across models using the filtered results
    primary_metric_global = primary_metric_model
    if not primary_metric_global or primary_metric_global not in results_df_best.columns:
        cols_available = [c for c in results_df_best.columns if c not in {"preprocessor", "model", "params", "error"}]
        primary_metric_global = cols_available[0] if cols_available else None
    if primary_metric_global:
        best_row = results_df_best.loc[results_df_best[primary_metric_global].idxmax()]
    else:
        best_row = results_df_best.iloc[0]
    # Retrieve preprocessor and estimator for best model (if ensemble, skip saving)
    best_preproc_name = best_row["preprocessor"]
    best_model_name = best_row["model"]
    best_params = best_row["params"] if isinstance(best_row.get("params"), dict) else {}
    # Find transformer
    transformer = None
    for name, ct in preprocessors:
        if name == best_preproc_name:
            transformer = ct
            break
    if transformer is None:
        transformer = preprocessors[0][1]
    # Construct estimator for the best model
    if best_model_name.startswith("Stacking") or best_model_name.startswith("Voting"):
        # Build ensemble pipeline fresh
        if best_model_name.startswith("Stacking"):
            base_names = cfg.ensembles.stacking.estimators
            final_name = cfg.ensembles.stacking.final_estimator
            ests: List[Tuple[str, object]] = []
            for bn in base_names:
                try:
                    cls_bn = _get_model_class(bn, problem_type)
                    ests.append((bn, cls_bn()))
                except Exception:
                    pass
            # Determine final estimator
            if final_name:
                try:
                    final_cls = _get_model_class(final_name, problem_type)
                    final_est = final_cls()
                except Exception:
                    final_est = None
            else:
                final_est = None
            from sklearn.linear_model import LinearRegression, LogisticRegression
            if final_est is None:
                final_est = LinearRegression() if problem_type == "regression" else LogisticRegression(max_iter=1000)
            estimator = build_stacking(transformer, ests, final_est, problem_type).named_steps["model"]
            if problem_type == "regression":
                estimator = _maybe_wrap_with_target_scaler(estimator, cfg, problem_type)
        else:
            base_names = cfg.ensembles.voting.estimators
            voting_scheme = cfg.ensembles.voting.voting or ("hard" if problem_type == "classification" else "soft")
            ests: List[Tuple[str, object]] = []
            for bn in base_names:
                try:
                    cls_bn = _get_model_class(bn, problem_type)
                    ests.append((bn, cls_bn()))
                except Exception:
                    pass
            estimator = build_voting(transformer, ests, voting_scheme, problem_type).named_steps["model"]
            if problem_type == "regression":
                estimator = _maybe_wrap_with_target_scaler(estimator, cfg, problem_type)
    else:
        # Base model: instantiate from class and params
        try:
            cls_best = _get_model_class(best_model_name, problem_type)
        except Exception:
            estimator = None
        else:
            try:
                estimator, _ = _build_estimator_with_defaults(
                    best_model_name,
                    cls_best,
                    best_params,
                    problem_type,
                    cfg,
                    len(y_train),
                )
            except Exception:
                estimator = None
    # Fit pipeline on full training data
    from sklearn.pipeline import Pipeline as SKPipeline
    full_pipeline = SKPipeline([
        ("preprocessor", transformer),
        ("model", estimator),
    ])
    full_pipeline.fit(X_train, y_train)
    # Save best model
    if cfg.output.save_models:
        model_dir = output_dir / "models"
        model_dir.mkdir(exist_ok=True)
        model_file = model_dir / "best_model.joblib"
        dump(full_pipeline, model_file)
    # Predicted vs actual for regression
    if cfg.output.generate_plots and problem_type == "regression" and cfg.data.test_size and len(X_test) > 0:
        pred_plot = output_dir / "predicted_vs_actual.png"
        try:
            plot_predicted_vs_actual(full_pipeline, X_test, y_test, pred_plot)
        except Exception as e:
            print(f"Warning: could not generate predicted vs actual plot: {e}")
    # Feature importance
    if cfg.interpretation.compute_feature_importance:
        try:
            fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
            fnames = fitted_preprocessor.get_feature_names_out()
        except Exception:
            fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
            transformed = fitted_preprocessor.transform(X_train)
            fnames = [f"f{i}" for i in range(transformed.shape[1])]
        imp_df = extract_feature_importance(full_pipeline.named_steps["model"], list(fnames))
        if imp_df is not None:
            fi_path = output_dir / "feature_importance.png"
            try:
                plot_feature_importance(imp_df, fi_path, title="Feature Importance")
            except Exception as e:
                print(f"Warning: could not plot feature importance: {e}")
    # SHAP analysis
    if cfg.interpretation.compute_shap:
        shap_path = output_dir / "shap_summary.png"
        try:
            plot_shap_summary(full_pipeline, X_train, shap_path)
        except Exception as e:
            print(f"Warning: could not generate SHAP summary: {e}")

    # ---------------------------------------------------------------------
    # Fit and save the best pipeline for each model
    # ---------------------------------------------------------------------
    # Build dictionary of best rows from results_df_best keyed by model name
    best_rows: Dict[str, pd.Series] = {
        row["model"]: row for _, row in results_df_best.iterrows()
    }
    # Create directories for per‑model visualizations
    models_all_dir = output_dir / "models"
    models_all_dir.mkdir(exist_ok=True)
    visual_pred_dir = output_dir / "scatter_plots"
    visual_resid_scatter_dir = output_dir / "residual_scatter"
    visual_resid_hist_dir = output_dir / "residual_hist"
    visual_fi_dir = output_dir / "feature_importances_models"
    visual_shap_dir = output_dir / "shap_summaries_models"
    # Create only if enabled and regression
    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.predicted_vs_actual:
        visual_pred_dir.mkdir(exist_ok=True)
    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.residual_scatter:
        visual_resid_scatter_dir.mkdir(exist_ok=True)
    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.residual_hist:
        visual_resid_hist_dir.mkdir(exist_ok=True)
    if cfg.visualizations.feature_importance:
        visual_fi_dir.mkdir(exist_ok=True)
    if cfg.visualizations.shap_summary:
        visual_shap_dir.mkdir(exist_ok=True)
    # Iterate over best models
    for model_label, row in best_rows.items():
        preproc_name = row["preprocessor"]
        params = row["params"] if isinstance(row.get("params"), dict) else {}
        # Lookup transformer
        transformer = None
        for name, ct in preprocessors:
            if name == preproc_name:
                transformer = ct
                break
        if transformer is None:
            transformer = preprocessors[0][1]
        # Instantiate estimator
        estimator_obj = None
        if model_label.startswith("Stacking") or model_label.startswith("Voting"):
            # Ensemble
            if model_label.startswith("Stacking"):
                base_names = cfg.ensembles.stacking.estimators
                final_name = cfg.ensembles.stacking.final_estimator
                ests: List[Tuple[str, object]] = []
                for bn in base_names:
                    try:
                        cls_bn = _get_model_class(bn, problem_type)
                        ests.append((bn, cls_bn()))
                    except Exception:
                        pass
                # Determine final estimator
                if final_name:
                    try:
                        final_cls = _get_model_class(final_name, problem_type)
                        final_est = final_cls()
                    except Exception:
                        final_est = None
                else:
                    final_est = None
                from sklearn.linear_model import LinearRegression, LogisticRegression
                if final_est is None:
                    final_est = LinearRegression() if problem_type == "regression" else LogisticRegression(max_iter=1000)
                estimator_obj = build_stacking(transformer, ests, final_est, problem_type).named_steps["model"]
            else:
                base_names = cfg.ensembles.voting.estimators
                voting_scheme = cfg.ensembles.voting.voting or ("hard" if problem_type == "classification" else "soft")
                ests: List[Tuple[str, object]] = []
                for bn in base_names:
                    try:
                        cls_bn = _get_model_class(bn, problem_type)
                        ests.append((bn, cls_bn()))
                    except Exception:
                        pass
                estimator_obj = build_voting(transformer, ests, voting_scheme, problem_type).named_steps["model"]
        else:
            # Base model
            try:
                cls = _get_model_class(model_label, problem_type)
            except Exception:
                continue
            try:
                estimator_obj, _ = _build_estimator_with_defaults(
                    model_label,
                    cls,
                    params,
                    problem_type,
                    cfg,
                    len(y_train),
                )
            except Exception:
                continue
        if problem_type == "regression":
            estimator_obj = _maybe_wrap_with_target_scaler(estimator_obj, cfg, problem_type)
        # Build and fit pipeline
        from sklearn.pipeline import Pipeline as SKPipeline
        pipeline_best = SKPipeline([
            ("preprocessor", transformer),
            ("model", estimator_obj),
        ])
        try:
            pipeline_best.fit(X_train, y_train)
        except Exception:
            continue
        need_plot_data = (
            problem_type == "regression"
            and cfg.output.generate_plots
            and (
                cfg.visualizations.predicted_vs_actual
                or cfg.visualizations.residual_scatter
                or cfg.visualizations.residual_hist
            )
        )
        y_train_array = np.asarray(y_train)
        y_pred_for_plots: Optional[np.ndarray] = None
        residuals_for_plots: Optional[np.ndarray] = None
        r2_for_plots: Optional[float] = None
        if need_plot_data:
            try:
                from sklearn.model_selection import cross_val_predict

                cv_inner = _get_cv_splitter(problem_type, len(X_train), cfg.cross_validation, y_train)
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("error", FutureWarning)
                    y_pred_cv = cross_val_predict(
                        pipeline_best,
                        X_train,
                        y_train,
                        cv=cv_inner,
                        n_jobs=None,
                    )
                y_pred_for_plots = np.asarray(y_pred_cv)
                r2_for_plots = float(r2_score(y_train_array, y_pred_for_plots))
                residuals_for_plots = y_train_array - y_pred_for_plots
            except Exception:
                try:
                    y_pred_tmp = np.asarray(pipeline_best.predict(X_train))
                    y_pred_for_plots = y_pred_tmp
                    residuals_for_plots = y_train_array - y_pred_tmp
                    r2_for_plots = float(r2_score(y_train_array, y_pred_tmp))
                except Exception:
                    y_pred_for_plots = None
                    residuals_for_plots = None
                    r2_for_plots = None
        # Save pipeline
        # Use model_label for naming; remove spaces and parentheses
        safe_name = f"{model_label.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}_{preproc_name.replace('|', '_')}"
        model_path = models_all_dir / f"{safe_name}.joblib"
        try:
            dump(pipeline_best, model_path)
        except Exception:
            pass
        # Visualizations
        if problem_type == "regression" and cfg.output.generate_plots:
            # Predicted vs actual
            if cfg.visualizations.predicted_vs_actual:
                scatter_path = visual_pred_dir / f"{safe_name}.png"
                try:
                    plot_predicted_vs_actual(
                        pipeline_best,
                        X_train,
                        y_train,
                        scatter_path,
                        title=f"{model_label} ({preproc_name})",
                        predictions=y_pred_for_plots,
                        r2_override=r2_for_plots,
                    )
                    if y_pred_for_plots is not None and residuals_for_plots is not None:
                        scatter_df = pd.DataFrame(
                            {
                                "actual": y_train_array,
                                "predicted": y_pred_for_plots,
                                "residual": residuals_for_plots,
                            }
                        )
                        scatter_df.to_csv(scatter_path.with_suffix(".csv"), index=False)
                except Exception:
                    pass
            # Residual scatter
            if cfg.visualizations.residual_scatter:
                resid_scatter_path = visual_resid_scatter_dir / f"{safe_name}.png"
                try:
                    plot_residual_scatter(
                        pipeline_best,
                        X_train,
                        y_train,
                        resid_scatter_path,
                        title=f"Residuals: {model_label} ({preproc_name})",
                        predictions=y_pred_for_plots,
                        residuals=residuals_for_plots,
                    )
                    if y_pred_for_plots is not None and residuals_for_plots is not None:
                        resid_scatter_df = pd.DataFrame(
                            {
                                "predicted": y_pred_for_plots,
                                "residual": residuals_for_plots,
                            }
                        )
                        resid_scatter_df.to_csv(resid_scatter_path.with_suffix(".csv"), index=False)
                except Exception:
                    pass
            # Residual histogram
            if cfg.visualizations.residual_hist:
                resid_hist_path = visual_resid_hist_dir / f"{safe_name}.png"
                try:
                    plot_residual_hist(
                        pipeline_best,
                        X_train,
                        y_train,
                        resid_hist_path,
                        title=f"Residual Histogram: {model_label} ({preproc_name})",
                        predictions=y_pred_for_plots,
                        residuals=residuals_for_plots,
                    )
                    if residuals_for_plots is not None:
                        resid_hist_df = pd.DataFrame({"residual": residuals_for_plots})
                        resid_hist_df.to_csv(resid_hist_path.with_suffix(".csv"), index=False)
                except Exception:
                    pass
        # Feature importance per model
        if cfg.visualizations.feature_importance:
            try:
                # Get feature names
                try:
                    fitted_preprocessor = pipeline_best.named_steps["preprocessor"]
                    fnames = fitted_preprocessor.get_feature_names_out()
                except Exception:
                    fitted_preprocessor = pipeline_best.named_steps["preprocessor"]
                    xt = fitted_preprocessor.transform(X_train)
                    fnames = [f"f{i}" for i in range(xt.shape[1])]
                imp_df = extract_feature_importance(pipeline_best.named_steps["model"], list(fnames))
                if imp_df is not None:
                    fi_plot_path = visual_fi_dir / f"{safe_name}.png"
                    plot_feature_importance(imp_df, fi_plot_path, title=f"Feature Importance: {model_label} ({preproc_name})")
            except Exception:
                pass
        # SHAP summary per model
        if cfg.visualizations.shap_summary:
            try:
                shap_plot_path = visual_shap_dir / f"{safe_name}.png"
                plot_shap_summary(pipeline_best, X_train, shap_plot_path)
            except Exception:
                pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the custom AutoML pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()
def main() -> None:
    """CLI entry point for training via ``python -m auto_ml.train``."""

    args = _parse_args()
    run_automl(args.config)


if __name__ == "__main__":
    main()
