"""Inference script for AutoML models.

This script loads trained model pipelines (preprocessor + estimator) saved by
the AutoML training pipeline and performs predictions on new data. It
supports two input modes:

1. CSV mode: load a CSV file of feature values and predict the target for
   each row.
2. Parameter search mode: define ranges or lists of values for each input
   variable and either enumerate all combinations (grid search) or use
   Optuna to sample candidate combinations via random, TPE or CMA-ES
   samplers. The objective is to maximize or minimize the predicted
   target.

For each model, predictions are saved to a per-model CSV file. An
aggregated CSV collecting predictions from all selected models is also
generated. Additionally, plots are produced to visualize predictions and
model comparisons. Correlation and agreement metrics between models are
computed and visualized to help assess consistency.

Example usage:

    python inference.py \
        --model-dir outputs/train/models \
        --models Ridge,RandomForest \
        --input-csv new_data.csv \
        --output-dir outputs/inference

    python inference.py \
        --model-dir outputs/train/models \
        --input-params params.json \
        --search-method tpe \
        --n-trials 50 \
        --goal max \
        --output-dir outputs/inference

    python inference.py --config inference_config.yaml

The JSON file for parameter search should have the following structure:

    {
        "variables": [
            {"name": "age", "type": "int", "method": "range", "min": 20, "max": 60, "step": 10},
            {"name": "gender", "type": "categorical", "method": "values", "values": ["M", "F"]},
            {"name": "income", "type": "float", "method": "values", "values": [50000.0, 75000.0, 100000.0]}
        ]
    }

Notes
-----
* The CMA-ES sampler in Optuna only supports continuous (float) and integer
  distributions. If categorical variables are present, CMA-ES search will
  raise an error. Use random or TPE search instead.
* For classification models, predictions are taken from the probability of
  the positive class if available. If the classifier does not support
  probabilities, predicted class labels are used and correlation analysis is
  skipped.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from auto_ml.inference import (
    compute_consistency_scores,
    enumerate_parameter_combinations,
    load_models,
    parse_param_spec,
    parse_variables_from_config,
    plot_agreement_heatmap,
    plot_consistency_bars,
    plot_correlation_heatmap,
    plot_predictions,
    run_grid_search,
    run_optimization,
    save_matrices_and_scores,
    save_results,
)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _resolve_config_path(config_arg: Optional[str]) -> Optional[Path]:
    """Return a path to the YAML config if supplied or if a default exists."""

    if config_arg:
        return Path(config_arg)
    default_config = Path("inference_config.yaml")
    return default_config if default_config.is_file() else None


def _load_config(config_arg: Optional[str]) -> Optional[Dict[str, Any]]:
    path = _resolve_config_path(config_arg)
    if path is None:
        return None
    if yaml is None:
        raise ImportError("PyYAML is required to load configuration file. Please install pyyaml.")
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _parse_model_names(models_section: Any) -> Optional[List[str]]:
    if not models_section:
        return None
    if isinstance(models_section, str):
        selected = [m.strip() for m in models_section.split(",") if m.strip()]
        if not selected:
            raise ValueError("At least one model name must be specified in 'models'")
        return selected
    if isinstance(models_section, list):
        parsed: List[str] = []
        for item in models_section:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    parsed.append(name)
                continue
            if isinstance(item, dict):
                name = item.get("name")
                if not name:
                    raise ValueError("Each model entry must include a 'name'")
                enabled_value = item.get("enable", item.get("enabled", True))
                if isinstance(enabled_value, str):
                    enabled = enabled_value.strip().lower() not in {"false", "0", "no", "off"}
                else:
                    enabled = bool(enabled_value)
                if enabled:
                    parsed.append(str(name).strip())
                continue
            raise ValueError("'models' list entries must be strings or dictionaries with 'name'/'enable'")
        if not parsed:
            raise ValueError("At least one model must be enabled in the configuration")
        return parsed
    raise ValueError("'models' in configuration must be a list or comma-separated string")


def _finalize_run(
    models: Dict[str, Any],
    aggregated: pd.DataFrame,
    per_model_results: Dict[str, pd.DataFrame],
    output_dir: str,
    prefix: str,
    goal_for_plot: Optional[str],
) -> None:
    save_results(aggregated, per_model_results, output_dir, prefix)
    print(f"Saved aggregated and per-model results to directory '{output_dir}'.")
    plot_predictions(
        aggregated,
        models.keys(),
        output_dir,
        per_model_results=per_model_results,
        goal=goal_for_plot,
    )
    corr_matrix, agreement_matrix, mean_corr, mean_agreement = compute_consistency_scores(
        aggregated, models.keys()
    )
    save_matrices_and_scores(corr_matrix, agreement_matrix, mean_corr, mean_agreement, output_dir, prefix)
    try:
        plot_correlation_heatmap(aggregated, models.keys(), output_dir)
    except Exception as exc:
        warnings.warn(f"Correlation heatmap could not be generated: {exc}")
    try:
        plot_agreement_heatmap(aggregated, models.keys(), output_dir)
    except Exception as exc:
        warnings.warn(f"Agreement heatmap could not be generated: {exc}")
    try:
        plot_consistency_bars(
            mean_corr,
            'mean_correlation',
            output_dir,
            'Average Pearson Correlation by Model',
            f'{prefix}_mean_correlation.png',
        )
    except Exception as exc:
        warnings.warn(f"Mean correlation bar chart could not be generated: {exc}")
    try:
        plot_consistency_bars(
            mean_agreement,
            'mean_agreement',
            output_dir,
            'Average Agreement Score by Model',
            f'{prefix}_mean_agreement.png',
        )
    except Exception as exc:
        warnings.warn(f"Mean agreement bar chart could not be generated: {exc}")
    with pd.option_context("display.max_columns", None):
        print("Aggregated results (first 5 rows):")
        print(aggregated.head())
        if not corr_matrix.empty:
            print("\nCorrelation matrix:")
            print(corr_matrix)
        if not agreement_matrix.empty:
            print("\nAgreement matrix:")
            print(agreement_matrix)


def _run_from_config(config_data: Dict[str, Any]) -> None:
    model_dir = config_data.get("model_dir")
    if not model_dir:
        raise ValueError("'model_dir' must be specified in the configuration")
    selected_models = _parse_model_names(config_data.get("models"))
    models = load_models(model_dir, selected_names=selected_models)
    if not models:
        print(
            f"No models were loaded from directory '{model_dir}'. Please check that the directory exists and contains "
            "joblib files, and that the model names in the configuration match the saved models."
        )
        return
    print(f"Loaded {len(models)} model(s): {', '.join(models.keys())}")

    output_dir = config_data.get("output_dir", "outputs/inference")
    os.makedirs(output_dir, exist_ok=True)

    input_conf = config_data.get("input")
    if not input_conf or not isinstance(input_conf, dict):
        raise ValueError("'input' section missing or invalid in configuration")
    mode = input_conf.get("mode")
    if mode is None:
        raise ValueError("'mode' must be specified in 'input' section (csv or params)")
    mode_lower = str(mode).lower()

    aggregated: pd.DataFrame
    per_model_results: Dict[str, pd.DataFrame]
    goal_for_plot: Optional[str] = None
    prefix = "results"

    if mode_lower == "csv":
        csv_path = input_conf.get("csv_path")
        if not csv_path:
            raise ValueError("'csv_path' must be provided for CSV input mode")
        df_input = pd.read_csv(csv_path)
        if df_input.empty:
            raise ValueError("Input CSV contains no data")
        print(f"Running predictions on CSV input '{csv_path}' for {len(models)} model(s)...")
        aggregated, per_model_results = run_grid_search(models, df_input.to_dict(orient="records"))
        prefix = "csv"
    elif mode_lower == "params":
        vars_spec = None
        if input_conf.get("variables"):
            vars_spec = parse_variables_from_config(input_conf["variables"])
        elif input_conf.get("params_path"):
            vars_spec = parse_param_spec(input_conf["params_path"])
        else:
            raise ValueError("For 'params' mode, specify either 'variables' list or 'params_path'")
        search_conf = config_data.get("search", {})
        method = str(search_conf.get("method", "grid")).lower()
        n_trials = int(search_conf.get("n_trials", 20))
        goal = str(search_conf.get("goal", "max")).lower()
        goal_for_plot = goal
        if method == "grid":
            combos = enumerate_parameter_combinations(vars_spec)
            print(f"Performing grid search over {len(combos)} parameter combinations for {len(models)} model(s)...")
            aggregated, per_model_results = run_grid_search(models, combos)
            prefix = "grid"
        else:
            print(
                f"Performing Optuna '{method}' optimization with {n_trials} trial(s) for {len(models)} model(s)..."
            )
            aggregated, per_model_results = run_optimization(
                models=models,
                vars_list=vars_spec,
                method=method,
                n_trials=n_trials,
                goal=goal,
            )
            prefix = method
    else:
        raise ValueError("Unknown input mode: choose 'csv' or 'params'")

    _finalize_run(models, aggregated, per_model_results, output_dir, prefix, goal_for_plot)


def _run_from_cli(args: argparse.Namespace) -> None:
    selected_models = None
    if args.models:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not args.model_dir:
        raise SystemExit("--model-dir is required if no configuration file is provided")
    models = load_models(args.model_dir, selected_names=selected_models)
    if not models:
        print(
            f"No models were loaded from directory '{args.model_dir}'. Please check that the directory exists and "
            "contains joblib files matching the selected models."
        )
        return
    print(f"Loaded {len(models)} model(s): {', '.join(models.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.input_csv and args.input_params:
        raise ValueError("Specify only one of --input-csv or --input-params")
    if not args.input_csv and not args.input_params:
        raise ValueError("One of --input-csv or --input-params must be provided")

    aggregated: pd.DataFrame
    per_model_results: Dict[str, pd.DataFrame]
    prefix = "results"
    goal_for_plot: Optional[str] = str(args.goal).lower() if args.goal else None

    if args.input_csv:
        df_input = pd.read_csv(args.input_csv)
        if df_input.empty:
            raise ValueError("Input CSV contains no data")
        print(f"Running predictions on CSV input '{args.input_csv}' for {len(models)} model(s)...")
        aggregated, per_model_results = run_grid_search(models, df_input.to_dict(orient="records"))
        prefix = "csv"
    else:
        vars_list = parse_param_spec(args.input_params)
        if args.search_method == "grid":
            combos = enumerate_parameter_combinations(vars_list)
            print(
                f"Performing grid search over {len(combos)} parameter combinations for {len(models)} model(s)..."
            )
            aggregated, per_model_results = run_grid_search(models, combos)
            prefix = "grid"
        else:
            print(
                f"Performing Optuna '{args.search_method}' optimization with {args.n_trials} trial(s) for "
                f"{len(models)} model(s)..."
            )
            aggregated, per_model_results = run_optimization(
                models=models,
                vars_list=vars_list,
                method=args.search_method,
                n_trials=args.n_trials,
                goal=args.goal,
            )
            prefix = args.search_method

    _finalize_run(models, aggregated, per_model_results, args.output_dir, prefix, goal_for_plot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference script for AutoML models")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML configuration file specifying model directory, models, input, search and output settings. "
            "If omitted, the script looks for 'inference_config.yaml' in the project root. Other command-line options "
            "are ignored when a configuration file is used."
        ),
    )
    parser.add_argument("--model-dir", default=None, help="Directory containing trained joblib models")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model names to use (e.g. 'Ridge,RandomForest'). If omitted, all models are used.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to CSV file containing input feature data for inference.",
    )
    parser.add_argument(
        "--input-params",
        type=str,
        default=None,
        help="Path to JSON file defining parameter ranges/values for search. See documentation for structure.",
    )
    parser.add_argument(
        "--search-method",
        type=str,
        default="grid",
        choices=["grid", "random", "tpe", "cmaes"],
        help="Search method for parameter search mode. 'grid' enumerates all combinations; other methods use Optuna.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials for random/TPE/CMA-ES search. Ignored for grid search.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="max",
        choices=["max", "min"],
        help="Objective direction for optimization: 'max' to maximize prediction, 'min' to minimize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inference",
        help="Directory to save output files.",
    )
    args = parser.parse_args()

    config_data = _load_config(args.config)
    if config_data is not None:
        _run_from_config(config_data)
    else:
        _run_from_cli(args)


if __name__ == "__main__":
    main()
