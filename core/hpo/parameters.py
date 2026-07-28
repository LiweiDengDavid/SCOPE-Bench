# coding: utf-8
"""
Unified Parameter Generation for HPO
===================================

This module provides unified parameter generation logic for different
hyperparameter optimization strategies, eliminating code duplication
between quick_start (grid) and smart_hpo (intelligent) approaches.
"""

from __future__ import annotations

import csv
import random
import numpy as np
from typing import Dict, List, Any, Optional
from itertools import product
import logging
import re
from pathlib import Path


def _coerce_numeric(value: Any) -> Any:
    """Try to convert a string value to int or float."""
    if isinstance(value, str):
        if re.fullmatch(r"[+-]?\d+", value):
            return int(value)
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
            return float(value)
    return value


def is_better(candidate: float, incumbent: float, objective: str) -> bool:
    """Return True if `candidate` beats `incumbent` under the given objective.

    The single source of truth for HPO best-selection direction. Every
    best-tracking site must route through this instead of a hardcoded `>`.
    """
    if objective == "maximize":
        return candidate > incumbent
    if objective == "minimize":
        return candidate < incumbent
    raise ValueError(
        f"optimization.objective must be 'maximize' or 'minimize', got {objective!r}"
    )


def worst_score(objective: str) -> float:
    """Return the worst possible score for the objective (loses to any real score)."""
    if objective == "maximize":
        return float("-inf")
    if objective == "minimize":
        return float("inf")
    raise ValueError(
        f"optimization.objective must be 'maximize' or 'minimize', got {objective!r}"
    )


def select_best_index(scores, objective: str):
    """Return the index of the best score in a pandas Series under the objective.

    Single source of truth for direction-aware best-trial selection (mirrors
    is_better): maximize -> idxmax, minimize -> idxmin. Use this instead of a
    fixed idxmax so minimize objectives keep the correct direction.
    """
    if objective == "maximize":
        return scores.idxmax()
    if objective == "minimize":
        return scores.idxmin()
    raise ValueError(
        f"optimization.objective must be 'maximize' or 'minimize', got {objective!r}"
    )


class ParameterGenerator:
    """Unified parameter generator for different HPO strategies"""
    
    def __init__(self, model_name: str, base_config: Dict[str, Any]):
        """Initialize parameter generator
        
        Args:
            model_name: Name of the model
            base_config: Base configuration dictionary
        """
        self.model_name = model_name
        self.base_config = base_config
        self.logger = logging.getLogger("nexusrec")
    
    def generate_random_combinations(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate random parameter combinations
        
        Args:
            n_samples: Number of random samples to generate
            
        Returns:
            List of parameter dictionaries
        """
        parameter_space = self.base_config["parameter_space"]
        if not parameter_space:
            raise ValueError(
                f"smart_hpo for {self.model_name} requires a non-empty parameter_space"
            )
        hyper_params = self.base_config["hyper_parameters"]
        if not hyper_params:
            raise ValueError(
                f"smart_hpo for {self.model_name} requires a non-empty hyper_parameters list"
            )
        
        combinations = []
        
        for _ in range(n_samples):
            params = {}
            
            for param_name in hyper_params:
                if param_name in parameter_space:
                    param_value = self._sample_parameter(
                        param_name,
                        parameter_space[param_name],
                    )
                else:
                    base_value = self.base_config[param_name]
                    if isinstance(base_value, list):
                        param_value = random.choice(base_value)
                    else:
                        param_value = base_value
                params[param_name] = _coerce_numeric(param_value)
            
            combinations.append(params)
        
        self.logger.info(f"Generated {len(combinations)} random combinations")
        return combinations
    
    def _sample_parameter(self, param_name: str, param_config: Dict[str, Any]) -> Any:
        """Sample a single parameter based on its configuration
        
        Args:
            param_name: Name of the parameter
            param_config: Parameter configuration dictionary
            
        Returns:
            Sampled parameter value
        """
        param_type = param_config["type"]

        if param_type == 'choice':
            values = param_config["values"]
            if not values:
                raise ValueError(f"Parameter '{param_name}' of type 'choice' must have 'values' list")
            coerced_values = [_coerce_numeric(value) for value in values]
            return random.choice(coerced_values)

        if param_type == 'uniform':
            low = _coerce_numeric(param_config["low"])
            high = _coerce_numeric(param_config["high"])
            return random.uniform(low, high)

        if param_type in {'loguniform', 'logscale'}:
            low = _coerce_numeric(param_config["low"])
            high = _coerce_numeric(param_config["high"])
            if low <= 0:
                raise ValueError(f"Parameter '{param_name}' of type '{param_type}' must have low > 0")

            log_low = np.log(low)
            log_high = np.log(high)
            return np.exp(random.uniform(log_low, log_high))

        if param_type == 'int':
            low = int(_coerce_numeric(param_config["low"]))
            high = int(_coerce_numeric(param_config["high"]))
            return random.randint(low, high)

        if param_type == 'logint':
            low = int(_coerce_numeric(param_config["low"]))
            high = int(_coerce_numeric(param_config["high"]))
            if low <= 0:
                raise ValueError(f"Parameter '{param_name}' of type 'logint' must have low > 0")

            log_low = np.log(low)
            log_high = np.log(high)
            return int(np.exp(random.uniform(log_low, log_high)))

        raise ValueError(
            f"Unsupported parameter type '{param_type}' for '{param_name}'. "
            "Expected one of: choice, uniform, loguniform, logscale, int, logint."
        )
    
def generate_grid_parameter_dicts(
    base_config: Dict[str, Any],
    existing_results_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate canonical grid-search combinations from resolved config."""
    hyper_params = list(base_config["hyper_parameters"])
    if not hyper_params:
        return []

    parameter_space = base_config["parameter_space"]
    default_parameters = base_config["default_parameters"]

    ordered_params: List[str] = []
    value_lists: List[List[Any]] = []

    for param_name in hyper_params:
        param_values = _resolve_grid_values(
            param_name,
            parameter_space[param_name] if param_name in parameter_space else None,
            default_parameters[param_name] if param_name in default_parameters else base_config[param_name],
        )
        if not param_values:
            continue
        ordered_params.append(param_name)
        value_lists.append(param_values)

    if not ordered_params:
        return []

    combinations = [
        dict(zip(ordered_params, combination))
        for combination in product(*value_lists)
    ]

    completed = _load_completed_combinations(existing_results_file, ordered_params)
    if not completed:
        return combinations

    completed_keys = {tuple(sorted(row.items())) for row in completed}
    return [
        combination
        for combination in combinations
        if tuple(sorted(combination.items())) not in completed_keys
    ]



def _resolve_grid_values(
    param_name: str,
    param_config: Any,
    default_value: Any,
) -> List[Any]:
    if isinstance(param_config, dict):
        if "grid_values" in param_config and isinstance(param_config["grid_values"], list):
            return list(param_config["grid_values"])
        if "type" in param_config and param_config["type"] == "choice" and isinstance(param_config["values"], list):
            return list(param_config["values"])

    if default_value is None:
        return []

    return [default_value]


def _load_completed_combinations(
    result_file: Optional[str],
    ordered_params: List[str],
) -> List[Dict[str, Any]]:
    if not result_file or not ordered_params:
        return []

    result_path = Path(result_file)
    if not result_path.exists():
        return []

    with result_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if not reader.fieldnames or not all(param in reader.fieldnames for param in ordered_params):
            logging.getLogger("nexusrec").warning(
                "Existing HPO result file %s is missing expected parameter columns",
                result_file,
            )
            return []
        # Coerce CSV string cells to int/float so the completed-combination
        # dedup keys are type-consistent with the YAML-derived (numeric)
        # combinations; otherwise resume never matches and re-runs the grid.
        return [{param: _coerce_numeric(row[param]) for param in ordered_params} for row in reader]
