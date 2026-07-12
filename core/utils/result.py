# coding: utf-8
"""Single-row experiment result CSV helpers."""

from __future__ import annotations

import datetime
import math
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import pandas as pd
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    """Return the current git commit, or 'unknown' when git is unavailable."""
    if shutil.which("git") is None:
        return "unknown"
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


class Result:
    """Read and write single-row result CSV artifacts."""

    @staticmethod
    def provenance(config: Dict[str, Any]) -> Dict[str, Any]:
        """Reproducibility fields shared by result CSVs and recommendation metadata."""
        import numpy

        row = {
            "__seed": config["seed"],
            "__git_commit": _git_commit(),
            "__python_version": platform.python_version(),
            "__torch_version": torch.__version__,
            "__numpy_version": numpy.__version__,
            "__timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if "hpo_lineage" in config and config["hpo_lineage"]:
            lineage = config["hpo_lineage"]
            if not isinstance(lineage, dict):
                raise ValueError("hpo_lineage must be a dict when present.")
            for key, value in lineage.items():
                row[f"__hpo_{key}"] = value
        return row

    @staticmethod
    def write(path: str | Path, row: Dict[str, Any]) -> None:
        """Persist one canonical result row."""
        csv_path = Path(path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(csv_path, index=False)

    @staticmethod
    def load(
        path: str | Path,
        required_columns: Sequence[str] | None = None,
        name: str = "Result CSV",
    ) -> Dict[str, Any]:
        """Load a CSV artifact that must contain exactly one row."""
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"{name} is empty: {path}")
        if len(frame) != 1:
            raise ValueError(f"{name} must contain exactly one row: {path}")

        row = frame.iloc[0].to_dict()
        if required_columns is not None:
            missing = [column for column in required_columns if column not in row]
            if missing:
                raise ValueError(
                    f"{name} is missing required columns {missing}: {path}"
                )
        return row

    @staticmethod
    def metrics(row: Dict[str, Any], exclude: Iterable[str]) -> Dict[str, float]:
        """Extract metric columns from a result row."""
        excluded = set(exclude)
        metrics: Dict[str, float] = {}
        for column, value in row.items():
            if column in excluded or str(column).startswith("__"):
                continue
            metric = float(value)
            if not math.isfinite(metric):
                raise ValueError(f"Result metric '{column}' is NaN/Inf.")
            metrics[str(column)] = metric
        return metrics

    @staticmethod
    def pair_key(row: Dict[str, Any], field: str, label: str) -> str:
        """Resolve a strict pairing key from a result row."""
        row_key = "__stem__" if field == "stem" else field
        if row_key not in row:
            raise ValueError(f"Pair field '{field}' not found in {label}.")
        value = row[row_key]
        if pd.isna(value):
            raise ValueError(f"Pair field '{field}' is empty in {label}.")
        return str(value)
