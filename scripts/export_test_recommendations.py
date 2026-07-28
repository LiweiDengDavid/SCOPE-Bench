#!/usr/bin/env python
# coding: utf-8
"""Run checkpoint-only test evaluation and export ranked item lists."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import ConfigManager
from core.model_registry import get_model, get_trainer
from core.runtime.logger import init_logger
from core.training.environment import prepare_data
from core.utils.result import Result
from core.utils.training import dict2str, init_seed


DEFAULT_MODELS = ["BM3", "BPR", "FlowCF", "GRCN", "LightGCN", "NCF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load best_model.pth, run only the test split, and write "
            "recommendation-list artifacts through output.export."
        )
    )
    parser.add_argument("--dataset", default="ShortVideoFull")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--checkpoint-root",
        default="outputs/checkpoints",
        help="Root containing <MODEL>/<DATASET>/<checkpoint-name>.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Explicit checkpoint path; allowed only with one --models entry.",
    )
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument(
        "--state-key",
        default="model_state_dict",
        help=(
            "Checkpoint state key to load. Use best_model_state when reading "
            "resume_state.pth and you want the validation-best weights."
        ),
    )
    parser.add_argument("--run-type", default="test_export")
    parser.add_argument("--comment", default="checkpoint_best")
    parser.add_argument("--topk", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument(
        "--ncf-eval-batch-size",
        "--neumf-eval-batch-size",
        dest="ncf_eval_batch_size",
        type=int,
        default=8,
        help="Safer override for NCF full-sort evaluation.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--export-dir",
        default="outputs/recommendations/ShortVideoFull/test_export",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["json"],
        choices=["json", "jsonl", "csv", "tsv"],
    )
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Do not store model scores in recommendation items.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional JSONL manifest path. One line is appended per model.",
    )
    return parser.parse_args()


def checkpoint_path(args: argparse.Namespace, model_name: str) -> Path:
    if args.checkpoint_path:
        if len(args.models) != 1:
            raise ValueError("--checkpoint-path can be used only with one model.")
        return Path(args.checkpoint_path)
    return (
        Path(args.checkpoint_root)
        / model_name
        / args.dataset
        / args.checkpoint_name
    )


def build_config(args: argparse.Namespace, model_name: str) -> Dict[str, Any]:
    eval_batch_size = (
        args.ncf_eval_batch_size
        if model_name in {"NCF", "NeuMF"}
        else args.eval_batch_size
    )
    export_path = str(Path(args.export_dir) / model_name)
    config = ConfigManager(
        model_name,
        args.dataset,
        {
            "gpu_id": args.gpu_id,
            "type": args.run_type,
            "comment": args.comment,
            "topk": args.topk,
            "eval_batch_size": eval_batch_size,
            "print_model_info": False,
            "save_model": False,
            "eval_test_during_training": False,
            "output": {
                "export": {
                    "enabled": True,
                    "formats": args.formats,
                    "include_scores": not args.no_scores,
                    "split": "test",
                    "topk": max(args.topk),
                    "path": export_path,
                },
                "save_recommended_topk": False,
            },
        },
    )
    return {key: value for key, value in config.items()}


def load_state(path: Path, state_key: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if state_key not in payload:
        raise ValueError(
            f"Checkpoint {path} does not contain state key {state_key!r}. "
            f"Available keys: {sorted(payload.keys())}"
        )
    state = payload[state_key]
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint state {state_key!r} must be a dict.")
    return state


def run_one(args: argparse.Namespace, model_name: str) -> Dict[str, Any]:
    ckpt_path = checkpoint_path(args, model_name)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    config = build_config(args, model_name)
    init_logger(config)
    logger = logging.getLogger("nexusrec")
    logger.info("Checkpoint-only test export: model=%s dataset=%s", model_name, args.dataset)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Export path: %s", config["export"]["path"])

    init_seed(config["seed"], config["deterministic_algorithms"])
    train_data, _valid_data, test_data = prepare_data(config)
    model = get_model(model_name)(config, train_data).to(config["device"])
    state = load_state(ckpt_path, args.state_key)
    model.load_state_dict(state, strict=True)

    trainer = get_trainer(
        model_name,
        config["is_federated"],
        config["is_sequential"],
    )(config, model)
    metrics = trainer.evaluate(test_data, is_test=True, idx=1, write_export=True)
    logger.info("[Checkpoint Test] %s", dict2str(metrics))

    result_row = {
        "model": config["model"],
        "dataset": config["dataset"],
        "type": config["type"],
        "comment": config["comment"],
        **Result.provenance(config),
        **metrics,
    }
    Result.write(config["result_file_name"], result_row)

    summary = {
        "model": model_name,
        "dataset": args.dataset,
        "checkpoint": str(ckpt_path),
        "state_key": args.state_key,
        "result_file": config["result_file_name"],
        "recommendation_exports": trainer.last_recommendation_export_paths,
        "metrics": metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def append_manifest(path: str, summaries: List[Dict[str, Any]]) -> None:
    if not path:
        return
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as file_obj:
        for summary in summaries:
            file_obj.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    summaries = [run_one(args, model_name) for model_name in args.models]
    append_manifest(args.manifest, summaries)


if __name__ == "__main__":
    main()
