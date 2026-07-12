# coding: utf-8
 

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
# Set CUDA/threading env vars BEFORE importing torch (pulled in transitively by
# core.config below). CUBLAS_WORKSPACE_CONFIG must be set before the CUDA context
# is created for torch.use_deterministic_algorithms(True) to make cuBLAS matmuls
# reproducible; setting it later (e.g. inside init_seed) is too late for the
# deterministic_algorithms toggle to control GPU kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from core.config import deep_merge_dict
from core.model_registry import _DEFAULT_MODEL, _MODEL_HELP


def load_config(argv=None):
    default_model = _DEFAULT_MODEL
    model_help = _MODEL_HELP
    parser = argparse.ArgumentParser(
        epilog=(
            "Model-specific or experiment-specific overrides must be passed through "
            "--param_overrides as a JSON object."
        )
    )
    parser.add_argument(
        "--model", "-m", type=str, default=default_model, help=model_help
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="MovieLens", help="name of datasets"
    )
    parser.add_argument(
        "--gpu_id", "-g", type=int, default=0, help="set the gpu id"
    )
    parser.add_argument(
        "--type", "-t", type=str, default="test", help="variant of the type"
    )
    parser.add_argument(
        "--comment", "-c", type=str, default="test", help="comment of the experiment"
    )
    # Training Parameters
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Number of training epochs (overrides config file)",
    )
    parser.add_argument(
        "--early_stopping",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        help="Enable/disable early stopping",
    )

    parser.add_argument(
        "--hyper_parameters",
        type=str,
        help="Hyperparameter list in JSON format (e.g., '[]' to disable HPO)",
    )

    # Intelligent Hyperparameter Optimization Options
    parser.add_argument(
        "--smart_hpo",
        action="store_true",
        help="Enable intelligent hyperparameter optimization",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["random", "grid", "bayesian", "tpe"],
        help="HPO strategy: random, grid, bayesian, or tpe",
    )
    parser.add_argument(
        "--hpo_budget",
        type=int,
        help="Maximum number of HPO trials (overrides config file)",
    )

    parser.add_argument(
        "--hpo_parallel",
        action="store_true",
        help="Run HPO as independent single-node GPU shards",
    )

    parser.add_argument(
        "--hpo_gpus",
        type=str,
        help="Comma-separated physical GPU ids for parallel HPO, e.g. 0,1,2; defaults to all visible GPUs",
    )

    parser.add_argument(
        "--hpo_parallel_dry_run",
        action="store_true",
        help="Print parallel HPO shard commands without launching them",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging during HPO training",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Start HPO from scratch instead of resuming from checkpoint",
    )

    parser.add_argument(
        "--resume-training",
        action="store_true",
        default=False,
        help=(
            "Continue an interrupted TRAINING run from its full-state checkpoint "
            "(checkpoint_dir/resume_state.pth). Distinct from --no-resume, which "
            "controls HPO trial-search resumption. Equivalent to "
            "--param_overrides '{\"resume_training\": true}'."
        ),
    )

    parser.add_argument(
        "--param_overrides",
        type=str,
        help='JSON object of parameter overrides, e.g. \'{"save_model": false}\'',
    )

    args = parser.parse_args(argv)

    config_dict = {}

    # Flatten parsed args into config_dict, skipping None values.
    # param_overrides, no_resume, max_epochs, hpo_budget, strategy, and
    # parallel HPO flags are handled explicitly below (top-level training
    # placement, nested optimization placement, or bool inversion).
    explicit_keys = {
        "param_overrides",
        "no_resume",
        "resume_training",
        "max_epochs",
        "hpo_budget",
        "strategy",
        "hpo_parallel",
        "hpo_gpus",
        "hpo_parallel_dry_run",
        "hyper_parameters",
        "verbose",
    }
    for key, value in vars(args).items():
        if value is not None and key not in explicit_keys:
            config_dict[key] = value

    # --resume-training opts into model-training resume (store_true default False;
    # only set when passed so a YAML/param_overrides value is not clobbered).
    if args.resume_training:
        config_dict["resume_training"] = True

    # --hyper_parameters is a JSON list (e.g. '[]' to disable HPO); parse it instead of
    # leaving the raw string in config (which split_configs would iterate char-by-char).
    if args.hyper_parameters is not None:
        hyper_parameters = json.loads(args.hyper_parameters)
        if not isinstance(hyper_parameters, list):
            parser.error("--hyper_parameters must be a JSON list")
        config_dict["hyper_parameters"] = hyper_parameters

    # Training flags are consumed top-level (config["max_epochs"]), so write
    # them there directly. Burying them in a nested "training" group makes them
    # lose to a model YAML's top-level value, because flatten_nested_groups only
    # fills missing top-level keys and would leave the CLI override unapplied.
    if args.max_epochs is not None:
        config_dict["max_epochs"] = args.max_epochs

    # Optimization/HPO flags are consumed nested (config["optimization"][...]),
    # so they stay grouped under "optimization".
    optimization_overrides = {}
    if args.hpo_budget is not None:
        optimization_overrides["budget"] = args.hpo_budget
    if args.strategy is not None:
        optimization_overrides["strategy"] = args.strategy
    if args.hpo_parallel:
        optimization_overrides["parallel"] = True
    if args.hpo_gpus is not None:
        optimization_overrides["parallel_gpus"] = args.hpo_gpus
    if args.hpo_parallel_dry_run:
        optimization_overrides["parallel_dry_run"] = True
    if optimization_overrides:
        config_dict = deep_merge_dict(config_dict, {"optimization": optimization_overrides})

    # Handle parameter overrides
    if args.param_overrides:
        param_overrides = json.loads(args.param_overrides)
        if not isinstance(param_overrides, dict):
            parser.error("--param_overrides must be a JSON object")
        config_dict = deep_merge_dict(config_dict, param_overrides)

    return config_dict, args


if __name__ == "__main__":
    config_dict, args = load_config()
    resume = not args.no_resume

    from core.training import quick_start

    quick_start(
        model=args.model,
        dataset=args.dataset,
        config_dict=config_dict,
        save_model=None,
        resume=resume,
        verbose=args.verbose,
    )
