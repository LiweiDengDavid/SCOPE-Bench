# coding: utf-8
"""
Training Core - Pure Training Logic
==================================

This module contains the core training logic that is paradigm-agnostic.
It provides a single point of truth for all training scenarios.

Functions:
- train_single(): Core training function for any configuration
"""

from typing import Dict, Any, Tuple
import logging
import torch.nn as nn

from ..model_registry import get_model, get_trainer
from ..utils.training import init_seed


def print_model_architecture(model: nn.Module, config: Dict[str, Any]) -> None:
    """Print model architecture and parameter statistics

    Args:
        model: The PyTorch model
        config: Configuration dictionary
    """
    logger = logging.getLogger("nexusrec")
    logger.info(f"{'-'*30} MODEL ARCHITECTURE {'-'*30}")
    # Model name, dataset, and paradigm on one line
    model_name = config["model"]
    dataset_name = config["dataset"]

    # Model paradigm and characteristics
    is_federated = config["is_federated"]
    is_multimodal = config["is_multimodal_model"]
    is_sequential = config["is_sequential"]

    paradigm_info = []
    if is_federated:
        paradigm_info.append("Federated")
    if is_multimodal:
        paradigm_info.append("Multimodal")
    if is_sequential:
        paradigm_info.append("Sequential")
    if not paradigm_info:
        paradigm_info.append("Standard")

    logger.info(
        f"Model: {model_name} | Dataset: {dataset_name} | Paradigm: {' + '.join(paradigm_info)}"
    )
    logger.info("")
    logger.info("Network Structure:")

    def print_module_tree(module, name="", indent=0):
        if indent == 0:
            logger.info(f"  {module.__class__.__name__}")
        else:
            prefix = "  " * indent + "├─ "
            msg = f"{prefix}{name}: {module.__class__.__name__}"
            if hasattr(module, "weight") and module.weight is not None:
                msg += f" {list(module.weight.shape)}"
            elif hasattr(module, "in_features") and hasattr(module, "out_features"):
                msg += f" [{module.in_features} → {module.out_features}]"
            elif hasattr(module, "embedding_dim") and hasattr(module, "num_embeddings"):
                msg += f" [{module.num_embeddings} × {module.embedding_dim}]"
            logger.info(msg)
        # Only show direct children for clarity
        for child_name, child_module in module.named_children():
            # Skip empty sequential containers
            if isinstance(child_module, nn.Sequential) and len(child_module) == 0:
                continue
            print_module_tree(child_module, child_name, indent + 1)

    print_module_tree(model)
    logger.info("")
    # Calculate parameter statistics
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    # Parameter breakdown by module type
    param_by_module = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            non_trainable_params += num_params
        # Get module type for breakdown
        module_type = name.split(".")[0] if "." in name else "main"
        if module_type not in param_by_module:
            param_by_module[module_type] = 0
        param_by_module[module_type] += num_params
    logger.info("Parameter Statistics:")
    # Ultra-compact parameter statistics - all on one line
    model_size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
    # Parameters and model size on one line
    logger.info(
        f"\tTotal: {total_params:,} | Trainable: {trainable_params:,} | Non-trainable: {non_trainable_params:,} | Size: {model_size_mb:.2f}MB"
    )
    # Module breakdown ratios on one line (if applicable)
    if len(param_by_module) > 1:
        sorted_modules = sorted(
            param_by_module.items(), key=lambda x: x[1], reverse=True
        )
        ratio_items = []
        for module, count in sorted_modules:
            percentage = (count / total_params) * 100
            ratio_items.append(f"{module}: {percentage:.1f}%")
        logger.info(f"\tBreakdown: {' | '.join(ratio_items)}")
    logger.info("-" * 80)


def train_single(
    config: Dict[str, Any],
    train_data: Any,
    valid_data: Any,
    test_data: Any,
    return_trainer: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    """Core training function - pure training logic without HPO specifics

    Args:
        config: Complete configuration dictionary
        train_data: Training data loader
        valid_data: Validation data loader
        test_data: Test data loader
        return_trainer: Whether to return trainer instance for model saving

    Returns:
        Tuple of (test_result_dict, valid_result_dict, trainer_instance)
        If return_trainer is False, trainer_instance will be None.
    """
    # Reset random seed for reproducibility
    init_seed(config["seed"], config["deterministic_algorithms"])
    # Setup data loader random state
    train_data.pretrain_setup()
    # Load and initialize model
    model = get_model(config["model"])(config, train_data).to(config["device"])
    # Validate the multi-negative contract: the dataloader emits K negatives per
    # positive, so models must opt in before training on all K rows.
    if config["num_negatives"] > 1 and not getattr(model, "supports_multi_negatives", False):
        raise ValueError(
            f"num_negatives={config['num_negatives']} but model '{config['model']}' "
            "consumes only one negative per positive (supports_multi_negatives=False). "
            "Set num_negatives=1, or use a model that declares "
            "supports_multi_negatives=True."
        )
    # Print model architecture and parameters
    if config["print_model_info"]:
        print_model_architecture(model, config)
    # Load and initialize trainer
    is_sequential = config["is_sequential"]
    trainer = get_trainer(config["model"], config["is_federated"], is_sequential)(
        config, model
    )
    # Training resume (opt-in): restore model/optimizer/RNG/best-tracking from a
    # full-state checkpoint and continue from the next epoch/round. Runs AFTER
    # model+trainer construction (so the multi-negative guard above still fires)
    # and AFTER init_seed — restore_rng_state then overrides the fresh seed so the
    # resumed run continues bit-identically rather than replaying epoch 0.
    if config["resume_training"]:
        trainer.load_training_state(train_data)
    # Execute training — single code path for all paradigms
    best_valid_score, best_valid_result, best_test_result = trainer.fit(
        train_data,
        valid_data=valid_data,
        test_data=test_data,
    )

    # Final test evaluation with the validation-best model state
    eval_test_during_training = config["eval_test_during_training"]
    eval_final_test = config["eval_final_test"]
    if eval_final_test and not eval_test_during_training and test_data is not None:
        final_test_result = trainer.evaluate_final_test(test_data)
        if final_test_result:
            best_test_result = final_test_result
            trainer.update_final_test_result(final_test_result)

    # Return trainer instance if requested for model saving
    if return_trainer:
        return best_test_result, best_valid_result, trainer
    else:
        # Omit the trainer object unless the caller explicitly requests it.
        return best_test_result, best_valid_result, None
