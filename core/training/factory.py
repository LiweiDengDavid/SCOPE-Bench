# coding: utf-8

"""
ComponentFactory - Unified Component Factory
==========================================

Eliminates component creation code duplication, unifies management of optimizers,
schedulers, and loss functions.
Follows DRY principle by extracting duplicate component creation code from various trainers.
"""

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from ..config import coerce_runtime_scalar


class Components:
    """Unified Component Factory

    Eliminates code duplication by unifying creation of various components needed during training.
    """

    @staticmethod
    def create_optimizer(model, config, params=None):
        """Unified optimizer creation.

        Args:
            model: Model object
            config: Configuration dictionary

        Returns:
            torch.optim.Optimizer: Created optimizer
        """
        custom = getattr(model, "build_optimizer", None)
        if callable(custom):
            return custom(config)

        optimizer_name = config["optimizer"].lower()
        learning_rate = coerce_runtime_scalar(config["learning_rate"])
        weight_decay = coerce_runtime_scalar(config["weight_decay"])

        if params is None:
            optimizer_params_getter = getattr(model, "get_optimizer_params", None)
            if callable(optimizer_params_getter):
                params = optimizer_params_getter()
            else:
                params = model.parameters()
            params = [param for param in params if param.requires_grad]

        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                params,
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=config['momentum']
            )
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(
                params,
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(
                f"Unsupported optimizer '{optimizer_name}'. "
                "Expected one of: adam, sgd, rmsprop, adamw."
            )

        return optimizer

    @staticmethod
    def create_lr_scheduler(optimizer, config):
        """Unified learning rate scheduler creation.

        Args:
            optimizer: Optimizer object
            config: Configuration dictionary

        Returns:
            torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler, returns None if not needed
        """
        scheduler_config = config['learning_rate_scheduler']

        if scheduler_config is None:
            return None

        if not isinstance(scheduler_config, list):
            raise ValueError(
                "learning_rate_scheduler must be a list like [gamma, step_size] "
                "or [gamma, milestone1, milestone2, ...]."
            )
        if len(scheduler_config) < 2:
            raise ValueError(
                "learning_rate_scheduler must contain at least [gamma, step_size]."
            )

        factor = scheduler_config[0]

        if len(scheduler_config) == 2:
            step_size = int(scheduler_config[1])
            return StepLR(optimizer, step_size=step_size, gamma=factor)

        milestones = [int(m) for m in scheduler_config[1:]]
        return MultiStepLR(optimizer, milestones=milestones, gamma=factor)

    @staticmethod
    def create_loss_function(config):
        """Unified loss function creation.

        Args:
            config: Configuration dictionary

        Returns:
            nn.Module: Loss function
        """
        loss_type = config['loss_type'].lower()

        if loss_type == 'bpr':
            from ..base.loss import BPRLoss
            return BPRLoss()
        elif loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"Unsupported loss_type '{loss_type}'. "
                "Expected one of: bpr, bce, mse, cross_entropy."
            )
