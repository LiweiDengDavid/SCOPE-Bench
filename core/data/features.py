# coding: utf-8

"""
Multimodal Feature Loading
===========================

Module-level functions for loading visual/text features. Used by centralized
models and federated trainers; in both cases the tensors are loaded and placed
on config['device'] immediately (no CPU cache / lazy GPU move).
"""

import os
import numpy as np
import torch
import logging

_logger = logging.getLogger("nexusrec")



def _setup_features(config, model_or_trainer, n_items) -> None:
    """Set v_feat and t_feat attributes on *model_or_trainer* from config.

    For non-multimodal models both attributes are set to ``None``.
    For end-to-end multimodal models feature files are skipped.
    Raises :exc:`AssertionError` if a multimodal model has no usable features.

    Args:
        config: Runtime configuration dictionary.
        model_or_trainer: Model or trainer object to receive feature tensors.
        n_items: Item-catalog size; every loaded feature file must have exactly
            one row per item, otherwise features are row-misaligned.
    """
    if not config["is_multimodal_model"]:
        model_or_trainer.v_feat = None
        model_or_trainer.t_feat = None
        return

    if config["end2end"]:
        model_or_trainer.v_feat = None
        model_or_trainer.t_feat = None
        return

    dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
    features = config["features"]
    v_feat_file_path = os.path.join(dataset_path, features["vision_feature_file"])
    t_feat_file_path = os.path.join(dataset_path, features["text_feature_file"])

    model_or_trainer.v_feat = _load_feature_file(v_feat_file_path, config["device"])
    model_or_trainer.t_feat = _load_feature_file(t_feat_file_path, config["device"])

    # Fail fast on a config/feature dimension mismatch at the shared load point.
    # Models size their projections from config["features"], so tensor shapes and
    # declared dimensions must agree before training starts.
    if model_or_trainer.t_feat is not None and model_or_trainer.t_feat.shape[1] != features["text_dim"]:
        raise ValueError(
            f"text feature dim {model_or_trainer.t_feat.shape[1]} != config features.text_dim "
            f"{features['text_dim']} (file {t_feat_file_path})."
        )
    if model_or_trainer.v_feat is not None and model_or_trainer.v_feat.shape[1] != features["visual_dim"]:
        raise ValueError(
            f"visual feature dim {model_or_trainer.v_feat.shape[1]} != config features.visual_dim "
            f"{features['visual_dim']} (file {v_feat_file_path})."
        )

    # Fail fast on a row-count mismatch with the item catalog: a stale .npy with
    # MORE rows silently row-misaligns every item's features, while fewer rows
    # die late with an opaque device-side IndexError.
    for modality, feat, file_path in (
        ("text", model_or_trainer.t_feat, t_feat_file_path),
        ("visual", model_or_trainer.v_feat, v_feat_file_path),
    ):
        if feat is not None and feat.shape[0] != n_items:
            raise ValueError(
                f"{modality} feature rows {feat.shape[0]} != item catalog size "
                f"{n_items} (file {file_path}); item features would be row-misaligned."
            )

    if model_or_trainer.v_feat is None and model_or_trainer.t_feat is None:
        _logger.warning("All features are None. Check feature files in %s", dataset_path)
        _logger.warning("Expected files: %s, %s", v_feat_file_path, t_feat_file_path)
        _logger.warning(
            "Files exist: %s, %s",
            os.path.exists(v_feat_file_path),
            os.path.exists(t_feat_file_path),
        )
        raise AssertionError(f"All features are None. Check feature files in {dataset_path}")


def _load_feature_file(file_path: str, device=None):
    """Load a feature .npy file, returning a float tensor on *device* (or CPU when device is None).

    Args:
        file_path: Absolute path to the ``.npy`` feature file.
        device: If provided, the returned tensor is moved to this device.

    Returns:
        ``torch.Tensor`` on *device*, or ``None`` if the file does not exist.

    Raises:
        RuntimeError: If the file exists but cannot be loaded (corrupt, wrong
            format, I/O error). Feature loading must fail explicitly.
    """
    if not os.path.isfile(file_path):
        return None

    # Multimodal feature files are dense float arrays, not pickled objects;
    # allow_pickle=False matches the on-disk bundle validators.
    # and avoids deserializing arbitrary objects from a .npy.
    feature_array = np.load(file_path, allow_pickle=False)
    tensor = torch.from_numpy(feature_array).float()
    tensor.requires_grad_(False)
    return tensor.to(device) if device is not None else tensor


def setup_centralized_features(config, model) -> None:
    """Load features for a centralized model (tensors placed on device immediately).

    No-op for federated configs.

    Args:
        config: Runtime configuration dictionary.
        model: Model object.
    """
    if config["is_federated"]:
        return
    # Models set n_items from the dataloader's full-catalog item_num before
    # calling setup_multimodal_features (RecommenderBase.__init__ runs first).
    _setup_features(config, model, model.n_items)


def setup_federated_features(config, trainer) -> None:
    """Load features for a federated trainer.

    Tensors are placed on ``config['device']`` (same as centralized models).
    No-op for non-federated configs.

    Args:
        config: Runtime configuration dictionary.
        trainer: Federated trainer object.
    """
    if not config["is_federated"]:
        return
    # TrainerBase.__init__ sets trainer.model before features are loaded; the
    # model carries the full-catalog item count.
    _setup_features(config, trainer, trainer.model.n_items)
