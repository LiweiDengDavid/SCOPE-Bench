# coding: utf-8
"""Training utilities — seed, batch, epoch, and early-stopping helpers."""

from __future__ import annotations

import datetime
import math
import os
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..runtime.logger import TrainLogger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def get_local_time():
    """Get current time as a formatted string."""
    cur = datetime.datetime.now()
    return cur.strftime("%b-%d-%Y-%H-%M-%S")


def artifact_token(value):
    """Return a filesystem-friendly token for evaluation/training artifacts."""
    return "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in str(value)
    )


def init_seed(seed, deterministic_algorithms=False):
    """Initialize random seeds for reproducible runs.

    Seeds Python ``random``, NumPy, and Torch (CPU + CUDA). CPU determinism
    relies solely on these seeds; the cudnn flags below are CUDA-only.

    When ``deterministic_algorithms`` is true, also force Torch to use
    deterministic algorithms (and set CUBLAS_WORKSPACE_CONFIG so cuBLAS matmuls
    are reproducible). This trades throughput for exact GPU reproducibility and
    will RAISE if an op lacks a deterministic implementation — surfacing
    nondeterminism rather than hiding it. The toggle lives in
    configs/overall.yaml (``deterministic_algorithms``).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def capture_rng_state() -> Dict[str, Any]:
    """Snapshot every RNG stream a run consumes, for bit-faithful resume.

    Covers Python ``random``, NumPy, and Torch (CPU + all CUDA devices) — the
    exact four streams ``init_seed`` seeds. Restoring these continues the run from
    where it left off rather than re-drawing the seed sequence from epoch 0. CUDA
    state is captured only when CUDA is available, so a CPU checkpoint stays
    CPU-loadable. Note: explicit ``torch.Generator`` objects (e.g. the sequential
    DataLoader's) are NOT global and must be captured separately by their owner.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG streams captured by :func:`capture_rng_state`.

    Fails fast (KeyError) on a malformed snapshot rather than falling back.
    CUDA state is restored only if it was captured AND CUDA is available now.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ---------------------------------------------------------------------------
# Batch & Epoch
# ---------------------------------------------------------------------------

def prepare_batch(batch: Any, device: torch.device) -> Any:
    """Move a supported batch container onto the target device."""
    if isinstance(batch, dict):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    if isinstance(batch, (tuple, list)):
        return [item.to(device) if torch.is_tensor(item) else item for item in batch]
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    clip_grad_norm: Optional[float] = None,
) -> torch.Tensor:
    """Execute a single optimization step."""
    optimizer.zero_grad(set_to_none=True)
    loss = model.calculate_loss(batch)

    if isinstance(loss, tuple):
        loss = sum(loss)

    if torch.isnan(loss) or torch.isinf(loss):
        import logging
        logging.getLogger("nexusrec").warning(
            "NaN/Inf loss detected, skipping backward pass"
        )
        return loss

    cfg = getattr(model, "config", None)
    detect_anomaly = bool(cfg["detect_anomaly"]) if cfg is not None else False

    if detect_anomaly:
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
    else:
        loss.backward()

    if clip_grad_norm:
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

    optimizer.step()
    return loss


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    device: torch.device,
    clip_grad_norm: Optional[float] = None,
    lr_scheduler: Optional[Any] = None,
    epoch_idx: int = 0,
    *,
    total_epochs: int,
    nan_abort_threshold: int,
) -> Dict[str, Any]:
    """Run a single centralized training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    nan_count = 0
    start_time = time.time()

    for batch in train_data:
        batch = prepare_batch(batch, device)
        loss = train_step(model, optimizer, batch, clip_grad_norm)
        loss_val = loss.item()

        if math.isnan(loss_val) or math.isinf(loss_val):
            nan_count += 1
            if nan_count >= nan_abort_threshold:
                raise ValueError(
                    f"Training diverged: {nan_count} consecutive NaN/Inf losses"
                )
            continue

        nan_count = 0
        total_loss += loss_val
        num_batches += 1

    if lr_scheduler:
        lr_scheduler.step()

    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(num_batches, 1)

    TrainLogger.log_epoch_progress(
        epoch=epoch_idx + 1,
        total_epochs=total_epochs,
        stage="Train",
        loss=avg_loss,
        epoch_time=epoch_time,
        num_batches=num_batches,
    )

    return {
        "loss": avg_loss,
        "num_batches": num_batches,
        "epoch_time": epoch_time,
        "total_loss": total_loss,
    }


# ---------------------------------------------------------------------------
# Training control
# ---------------------------------------------------------------------------

def early_stopping(value, best, cur_step, bigger=True):
    """Update the best score and the consecutive-no-improvement counter.

    The trainer decides when to stop via its own patience check, so this
    helper only maintains (best, cur_step, update_flag).
    """
    update_flag = False

    is_better = (value > best) if bigger else (value < best)
    if is_better:
        cur_step = 0
        best = value
        update_flag = True
    else:
        cur_step += 1

    return best, cur_step, update_flag


def dict2str(result_dict):
    """Convert a metric/result dictionary to a readable string."""
    result_str = ""
    for metric, value in result_dict.items():
        formatted_value = "%.4f" % float(value) if isinstance(value, (int, float)) else str(value)
        result_str += str(metric) + ": " + formatted_value + ", "
    return result_str
