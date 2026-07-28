# coding: utf-8

"""Framework-level logging configuration and structured training logger."""

import logging
import importlib.util
import re
from typing import Dict, Optional, List

if importlib.util.find_spec("coloredlogs") is not None:
    import coloredlogs
else:
    coloredlogs = None

LOGGER_NAME = "nexusrec"
DEFAULT_KEY_METRICS = ("Recall@10", "NDCG@10", "Precision@10")

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


class _StripAnsiFormatter(logging.Formatter):
    """Formatter that strips ANSI escape codes for file output."""

    def format(self, record):
        msg = super().format(record)
        return _ANSI_RE.sub('', msg)


def _get_logger():
    """Return the framework namespace logger."""
    return logging.getLogger(LOGGER_NAME)


def init_logger(config):
    """Configure the ``nexusrec`` namespace logger.

    Only touches the ``nexusrec`` logger — the root logger and any
    third-party loggers remain unaffected.

    Args:
        config: Configuration dict with ``log_file_name``, ``state``, etc.
    """
    log_file = config['log_file_name']

    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False
    logger.handlers.clear()

    fmt = "%(asctime)-15s %(levelname)s %(message)s"
    console_datefmt = "%d %b %H:%M"
    file_datefmt = "%a %d %b %Y %H:%M:%S"

    state = config["state"]
    level = getattr(logging, state.upper(), logging.INFO)

    if coloredlogs is not None:
        # coloredlogs replaces existing handlers, so install it before the file handler.
        coloredlogs.install(
            level=level,
            logger=logger,
            fmt=fmt,
            datefmt=console_datefmt,
        )
    else:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(fmt, console_datefmt))
        logger.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file, 'w', 'utf-8')
        fh.setLevel(level)
        fh.setFormatter(_StripAnsiFormatter(fmt, file_datefmt))
        logger.addHandler(fh)


# ---------------------------------------------------------------------------
# TrainLogger — training-output formatting (merged from core/utils/logger.py)
# ---------------------------------------------------------------------------

from .monitor import get_system_status


class TrainLogger:
    """Centralized training logger

    Single point for all training-related logging to eliminate scattered logs.
    """

    @staticmethod
    def log_epoch_progress(
        epoch: int,
        total_epochs: int,
        stage: str,
        loss: Optional[float] = None,
        score: Optional[float] = None,
        epoch_time: Optional[float] = None,
        num_batches: int = 0,
    ):
        """Log epoch progress with unified format

        Args:
            epoch: Current epoch (1-based)
            total_epochs: Total epochs
            stage: Train/Valid
            loss: Training loss (for Train stage)
            score: Validation score (for Valid stage)
            epoch_time: Time elapsed for this epoch
            num_batches: Number of batches processed
        """
        logger = _get_logger()

        # Build time info
        time_str = ""
        if epoch_time is not None:
            time_str = f"Time: {epoch_time:.2f}s, "

        if stage == "Train":
            system_status = get_system_status()
            it_per_sec = (
                num_batches / epoch_time if epoch_time and epoch_time > 0 else 0
            )
            loss_str = f"Loss: {loss:.4f}" if loss is not None else ""

            logger.info(
                f"[Epoch {epoch}/{total_epochs}][Train] "
                f"{time_str}"
                f"{system_status}, "
                f"Speed: {it_per_sec:.1f}it/s, "
                f"{loss_str}"
            )

        elif stage == "Valid":
            # Validation progress
            score_str = f"Score: {score:.4f}" if score is not None else ""

            logger.info(
                f"[Epoch {epoch}/{total_epochs}][Valid] " f"{time_str}" f"{score_str}"
            )

    @staticmethod
    def log_detailed_metrics(
        metrics: Dict[str, float],
        stage: str = "Valid",
        config_metrics: Optional[List[str]] = None,
        config_topk: Optional[List[int]] = None,
    ):
        """Log detailed metrics in demo format

        Args:
            metrics: Dictionary of evaluation metrics
            stage: Valid/Test
            config_metrics: Metric types from config (e.g., ['Recall', 'NDCG', 'Precision'])
            config_topk: TopK values from config (e.g., [10, 20])
        """
        if not metrics:
            return

        logger = _get_logger()

        # Use config-based metric order when provided.
        if config_metrics and config_topk:
            metric_order = [f"{m.lower()}@{k}" for m in config_metrics for k in config_topk]
        else:
            metric_order = sorted(metrics.keys(), key=str.lower)

        # Build a normalized lookup for O(1) matching
        norm_metrics = {k.lower(): v for k, v in metrics.items()}
        metric_parts = []
        for metric in metric_order:
            if metric in norm_metrics:
                metric_parts.append(f"{metric}: {norm_metrics[metric]:.4f}")

        if metric_parts:
            metric_str = ", ".join(metric_parts)
            logger.info(f"[{stage}] {metric_str}")

    @staticmethod
    def log_best_result(
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = True,
        key_metrics: Optional[List[str]] = None,
    ):
        """Log best result notification

        Args:
            epoch: Epoch number
            metrics: Metrics achieved
            is_best: Whether this is a new best result
            key_metrics: List of key metrics to display (defaults to common metrics)
        """
        logger = _get_logger()

        if key_metrics is None:
            key_metrics = DEFAULT_KEY_METRICS

        # Format key metrics
        metric_parts = []
        for metric in key_metrics:
            if metric in metrics:
                metric_parts.append(f"{metric}={metrics[metric]:.4f}")

        metric_str = ", ".join(metric_parts)

        if is_best:
            logger.info(f">>> New Best Results on Epoch {epoch} | {metric_str}")
        else:
            logger.info(f">>> Current: Epoch {epoch} | {metric_str}")

    @staticmethod
    def log_training_summary(
        total_time: float,
        best_valid_metrics: Dict[str, float],
        total_epochs: int,
        best_epoch: int,
        key_metrics: Optional[List[str]] = None,
        best_test_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log training completion summary with both validation and test results

        Args:
            total_time: Total training time in seconds
            best_valid_metrics: Best validation metrics achieved
            total_epochs: Total epochs trained
            best_epoch: Epoch when best result was achieved
            key_metrics: List of key metrics to display (defaults to common metrics)
            best_test_metrics: Best test metrics corresponding to best validation
        """
        logger = _get_logger()

        # Format time
        if total_time < 60:
            time_str = f"{total_time:.0f}s"
        elif total_time < 3600:
            time_str = f"{total_time/60:.1f}min"
        else:
            time_str = f"{total_time/3600:.1f}h"

        if key_metrics is None:
            key_metrics = DEFAULT_KEY_METRICS

        # Format validation metrics
        valid_parts = []
        for metric in key_metrics:
            if metric in best_valid_metrics:
                valid_parts.append(f"{metric}={best_valid_metrics[metric]:.4f}")

        valid_str = ", ".join(valid_parts)

        # Format test metrics if available
        test_str = ""
        if best_test_metrics:
            test_parts = []
            for metric in key_metrics:
                if metric in best_test_metrics:
                    test_parts.append(f"{metric}={best_test_metrics[metric]:.4f}")
            test_str = ", ".join(test_parts)

        # Create comprehensive summary
        base_summary = (
            f"Training Completed: Total Time: {time_str} | Best Epoch {best_epoch}"
        )

        if test_str:
            # Show both validation and test metrics clearly
            logger.info("=" * 60)
            logger.info(f"{base_summary}")
            logger.info(f"Best Valid Metrics: {valid_str}")
            logger.info(f"Best Test Metrics:  {test_str}")
            logger.info("=" * 60)
        else:
            summary = f"{base_summary} | Best Metrics: {valid_str}"
            logger.info(summary)
