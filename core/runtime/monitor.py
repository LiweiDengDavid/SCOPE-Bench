# coding: utf-8
"""System resource monitoring helpers."""

from __future__ import annotations

import logging

import psutil
import torch

_logger = logging.getLogger("nexusrec")


def get_memory_usage():
    """Return `(used_gb, total_gb)` for system memory."""
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    return used_gb, total_gb


def get_gpu_usage():
    """Return `(reserved_gb, total_gb)` for the active GPU, if available."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None, None

    device_idx = torch.cuda.current_device()
    if device_idx >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid CUDA device index: {device_idx}")
    reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
    total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
    return reserved, total


def get_system_status():
    """Return a compact system status string for runtime logging."""
    mem_used, mem_total = get_memory_usage()
    gpu_used, gpu_total = get_gpu_usage()
    cpu_percent = psutil.cpu_percent()

    status_parts = [
        f"CPU: {cpu_percent:.1f}%",
        f"RAM: {mem_used:.1f}/{mem_total:.1f}GB",
    ]
    if gpu_used is not None:
        status_parts.append(f"GPU: {gpu_used:.1f}/{gpu_total:.1f}GB")
    return ", ".join(status_parts)
