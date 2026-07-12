"""DiffMM support components."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "UserFeatureAggregator": (".fusion", "UserFeatureAggregator"),
    "NoiseScheduler": (".diffusion", "NoiseScheduler"),
    "TimeEmbedding": (".diffusion", "TimeEmbedding"),
    "DenoisingNetwork": (".diffusion", "DenoisingNetwork"),
    "DiffusionProcess": (".diffusion", "DiffusionProcess"),
    "create_diffusion_process": (".diffusion", "create_diffusion_process"),
    "MultiModalAugmentation": (".contrastive", "MultiModalAugmentation"),
    "InfoNCELoss": (".contrastive", "InfoNCELoss"),
    "ContrastiveLearningModule": (".contrastive", "ContrastiveLearningModule"),
    "create_contrastive_learning_module": (".contrastive", "create_contrastive_learning_module"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
