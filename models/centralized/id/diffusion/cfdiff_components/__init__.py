"""CFDiff support components."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "CAM_AE": (".cam_ae", "CAM_AE"),
    "CAM_AE_multihops": (".cam_ae_multihops", "CAM_AE_multihops"),
    "GaussianDiffusion": (".gaussian_diffusion", "GaussianDiffusion"),
    "ModelMeanType": (".gaussian_diffusion", "ModelMeanType"),
    "MultiHopProcessor": (".multihop_processor", "MultiHopProcessor"),
    "get_named_beta_schedule": (".diffusion_utils", "get_named_beta_schedule"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
