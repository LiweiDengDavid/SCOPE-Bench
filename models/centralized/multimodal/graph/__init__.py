"""Centralized multimodal graph models."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "BeFA": (".befa", "BeFA"),
    "BGCC": (".bgcc", "BGCC"),
    "CCDRec": (".ccdrec", "CCDRec"),
    "CM3": (".cm3", "CM3"),
    "COHESION": (".cohesion", "COHESION"),
    "DA_MRS": (".da_mrs", "DA_MRS"),
    "DGMRec": (".dgmrec", "DGMRec"),
    "DRAGON": (".dragon", "DRAGON"),
    "DualGNN": (".dualgnn", "DualGNN"),
    "FITMM": (".fitmm", "FITMM"),
    "FREEDOM": (".freedom", "FREEDOM"),
    "GRCN": (".grcn", "GRCN"),
    "LATTICE": (".lattice", "LATTICE"),
    "LGMRec": (".lgmrec", "LGMRec"),
    "MENTOR": (".mentor", "MENTOR"),
    "MIG_GT": (".mig_gt", "MIG_GT"),
    "MGCN": (".mgcn", "MGCN"),
    "MMGCN": (".mmgcn", "MMGCN"),
    "MVGAE": (".mvgae", "MVGAE"),
    "PGL": (".pgl", "PGL"),
    "R2MR": (".r2mr", "R2MR"),
    "REARM": (".rearm", "REARM"),
    "STAIR": (".stair", "STAIR"),
    "TimeMM": (".timemm", "TimeMM"),
    "TMLP": (".tmlp", "TMLP"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
