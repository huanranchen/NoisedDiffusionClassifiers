# from .diffpure import DiffusionPure
# from .DiffusionLikelihoodMaximizer import diffusion_likelihood_maximizer_defense
# from .RS import DiffPureForRandomizedSmoothing, CarliniDiffPureForRS, XiaoDiffPureForRS

import sys
from utils.imports import LazyModule

_import_structure = {
    "diffpure": ["DiffusionPure"],
    "DiffusionLikelihoodMaximizer": ["diffusion_likelihood_maximizer_defense"],
    "RS": ["DiffPureForRandomizedSmoothing", "CarliniDiffPureForRS", "XiaoDiffPureForRS"],
}

sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
