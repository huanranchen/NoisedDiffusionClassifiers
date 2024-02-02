"""
Diffusion Classifier could also be applied into noised inputs, with distribution p_t(x) or p_sigma(x)
Previously I directly write this into randomized smoothing. But now, other function also need this.
Thus, I reconstruct all p_t(x) classifiers here.
Dependency: All inherited from .DiffusionClassifier
"""
# from .NDCSiftRefine import APNDCEnsembleSiftRefine
# from .NDC import *

import sys
from utils.imports import LazyModule

_import_structure = {
    "NDCSiftRefine": ["APNDCEnsembleSiftRefine"],
    "NDC": [
        "CorrectAPNDC",
        "CorrectAPNDCCondY",
        "EPNDC",
        "APNDCEnsemble",
        "LearnWeightEPNDC",
    ],
}

sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
