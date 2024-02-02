# from .DiffusionClassifier import DiffusionClassifier, RobustDiffusionClassifier
# from .DiffusionClassifierImageNet import DiffusionAsClassifierImageNetWraped
# from .OptimalDiffusionClassifier import OptimalDiffusionClassifier
# from .SBGC import SBGC
# from .RS import DiffusionClassifierForRandomizedSmoothing, EDMEulerIntegralClassifierForRandomizedSmoothing
# from .PredictX import *
# from .EDMDC import EDMEulerIntegralDC, EDMEulerIntegralWraped
# from .NoiseDiffusionClassifier import *

# from .DiffusionClassifierBase import DiffusionClassifierSingleHeadBaseWraped
from .SearchBasedDC import *

import sys
from utils.imports import LazyModule

_import_structure = {
    "DiffusionClassifierBase": ["DiffusionClassifierSingleHeadBaseWraped"],
    "RS": [
        "DiffusionClassifierForRandomizedSmoothing",
        "EDMEulerIntegralClassifierForRandomizedSmoothing",
    ],
    "EDMDC": ["EDMEulerIntegralDC", "EDMEulerIntegralWraped"],
    "PredictX": ["PredictXDiffusionClassifier", "PredictXDotProductDiffusionClassifier"],
    "SBGC": ["SBGC"],
    "OptimalDiffusionClassifier": ["OptimalDiffusionClassifier"],
    "DiffusionClassifierImageNet": ["DiffusionAsClassifierImageNetWraped"],
    "DiffusionClassifier": ["DiffusionClassifier", "RobustDiffusionClassifier"],
    "NoiseDiffusionClassifier": [
        "APNDCEnsembleSiftRefine",
        "CorrectAPNDC",
        "CorrectAPNDCCondY",
        "EPNDC",
        "APNDCEnsemble",
        "LearnWeightEPNDC",
    ],
    "SearchBasedDC": ["EDMEulerIntegralEliminateFineGrainDC", "EDMEulerIntegralEliminator"],
}

sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
