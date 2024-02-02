# from .Transformations import BitDepthReduction, Randomization, JPEGCompression
# from .NeuralRepresentationPurifier import NeuralRepresentationPurifier
# from .RandomizedSmoothing import randomized_smoothing_resnet50
from .PurificationDefenses import _import_structure as purification_import_structure
# from .AdvTrain import AdversarialTraining, ClassifierSolver
import sys
from utils.imports import LazyModule, sum_all_module_from_submodule

_import_structure = {
    "Transformations": ["BitDepthReduction", "Randomization", "JPEGCompression"],
    "NeuralRepresentationPurifier": ["NeuralRepresentationPurifier"],
    "RandomizedSmoothing": ["Smooth", "randomized_smoothing_resnet50"],
    "AdvTrain": ["AdversarialTraining", "ClassifierSolver"],
    "PurificationDefenses": sum_all_module_from_submodule(purification_import_structure),
}


sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
