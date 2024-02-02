from .DiffPure import _import_structure as diffpure_import_structure
from .DiffusionClassifier import _import_structure as dc_import_structure
from .sampler import _import_structure as sampler_import_structure

import sys
from utils.imports import LazyModule, sum_all_module_from_submodule

_import_structure = {
    "DiffPure": sum_all_module_from_submodule(diffpure_import_structure),
    "DiffusionClassifier": sum_all_module_from_submodule(dc_import_structure),
    "sampler": sum_all_module_from_submodule(sampler_import_structure),
}

sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
