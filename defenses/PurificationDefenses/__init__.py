from .DiffPure import _import_structure as diffpure_import_structure
# from .PurificationDefense import PurificationDefense


import sys
from utils.imports import LazyModule, sum_all_module_from_submodule


_import_structure = {
    "DiffPure": sum_all_module_from_submodule(diffpure_import_structure),
    "PurificationDefense": ["PurificationDefense"],
}


sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
