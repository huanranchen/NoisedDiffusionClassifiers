# from .EliminateFineGrain import EDMEulerIntegralEliminateFineGrainDC, EDMEulerIntegralEliminator

import sys
from utils.imports import LazyModule

_import_structure = {
    "EliminateFineGrain": ["EDMEulerIntegralEliminateFineGrainDC", "EDMEulerIntegralEliminator"],
}

sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
