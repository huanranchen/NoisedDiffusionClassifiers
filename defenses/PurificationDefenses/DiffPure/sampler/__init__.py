# from .sde import DiffusionSde, DiffusionOde
# from .ddim import DDIM
# from .edm import EDMStochasticSampler
# from .PreconditionChanger import *
import sys
from utils.imports import LazyModule

_import_structure = {
    "PreconditionChanger": ["EDM2VP", "VP2EDM"],
    "edm": ["EDMStochasticSampler", "EulerMaruyamaWithExtraLangevin"],
    "ddim": ["DDIM"],
    "sde": ["DiffusionSde", "DiffusionOde"],
}

sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
