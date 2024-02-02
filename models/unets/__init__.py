from utils.imports import LazyModule
import sys

_import_structure = {
    "score_sde": ["get_NCSNPP", "get_NCSNPP_cached", "DiffusionClassifierCached"],
    "EDM": ["get_edm_cifar_uncond", "get_edm_cifar_cond", "get_edm_cifar_unet", "get_edm_imagenet_64x64_cond"],
    "guided_diffusion": ["get_guided_diffusion_unet"],
    "DiffuserUNets": ["DDPMCIFARUnet"],
}

sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
