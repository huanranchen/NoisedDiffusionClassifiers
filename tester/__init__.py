from utils.imports import LazyModule
import sys

_import_structure = {
    "TestAcc": ["test_acc"],
    "TransferAttackAcc": ["test_transfer_attack_acc", "test_robustness"],
    "AutoAttack": ["test_apgd_dlr_acc", "test_autoattack_acc"],
    "CertifyRobustness": ["certify_robustness"],
}

sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
