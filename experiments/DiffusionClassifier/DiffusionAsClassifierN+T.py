import torch
from models.unets import get_NCSNPP_cached, DiffusionClassifierCached, get_edm_cifar_uncond
from data import get_CIFAR10_test
from tester import test_acc, test_apgd_dlr_acc
import argparse
from defenses.PurificationDefenses.DiffPure.DiffPure.DiffusionLikelihoodMaximizer import (
    diffusion_likelihood_maximizer_defense,
)
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC
from defenses.PurificationDefenses.DiffPure.LikelihoodMaximization import EDMEulerIntegralLM
from attacks import StAdvAttack
from tester import test_robustness

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()
begin, end = args.begin, args.end


# FIXME: 目前likelihood maximization的diffusion和分类的diffusion不一致
model = get_NCSNPP_cached(grad_checkpoint=True).cuda()
test_loader = get_CIFAR10_test(batch_size=1)
test_loader = [item for i, item in enumerate(test_loader) if begin <= i < end]
model.load_state_dict(
    torch.load("/workspace/home/chenhuanran2022/work/Diffusion/" "checkpoints/cached_kd_edm_all/ema_1800.pt")
)

edm_unet = get_edm_cifar_uncond()
edm_unet.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
purify_dc = EDMEulerIntegralDC(unet=edm_unet)
lm = EDMEulerIntegralLM(purify_dc)

classifier = DiffusionClassifierCached(model)

defensed = diffusion_likelihood_maximizer_defense(classifier, lm.likelihood_maximization_T1)

# test_apgd_dlr_acc(defensed, loader=test_loader, norm="L2", eps=0.5)
attacker = StAdvAttack(defensed, num_iterations=100, bound=0.05)
test_robustness(attacker, test_loader, [defensed])
