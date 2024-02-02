from defenses.PurificationDefenses.DiffPure.DiffusionClassifier import (
    EPNDC
)
from models.unets import get_edm_cifar_uncond, get_edm_cifar_cond
from models import BaseNormModel, FixNoiseModel
import torch
from data import get_CIFAR10_test
from defenses.RandomizedSmoothing.core import Smooth
from tester import certify_robustness, test_acc
from tester.CertifyRobustness import nA_and_n_to_radii, radii_discretion
import argparse
from utils.seed import set_seed
from utils.saver import print_to_file

set_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int, default=0)
parser.add_argument("--end", type=int, default=512)
parser.add_argument("--sigma", type=float, default=0.25)
parser.add_argument("--t_steps", type=int, default=126)
args = parser.parse_args()
begin, end, sigma, t_steps = args.begin, args.end, args.sigma, args.t_steps

loader = get_CIFAR10_test(batch_size=1)
loader = [item for i, item in enumerate(loader) if begin <= i < end]
device = torch.device("cuda")
# classifier = NoisedEDMEulerIntegralDC(
#     get_edm_cifar_uncond(use_fp16=True), get_edm_cifar_cond(use_fp16=True), sigma=sigma, steps=t_steps
# )
classifier = EPNDC(
    get_edm_cifar_cond(use_fp16=True), sigma=sigma, steps=t_steps
)

classifier.share_noise = True
# classifier.share_noise = False
# classifier = FixNoiseModel(classifier)
classifier.eval().requires_grad_(False).to(device)
# test_acc(BaseNormModel(classifier, lambda x: x + torch.randn_like(x) * sigma), loader, verbose=True)

# g = Smooth(classifier, batch_size=1, verbose=True)
g = Smooth(classifier, batch_size=32, verbose=False, sigma=sigma)
result, nAs, ns, radii = certify_robustness(g, loader, n0=100, n=1000)
n1000_r = [result, nAs, ns, radii]
print_to_file(n1000_r, f"log_{begin}_{end}_{sigma}_{t_steps}.txt")
nAs = [i * 10 for i in nAs]
ns = [i * 10 for i in ns]
radii = nA_and_n_to_radii(nAs, ns, sigma=sigma)
radii_discretion(radii)
nAs = [i * 10 for i in nAs]
ns = [i * 10 for i in ns]
radii = nA_and_n_to_radii(nAs, ns, sigma=sigma)
radii_discretion(radii)
