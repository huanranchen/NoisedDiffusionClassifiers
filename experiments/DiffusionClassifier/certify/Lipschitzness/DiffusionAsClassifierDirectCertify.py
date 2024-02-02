import torch
from models.unets import get_edm_cifar_uncond, get_edm_cifar_cond
from data import get_CIFAR10_test
from tester.CertifyRobustness import certify_robustness_via_lipschitz
import math
import argparse
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC, EDMEulerIntegralWraped


parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int, default=0)
parser.add_argument("--end", type=int, default=512)
args = parser.parse_args()
begin, end = args.begin, args.end

model = get_edm_cifar_cond().cuda()
test_loader = get_CIFAR10_test(batch_size=1)
test_loader = [item for i, item in enumerate(test_loader) if begin <= i < end]
model.load_state_dict(torch.load("../../../../resources/checkpoints/EDM/edm_cifar_cond.pt"))

dc = EDMEulerIntegralDC(unet=model, timesteps=torch.linspace(0.5, 3, 126))
dc.share_noise = True
sigma_i_plus_one = dc.timesteps
# weight = (sigma_i_plus_one**2 + dc.sigma_data**2) / (sigma_i_plus_one * dc.sigma_data) ** 2 * dc.p_x * dc.dt
weight = 1
dc.weight = weight
lipschitz = (1 + math.sqrt(2 / math.pi)) * torch.mean(weight / dc.timesteps).item()
print(weight)
print("lipschitz constant is: ", lipschitz)
certify_robustness_via_lipschitz(dc, test_loader, lipschitz, verbose=True)
