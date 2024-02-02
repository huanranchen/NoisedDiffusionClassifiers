import torch
from models.unets import get_edm_cifar_cond
from data import get_CIFAR10_test
from optimizer.GradientNormMaximizer import GradientNormMaximizer
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

dcw = EDMEulerIntegralWraped(unet=model, timesteps=torch.linspace(0.5, 3, 126))
dc = dcw.knnclassifier
dc.share_noise = True
sigma_i_plus_one = dc.timesteps
weight = (sigma_i_plus_one**2 + dc.sigma_data**2) / (sigma_i_plus_one * dc.sigma_data) ** 2 * dc.p_x * dc.dt
# weight = 1
dc.weight = weight
lipschitz = (1 + math.sqrt(2 / math.pi)) * torch.sum(weight / dc.timesteps).item()
print(weight)
print("theoretical logit lipschitz constant is: ", lipschitz)
in_lr, out_lr, iteration = 0.01, 0.1, 1000
for x, y in test_loader:
    x, y = x.cuda(), y.cuda()
    natural_x = x.clone()
    x.requires_grad_(True)
    inner_optim = torch.optim.SGD([x], lr=in_lr)
    outer_optim = torch.optim.Adam([x], lr=out_lr)
    optim = GradientNormMaximizer([x], inner_optim, outer_optim, lr=in_lr)
    for i in range(iteration):
        ori_x = x.clone()
        logit = dcw(x)[0, 0]
        optim.zero_grad()
        logit.backward()
        print(torch.norm(x.grad, p=2), torch.norm(x.grad.view(-1), p=2), torch.norm(x.grad, p="fro"), lipschitz)
        optim.first_step()
        logit = dcw(x)[0, 0]
        optim.zero_grad()
        logit.backward()
        optim.second_step()
        with torch.no_grad():
            x.clamp_(min=0, max=1)
            # x.clamp_(min=natural_x - 8 / 255, max=natural_x + 8 / 255)
            print((torch.abs(x - ori_x)).mean(), torch.min(x), torch.max(x))
