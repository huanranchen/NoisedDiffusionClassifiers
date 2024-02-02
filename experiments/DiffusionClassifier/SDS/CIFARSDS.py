import torch
from data import get_someset_loader, get_CIFAR10_test
from tqdm import tqdm
from models.unets import get_guided_diffusion_unet, get_edm_cifar_uncond
from defenses.PurificationDefenses.DiffPure import (
    EDMEulerIntegralDC,
    EDMEulerIntegralLM,
    VP2EDM,
    EDM2VP,
    EDMStochasticSampler,
    DDIM,
    DiffusionPure,
)
from utils.seed import set_seed
from torchvision import transforms
from utils import concatenate_image
from tester import test_acc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
args = parser.parse_args()
begin, end = args.begin, args.end

set_seed(1)

# ImageNet
# loader = get_someset_loader(
#     "./resources/RestrictedImageNet256",
#     "./resources/RestrictedImageNet256/gt.npy",
#     batch_size=1,
#     shuffle=False,
#     transform=transforms.Compose(
#         [
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#         ]
#     ),
# )
# loader = [item for i, item in enumerate(loader) if begin <= i < end]
# unet = get_guided_diffusion_unet(resolution=256)
#
# dc = EDMEulerIntegralDC(VP2EDM(unet))
# lm = EDMEulerIntegralLM(dc)
# sampler = EDMStochasticSampler(VP2EDM(unet))
#
# xs = []
# for x, _ in tqdm(loader):
#     x = x.cuda()
#     x = x + torch.randn_like(x) * 0.25
#     xs.append(x.clone().cpu())
#     intermediate = lm.SDS_T1(x, iter_step=300, return_intermediate_result=True)[1]
#     xs.extend(intermediate[::30])
# concatenate_image(xs, img_shape=(256, 256, 3), col=len(xs) // (end - begin), row=(end - begin))

# CIFAR10
# -------------------------------------------------------------------------------------------
loader = get_CIFAR10_test(batch_size=1)
loader = [item for i, item in enumerate(loader) if begin <= i < end]
unet = get_edm_cifar_uncond()

dc = EDMEulerIntegralDC(unet)
lm = EDMEulerIntegralLM(dc)
edm_sampler = EDMStochasticSampler(unet)
# ddim = DDIM(EDM2VP(unet), ddpm=False)
sampler = lambda *args, **kwargs: edm_sampler(*args, **kwargs, add_noise=False)

xs = []
for x, _ in tqdm(loader):
    x = x.cuda()
    x = x + torch.randn_like(x) * 0.25
    xs.append(x.clone())
    intermediate = lm.VSD(x, iter_step=10, return_intermediate_result=True)[1]
    xs.extend(intermediate[::1])
    # xs.append(sampler(x, sigma_max=0.5))
concatenate_image(xs, img_shape=(32, 32, 3), col=len(xs) // (end - begin), row=(end - begin))
