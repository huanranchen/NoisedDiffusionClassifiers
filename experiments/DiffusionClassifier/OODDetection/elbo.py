import torch
from data import get_CIFAR10_test, get_CIFAR10_train, get_cifar_10_c_loader
from defenses.PurificationDefenses.DiffPure import SBGC, EDM2VP, EDMEulerIntegralDC
from models.unets import get_edm_cifar_uncond
from models import WideResNet_70_16_dropout
from attacks import BIM, SpectrumSimulationAttack
from tqdm import tqdm

corruptions = [
    "glass_blur",
    "gaussian_noise",
    "shot_noise",
    "speckle_noise",
    "impulse_noise",
    "defocus_blur",
    "gaussian_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "spatter",
    "saturate",
    "frost",
]


unet = get_edm_cifar_uncond(use_fp16=True)
dc = EDMEulerIntegralDC(
    unet,
    timesteps=torch.linspace(1e-4, 3, 126),
)
dc.share_noise = True
dc.eval().requires_grad_(False).cuda()
elbos = []
for corruption in corruptions:
    loader = get_cifar_10_c_loader(batch_size=1, name=corruption)
    for i, (x, y) in enumerate(tqdm(loader)):
        if i > 100:
            break
        x, y = x.cuda(), y.cuda()
        elbo = dc.unet_loss_without_grad(x).item()
        elbos.append(elbo)
    print(corruption, elbos)
    elbos.clear()
# loader = get_CIFAR10_test(batch_size=1)
# attacker = SpectrumSimulationAttack([WideResNet_70_16_dropout()], epsilon=8 / 255, step_size=1 / 255, total_step=100)
# for i, (x, y) in enumerate(tqdm(loader)):
#     if i > 100:
#         break
#     x, y = x.cuda(), y.cuda()
#     adv_x = attacker(x, y)
#     elbo = dc.unet_loss_without_grad(x).item()
#     elbos.append(elbo)
# print(elbos)
