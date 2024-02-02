import torch
from data import get_cifar_10_c_loader
from tester import test_acc
from torchvision import transforms
from models.unets import get_edm_cifar_cond
from models import WideResNet_70_16_dropout
from models.RobustBench.cifar10 import Wang2023Better, Rebuffi2021Fixing
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC, DiffusionPure
from utils import set_seed


set_seed(1)

to_img = transforms.ToPILImage()
corruptions = [
    'glass_blur',
    'gaussian_noise',
    'shot_noise',
    'speckle_noise',
    'impulse_noise',
    'defocus_blur',
    'gaussian_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
    'spatter',
    'saturate',
    'frost',
]

device = torch.device('cuda')
# classifier = EDMEulerIntegralDC(get_edm_cifar_cond(use_fp16=True), timesteps=torch.linspace(1e-4, 3, 126))
# classifier.eval().requires_grad_(False)
# classifier = WideResNet_70_16_dropout()
# classifier = Rebuffi2021Fixing()
classifier = DiffusionPure()
classifier.share_noise = True
accs = []
for name in corruptions:
    loader = get_cifar_10_c_loader(name=name, batch_size=500)
    print(f'now testing **{name}**, total images {len(loader)}')
    _, now_acc = test_acc(classifier, loader)
    accs.append(now_acc)
    print('-' * 100)
acc = sum(accs) / len(accs)
print(acc)
