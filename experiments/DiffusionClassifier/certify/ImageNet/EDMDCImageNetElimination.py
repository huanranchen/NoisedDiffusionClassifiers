from defenses.PurificationDefenses.DiffPure.DiffusionClassifier import (
    EDMEulerIntegralEliminateFineGrainDC,
    EDMEulerIntegralEliminator
)
from models.unets import get_edm_imagenet_64x64_cond
import torch
from data import get_someset_loader
from tester import test_range_predictor_acc
import argparse
from utils.seed import set_seed
from torchvision import transforms
from copy import deepcopy

set_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--t_steps", type=int, default=126)
args = parser.parse_args()
begin, end, t_steps = args.begin, args.end, args.t_steps


class ResizeTransform:
    def __init__(self):
        self.resizer = transforms.Resize((64, 64))

    def __call__(self, x):
        x = self.resizer(x)
        x = (x - 0.5) * 2
        return x


# loader = get_imagenet_subset(batch_size=1000)
loader = get_someset_loader(
    "./resources/ImageNet512/",
    "./resources/ImageNet512/labels.npy",
    batch_size=1,
)
x, y = next(iter(loader))

loader = [deepcopy(item) for i, item in enumerate(loader) if begin <= i < end]
device = torch.device("cuda")
cond_unet = get_edm_imagenet_64x64_cond(pretrained=True, use_fp16=True)
classifier = EDMEulerIntegralEliminator(
    cond_unet, timesteps=torch.linspace(1e-4, 3, steps=t_steps), transform=ResizeTransform(), num_classes=1000
)

classifier.share_noise = True
classifier.eval().requires_grad_(False).to(device)
test_range_predictor_acc(classifier, loader, verbose=True)
