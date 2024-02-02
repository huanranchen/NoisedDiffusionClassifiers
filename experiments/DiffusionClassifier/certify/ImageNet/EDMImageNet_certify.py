from models.unets import get_edm_imagenet_64x64_cond, get_guided_diffusion_unet
from models import BaseNormModel, FixNoiseModel
import torch
from defenses.PurificationDefenses.DiffPure.DiffusionClassifier import APNDCEnsembleSiftRefine
from data import get_someset_loader
from defenses.RandomizedSmoothing.core import Smooth
from tester import certify_robustness, test_acc
from tester.CertifyRobustness import nA_and_n_to_radii, radii_discretion
import argparse
from utils.seed import set_seed
from utils.saver import print_to_file
from defenses.PurificationDefenses.DiffPure.sampler import VP2EDM
from torchvision import transforms
from data.someset import save_to_someset
from models import resnet50
from copy import deepcopy


set_seed(1)


parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--sigma", type=float, default=0.25)
parser.add_argument("--t_steps", type=int, default=1001)
args = parser.parse_args()
begin, end, sigma, t_steps = args.begin, args.end, args.sigma, args.t_steps


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
uncond_unet = VP2EDM(get_guided_diffusion_unet(pretrained=True, resolution=256, cond=False, use_fp16=True))
cond_unet = get_edm_imagenet_64x64_cond(pretrained=True, use_fp16=True)
classifier = APNDCEnsembleSiftRefine(
    uncond_unet, cond_unet, sigma=sigma, steps=t_steps, transform=ResizeTransform(), num_classes=1000
)

classifier.share_noise = True
classifier.eval().requires_grad_(False).to(device)
# test_acc(BaseNormModel(classifier, lambda x: x + torch.randn_like(x) * sigma), loader, verbose=True)

g = Smooth(classifier, batch_size=8, verbose=False, num_classes=1000, sigma=sigma)
result, nAs, ns, radii = certify_robustness(g, loader, n0=10, n=100, epsilons=(0.5, 1, 1.5, 2, 3))
n100_r = [result, nAs, ns, radii]
print_to_file(n100_r, f"log_imagenet_{begin}_{end}_{sigma}_{t_steps}.txt")
nAs = [i * 10 for i in nAs]
ns = [i * 10 for i in ns]
radii = nA_and_n_to_radii(nAs, ns, sigma=sigma)
radii_discretion(radii, epsilons=(0, 0.25, 0.5, 0.75, 1))
nAs = [i * 10 for i in nAs]
ns = [i * 10 for i in ns]
radii = nA_and_n_to_radii(nAs, ns, sigma=sigma)
radii_discretion(radii, epsilons=(0, 0.25, 0.5, 0.75, 1))
