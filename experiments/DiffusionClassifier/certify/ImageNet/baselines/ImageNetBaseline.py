import torch
from torchvision.models import resnet50, resnet152
from torchvision import transforms
from models import BaseNormModel
from models.unets import get_guided_diffusion_unet
from tester import test_acc, certify_robustness
from tester.CertifyRobustness import radii_discretion, nA_and_n_to_radii
from data import get_someset_loader
from defenses.RandomizedSmoothing import Smooth
from defenses.PurificationDefenses.DiffPure import CarliniDiffPureForRS, XiaoDiffPureForRS
import argparse
from utils.saver import print_to_file


parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=0.25)
parser.add_argument("--mode", type=str, default="carlini", choices=["carlini", "xiao", "xiao_ensemble"])
parser.add_argument("--n", type=int, default=1000)
args = parser.parse_args()
sigma, mode, n = args.sigma, args.mode, args.n

loader = get_someset_loader(
    "./resources/ImageNet512/",
    "./resources/ImageNet512/labels.npy",
    batch_size=1,
    transform=transforms.Compose([transforms.Resize((256, 256), antialias=None), transforms.ToTensor()]),
)

fair_transform = transforms.Compose(
    [
        transforms.Resize((64, 64), antialias=None),
        transforms.Resize((256, 256), antialias=None),
    ]
)
unet = get_guided_diffusion_unet(pretrained=True, resolution=256, cond=False, use_fp16=True)
model = BaseNormModel(resnet50(pretrained=True)).eval().requires_grad_(False).cuda()
model.load_state_dict(torch.load("./checkpoints/student_resnet50_fair_scratch.pth"))
xiao_diffpure = XiaoDiffPureForRS(
    unet=unet,
    img_shape=(3, 256, 256),
    sigma=sigma,
    post_transforms=fair_transform,
    model=model,
    ensemble_time=1,
)
xiao_ensemble_diffpure = XiaoDiffPureForRS(
    unet=unet,
    img_shape=(3, 256, 256),
    sigma=sigma,
    post_transforms=fair_transform,
    model=model,
)
carlini_diffpure = CarliniDiffPureForRS(
    unet=unet, img_shape=(3, 256, 256), sigma=sigma, post_transforms=fair_transform, model=model
)

classifier = carlini_diffpure if mode == "carlini" else xiao_diffpure if mode == "xiao" else xiao_ensemble_diffpure
print(mode, classifier.__class__)

# test the noisy data acc, to check the code.
test_acc(BaseNormModel(classifier, lambda x: x + torch.randn_like(x) * sigma), loader, verbose=False)

# certify robustness
g = Smooth(classifier, batch_size=32, verbose=False, num_classes=1000, sigma=sigma)
result, nAs, ns, radii = certify_robustness(g, loader, n0=10, n=n, epsilons=(0, 0.25, 0.5, 0.75, 1))
n1000_r = [result, nAs, ns, radii]
print_to_file(n1000_r, f"log_imagenet_baseline_{mode}_{sigma}.txt")
nAs = [i * 10 for i in nAs]
ns = [i * 10 for i in ns]
radii = nA_and_n_to_radii(nAs, ns, sigma=sigma)
radii_discretion(radii, epsilons=(0, 0.25, 0.5, 0.75, 1))
