import torch
from torchvision.models import resnet50, resnet152
from torchvision import transforms
from models import BaseNormModel
from tester import test_acc, certify_robustness
from tester.CertifyRobustness import radii_discretion, nA_and_n_to_radii
from data import get_someset_loader, get_imagenet_loader
from defenses.RandomizedSmoothing import Smooth
import argparse
from utils.saver import print_to_file
from copy import deepcopy


parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=0.25)
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--n", type=int, default=1000)
parser.add_argument("--set", type=str, default="test")
args = parser.parse_args()
sigma, ckpt_path, n = args.sigma, args.ckpt_path, args.n

if args.set == "test":
    loader = get_someset_loader(
        "./resources/ImageNet512/",
        "./resources/ImageNet512/labels.npy",
        batch_size=1,
        transform=transforms.Compose([transforms.Resize((256, 256), antialias=None), transforms.ToTensor()]),
    )
else:
    train_loader = iter(get_imagenet_loader(
        split="train",
        batch_size=128,
        shuffle=True,
        transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
    ))
    loader = []
    for i in range(4):
        loader.append(next(train_loader))

fair_transform = transforms.Compose(
    [
        transforms.Resize((64, 64), antialias=None),
        transforms.Resize((224, 224), antialias=None),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
cnn = BaseNormModel(resnet50(), transform=fair_transform).cuda()
cnn.load_state_dict(torch.load(ckpt_path))
cnn.eval().requires_grad_(False).cuda()
print(ckpt_path, cnn.__class__)

# test the noisy data acc, to check the code.
test_acc(BaseNormModel(cnn, lambda x: x + torch.randn_like(x) * sigma), loader, verbose=False)

# # certify robustness
g = Smooth(cnn, batch_size=256, verbose=False, num_classes=1000, sigma=sigma)
result, nAs, ns, radii = certify_robustness(g, loader, n0=10, n=n, epsilons=(0, 0.25, 0.5, 0.75, 1))
n1000_r = [result, nAs, ns, radii]
ckpt_name = ckpt_path.split("/")[-1]  # for naming the log
print_to_file(n1000_r, f"log_imagenet_baseline_{ckpt_name}_{sigma}.txt")
nAs = [i * 10 for i in nAs]
ns = [i * 10 for i in ns]
radii = nA_and_n_to_radii(nAs, ns, sigma=sigma)
radii_discretion(radii, epsilons=(0, 0.25, 0.5, 0.75, 1))
