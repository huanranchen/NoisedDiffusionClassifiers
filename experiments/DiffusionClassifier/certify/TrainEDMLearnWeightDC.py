from defenses.PurificationDefenses.DiffPure.DiffusionClassifier import (
    LearnWeightEPNDC,
)
from defenses import ClassifierSolver
from models.unets import get_edm_cifar_uncond, get_edm_cifar_cond
from models import BaseNormModel
import torch
from data import get_CIFAR10_test, get_CIFAR10_train
from attacks import IdentityAttacker
import argparse
from optimizer import IdentityScheduler
from utils.seed import set_seed
from torchvision import transforms

set_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--bs", type=int, default=256)
args = parser.parse_args()
begin, end, bs = args.begin, args.end, args.bs
sigma = 0.25
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


def data_transformation(x):
    x = transform(x)
    x = torch.randn_like(x) * sigma + x
    return x


train_loader = get_CIFAR10_train(transform=data_transformation, batch_size=bs)


loader = get_CIFAR10_test(batch_size=1)
loader = [item for i, item in enumerate(loader) if begin <= i < end]
device = torch.device("cuda")

classifier = LearnWeightEPNDC(get_edm_cifar_cond(), sigma=sigma)
classifier.eval().requires_grad_(False).to(device)

solver = ClassifierSolver(
    classifier,
    optimizer=lambda x: torch.optim.Adam(x.parameters(), lr=1e-2),
    scheduler=IdentityScheduler,
    writer_name="TrainEDMLearnWeightDC",
)
solver.train(train_loader, eval_loader=loader)
