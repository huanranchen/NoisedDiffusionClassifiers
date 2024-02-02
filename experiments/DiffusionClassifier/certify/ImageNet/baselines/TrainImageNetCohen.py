from data import get_imagenet_loader
from models import resnet50, BaseNormModel
from defenses.AdvTrain import ClassifierSolver
import torch
from optimizer import IdentityScheduler
from torchvision import transforms
import argparse
from tester import test_acc

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", default=0.25, type=float)
args = parser.parse_args()
sigma = args.sigma

loader = get_imagenet_loader(
    split="train",
    batch_size=128,
    augment=True,
    shuffle=True,
    transform=transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
)


fair_transform = transforms.Compose(
    [
        lambda x: x + sigma * torch.randn_like(x),
        transforms.Resize((64, 64)),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_loader = get_imagenet_loader(split="val", batch_size=128)
cnn = BaseNormModel(resnet50(), transform=fair_transform).cuda()

solver = ClassifierSolver(
    cnn,
    optimizer=lambda x: torch.optim.Adam(x.parameters(), lr=1e-3),
    scheduler=IdentityScheduler,
    writer_name=f"resnet50_cohen_sigma={sigma}",
)
solver.train(loader, 20, eval_loader=test_loader)
