from data import get_imagenet_loader
from models import resnet50, BaseNormModel
from defenses.AdvTrain import AdversarialTraining
import torch
from torch import Tensor
from optimizer import ALRS, SGD
from torchvision import transforms
import argparse
from attacks import BIM, IdentityAttacker


parser = argparse.ArgumentParser()
parser.add_argument("--sigma", default=0.25, type=float)
args = parser.parse_args()
sigma = args.sigma

loader = get_imagenet_loader(
    split="train",
    batch_size=128,
    shuffle=True,
    transform=transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            lambda x: x + sigma * torch.randn_like(x),  # do not put it into fair transform
        ]
    ),
)


fair_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_loader = get_imagenet_loader(
    split="val",
    batch_size=128,
    transform=transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            lambda x: x + sigma * torch.randn_like(x),  # do not put it into fair transform
        ]
    ),
)
cnn = BaseNormModel(resnet50(), transform=fair_transform).cuda()


class SalmanRSAttack(BIM):
    def __init__(self, *a, epsilon=0.25, total_step=2, **kwargs):
        kwargs["norm"] = "L2"
        kwargs["epsilon"] = epsilon
        kwargs["total_step"] = total_step
        kwargs["step_size"] = 2 / 255
        super().__init__(*a, **kwargs)

    @torch.no_grad()
    def clamp(self, x: Tensor, ori_x: Tensor) -> Tensor:
        """
        Rewrite clamp. So that won't clamp x into [0, 1]
        """
        B = x.shape[0]
        difference = x - ori_x
        distance = torch.norm(difference.view(B, -1), p=2, dim=1)
        mask = distance > self.epsilon
        if torch.sum(mask) > 0:
            difference[mask] = difference[mask] / distance[mask].view(torch.sum(mask), 1, 1, 1) * self.epsilon
            x = ori_x + difference
        return x


attacker = SalmanRSAttack([cnn], epsilon=sigma)
solver = AdversarialTraining(
    attacker,
    cnn,
    optimizer=lambda x: torch.optim.SGD(x.parameters(), lr=1e-1),
    scheduler=ALRS,
    writer_name=f"resnet50_salman_sigma={sigma}",
)
solver.train(
    loader,
    50,
    eval_loader=test_loader,
    test_attacker=IdentityAttacker(),
)
