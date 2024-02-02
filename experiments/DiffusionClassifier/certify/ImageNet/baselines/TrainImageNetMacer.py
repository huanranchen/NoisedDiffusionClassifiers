"""
Reference: https://github.com/jh-jeong/smoothing-consistency/blob/master/code/train_macer.py
"""
from data import get_imagenet_loader
from models import resnet50, BaseNormModel
from defenses.AdvTrain import ClassifierSolver
import torch
from optimizer import IdentityScheduler
from torchvision import transforms
import argparse
from torch.nn import functional as F
from torch.distributions.normal import Normal


parser = argparse.ArgumentParser()
parser.add_argument("--sigma", default=0.25, type=float)
args = parser.parse_args()
sigma = args.sigma


class MacerCriterion:
    def __init__(self, lbd=16, gauss_num=2, beta=16, gamma=0.1, num_classes=1000, device=torch.device("cuda")):
        self.m = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        self.sigma = sigma
        self.lbd = lbd
        self.gauss_num = gauss_num
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        targets = targets[: targets.numel() // self.gauss_num]
        input_size = targets.numel()
        outputs = outputs.reshape(self.gauss_num, input_size, self.num_classes)
        outputs = outputs.permute(1, 0, 2)  # (input_size, gauss_num, num_classes)
        # Classification loss
        outputs_softmax = F.softmax(outputs, dim=2).mean(1)
        outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
        classification_loss = F.nll_loss(outputs_logsoftmax, targets, reduction="sum")

        if self.lbd == 0:
            robustness_loss = classification_loss * 0
        else:
            # Robustness loss
            beta_outputs = outputs * self.beta  # only apply beta to the robustness loss
            beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
            top2 = torch.topk(beta_outputs_softmax, 2)
            top2_score = top2[0]
            top2_idx = top2[1]
            indices_correct = top2_idx[:, 0] == targets  # G_theta

            out0, out1 = top2_score[indices_correct, 0], top2_score[indices_correct, 1]
            robustness_loss = self.m.icdf(out1) - self.m.icdf(out0)
            indices = (
                ~torch.isnan(robustness_loss)
                & ~torch.isinf(robustness_loss)
                & (torch.abs(robustness_loss) <= self.gamma)
            )  # hinge
            out0, out1 = out0[indices], out1[indices]
            robustness_loss = self.m.icdf(out1) - self.m.icdf(out0) + self.gamma
            robustness_loss = robustness_loss.sum() * sigma / 2

        # Final objective function
        loss = classification_loss + self.lbd * robustness_loss
        loss /= input_size
        return loss


class MacerRSDataExpander:
    def __init__(self, gauss_num=2, num_classes=1000):
        self.gauss_num = gauss_num
        self.num_classes = num_classes

    def __call__(self, inputs, y):
        inputs = inputs.repeat((self.gauss_num, 1, 1, 1))
        noise = torch.randn_like(inputs, device=inputs.device) * sigma
        noisy_inputs = inputs + noise
        return noisy_inputs, y.repeat(self.gauss_num)


loader = get_imagenet_loader(
    split="train",
    batch_size=128,
    shuffle=True,
    transform=transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    drop_last=True,
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
            lambda x: x + torch.randn_like(x) * sigma,
        ]
    ),
)
cnn = BaseNormModel(resnet50(), transform=fair_transform).cuda()


solver = ClassifierSolver(
    cnn,
    optimizer=lambda x: torch.optim.Adam(x.parameters(), lr=1e-3),
    scheduler=IdentityScheduler,
    writer_name=f"resnet50_macer_sigma={sigma}",
    criterion=MacerCriterion(),
    pre_processor=MacerRSDataExpander(),
)
solver.train(loader, 50, eval_loader=test_loader)
