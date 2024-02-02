"""
Reference: https://github.com/jh-jeong/smoothing-consistency/blob/master/code/consistency.py
"""
from data import get_imagenet_loader
from models import resnet50, BaseNormModel
from defenses.AdvTrain import ClassifierSolver
import torch
from optimizer import IdentityScheduler
from torchvision import transforms
import argparse
from torch.nn import functional as F


parser = argparse.ArgumentParser()
parser.add_argument("--sigma", default=0.25, type=float)
args = parser.parse_args()
sigma = args.sigma


def kl_div(input, targets, eps=1e-7):
    log_input = F.log_softmax(input, dim=1)
    targets = torch.clamp(targets, min=eps, max=1 - eps)
    return torch.nn.KLDivLoss(reduction="none")(log_input, targets).sum(1)
    # return F.kl_div(F.log_softmax(input, dim=1), targets, reduction="none").sum(1)


def entropy(input):
    logsoftmax = torch.log(input.clamp(min=1e-7))
    xent = (-input * logsoftmax).sum(1)
    return xent


def consistency_loss(logits, lbd, eta=0.5, loss="default"):
    """
    Consistency regularization for certified robustness.

    Parameters
    ----------
    logits : List[torch.Tensor]
        A list of logit batches of the same shape, where each
        is sampled from f(x + noise) with i.i.d. noises.
        len(logits) determines the number of noises, i.e., m > 1.
    lbd : float
        Hyperparameter that controls the strength of the regularization.
    eta : float (default: 0.5)
        Hyperparameter that controls the strength of the entropy term.
        Currently used only when loss='default'.
    loss : {'default', 'xent', 'kl', 'mse'} (optional)
        Which loss to minimize to obtain consistency.
        - 'default': The default form of loss.
            All the values in the paper are reproducible with this option.
            The form is equivalent to 'xent' when eta = lbd, but allows
            a larger lbd (e.g., lbd = 20) when eta is smaller (e.g., eta < 1).
        - 'xent': The cross-entropy loss.
            A special case of loss='default' when eta = lbd. One should use
            a lower lbd (e.g., lbd = 3) for better results.
        - 'kl': The KL-divergence between each predictions and their average.
        - 'mse': The mean-squared error between the first two predictions.

    """

    m = len(logits)
    softmax = [F.softmax(logit, dim=1) for logit in logits]
    avg_softmax = sum(softmax) / m

    loss_kl = [kl_div(logit, avg_softmax) for logit in logits]
    loss_kl = sum(loss_kl) / m

    if loss == "default":
        loss_ent = entropy(avg_softmax)
        consistency = lbd * loss_kl + eta * loss_ent
    elif loss == "xent":
        loss_ent = entropy(avg_softmax)
        consistency = lbd * (loss_kl + loss_ent)
    elif loss == "kl":
        consistency = lbd * loss_kl
    elif loss == "mse":
        sm1, sm2 = softmax[0], softmax[1]
        loss_mse = ((sm2 - sm1) ** 2).sum(1)
        consistency = lbd * loss_mse
    else:
        raise NotImplementedError()

    return consistency.mean()


class ConsistencyRSDataExpander:
    def __init__(self, m=2):
        self.m = m

    def __call__(self, x, y):
        noise = torch.randn(self.m * x.shape[0], *x.shape[1:], device=x.device) * sigma
        x, y = x.repeat(self.m, 1, 1, 1), y.repeat(self.m)
        x = x + noise
        return x, y


class ConsistencyRSCriterion:
    def __init__(self, m=2, criterion=torch.nn.CrossEntropyLoss()):
        self.criterion = criterion
        self.m = m

    def __call__(self, logits, y):
        loss_xent = self.criterion(logits, y)
        logits_chunk = torch.chunk(logits, self.m, dim=0)
        loss_con = consistency_loss(logits_chunk, 10, 0.5)
        loss = loss_xent + loss_con
        return loss


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
    writer_name=f"resnet50_consistency_sigma={sigma}",
    criterion=ConsistencyRSCriterion(),
    pre_processor=ConsistencyRSDataExpander(),
)
solver.train(loader, 50, eval_loader=test_loader)
