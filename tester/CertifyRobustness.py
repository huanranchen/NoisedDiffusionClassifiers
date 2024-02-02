import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Iterable, Dict, Tuple, List
from tqdm import tqdm
from defenses.RandomizedSmoothing import Smooth
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

__WRONG_PREDICTION__ = -1


def robust_radius_given_correct_num(nA: int = 990, n: int = 1000, alpha: float = 0.001, sigma: float = 0.25) -> float:
    pABar = proportion_confint(nA, n, alpha=2 * alpha, method="beta")[0]
    radius = sigma * norm.ppf(pABar)
    return radius


def radii_discretion(radii: List[float], epsilons: Iterable = (0, 0.25, 0.5, 0.75, 1)):
    radii_tensor = torch.tensor(radii)
    denominator = len(radii)
    result = dict()
    for eps in epsilons:
        print("-" * 100)
        result[eps] = torch.sum(radii_tensor >= eps).item() / denominator
        print(f"certified robustness at {eps} is {result[eps]}")
        print("-" * 100)
    return result


def nA_and_n_to_radii(nAs: List, ns: List, *args, **kwargs):
    radii = []
    for nA, n in zip(nAs, ns):
        radii.append(robust_radius_given_correct_num(nA, n, *args, **kwargs))
    return radii


@torch.no_grad()
def certify_robustness(
    model: Smooth,
    loader: DataLoader or Iterable,
    epsilons: Iterable = (0, 0.25, 0.5, 0.75, 1),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    *args,
    **kwargs,
) -> Tuple[Dict, List, List, List]:
    model.base_classifier.to(device).eval()
    radii, nAs, ns = [], [], []
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        label, radius, nA, n = model.certify(x.squeeze(), *args, **kwargs)
        radii.append(radius if label == y.item() else __WRONG_PREDICTION__)
        nAs.append(nA if label == y.item() else 0)
        ns.append(n)
    result = radii_discretion(radii, epsilons)
    return result, nAs, ns, radii


def read_log_and_merge(
    log_dir: str = "./logs",
    epsilons: Iterable = (0, 0.25, 0.5, 0.75, 1),
    estimate_fold: Iterable = (1, 10),
    sigma=0.25,
) -> List[float]:
    nAs, ns = [], []
    if os.path.isdir(log_dir):
        log_path = os.listdir(log_dir)
        for i in log_path:
            with open(os.path.join(log_dir, i), "r") as f:
                now_log = eval(f.readline())
                nAs.extend(now_log[1])
                ns.extend(now_log[2])
    else:
        with open(log_dir, "r") as f:
            now_log = eval(f.readline())
            nAs.extend(now_log[1])
            ns.extend(now_log[2])
    for fold in estimate_fold:
        nAs = [i * fold for i in nAs]
        ns = [i * fold for i in ns]
        print(f"Certified Robustness estimated at n={ns[0]} is:")
        radii = nA_and_n_to_radii(nAs, ns, sigma=sigma)
        radii_discretion(radii, epsilons)
        radii = [i if i >= 0 else 0 for i in radii]
        print(f"Average radius is {sum(radii) / len(radii)}")
        print("*" * 50)
    print(f"Certified Robustness of folder {log_dir} processed. Total {len(nAs)} data.")
    return radii


@torch.no_grad()
def certify_robustness_via_lipschitz(
    model: nn.Module,
    loader: DataLoader or Iterable,
    logit_lipschitz: float,
    epsilons: Iterable = (0, 0.25, 0.5, 0.75, 1),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=False,
) -> Tuple[Dict, List, List, List]:
    model.to(device).eval().requires_grad_(False)
    radii, nAs, ns = [], [], []
    for x, y in tqdm(loader):
        x, y, B = x.to(device), y.to(device), y.numel()
        logit = model(x)
        negative = logit.clone()
        negative[torch.arange(B, device=device), y] = float("-inf")
        distance = logit[torch.arange(B, device=device), y] - torch.max(negative, dim=1)[0]
        radius = distance / (2 * logit_lipschitz)
        radius = torch.max(torch.stack([radius, torch.zeros_like(radius)]), dim=0)[0]
        if verbose:
            print(radius.item(), distance.item())
        radii.append(*radius.cpu().numpy().tolist())
    result = radii_discretion(radii, epsilons)
    return result, nAs, ns, radii
