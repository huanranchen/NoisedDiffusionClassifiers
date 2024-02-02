import torch
from data import get_CIFAR10_test
from torchvision import transforms
import numpy as np
import random
from tester import test_acc
from defenses.PurificationDefenses.DiffPure.DiffusionClassifier import PredictXDotProductDiffusionClassifier

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

to_img = transforms.ToPILImage()
loader = get_CIFAR10_test(batch_size=1)
loader = [item for i, item in enumerate(loader) if 0 <= i < 256]
device = torch.device('cuda')
xs, ys = [], []
for x, y in loader:
    xs.append(x)
    ys.append(y)
x = torch.concat(xs, dim=0).cuda()
y = torch.concat(ys, dim=0).cuda()

dc = PredictXDotProductDiffusionClassifier()
test_acc(dc, loader)
