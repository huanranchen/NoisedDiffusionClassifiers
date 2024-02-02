import torch
from data import get_CIFAR10_test
from torchvision import transforms
from defenses.RandomizedSmoothing.core import Smooth
from tester.CertifyRobustness import certify_robustness
from defenses.PurificationDefenses.DiffPure.DiffusionClassifier import \
    DiffusionClassifierForRandomizedSmoothing
import argparse
from utils.seed import set_seed

set_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--begin', type=int)
parser.add_argument('--end', type=int)
args = parser.parse_args()
begin, end = args.begin, args.end

to_img = transforms.ToPILImage()
loader = get_CIFAR10_test(batch_size=1)
loader = [item for i, item in enumerate(loader) if begin <= i < end]
device = torch.device('cuda')
#
diffpure = DiffusionClassifierForRandomizedSmoothing()

diffpure.eval().requires_grad_(False).to(device)
# g = Smooth(diffpure, batch_size=1, verbose=True)
g = Smooth(diffpure, batch_size=32, verbose=False)
certify_robustness(g, loader)
# # print(g.predict(x[0]))
# x, y = loader[0]
# print(y[0])
# save_image(x[0], './debug/ori.png')
# print(g.certify(x[0].cuda()))
# test_acc(diffpure, loader)
