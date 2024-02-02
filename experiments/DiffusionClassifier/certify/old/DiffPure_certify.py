import torch
from data import get_CIFAR10_test
from torchvision import transforms
from defenses.PurificationDefenses.DiffPure.DiffPure import DiffPureForRandomizedSmoothing
from tester.CertifyRobustness import certify_robustness
from defenses.RandomizedSmoothing.core import Smooth
from torch import nn
from transformers import AutoModelForImageClassification
from models.RobustBench.cifar10 import Wang2020Improving


# class ViTClassifier(nn.Module):
#     def __init__(self):
#         super(ViTClassifier, self).__init__()
#         self.transform = transforms.Resize((224, 224), antialias=True)
#         self.vit = AutoModelForImageClassification.from_pretrained(
#             "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10").cuda()
#
#     def forward(self, x):
#         x = self.transform(x)
#         return self.vit(x).logits


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

# DiffPure robustness certification

classifier = DiffPureForRandomizedSmoothing(mode='ddpm', model=Wang2020Improving())
# classifier.diffusion.stride = 1000
g = Smooth(classifier, batch_size=128)
print(certify_robustness(g, loader, n=1000))
