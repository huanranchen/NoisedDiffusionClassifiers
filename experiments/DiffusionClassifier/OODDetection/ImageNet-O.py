from data.ImageNetAO import get_imagenet_o_loader, calibration_tools
import torch
import numpy as np
from defenses.PurificationDefenses.DiffPure.DiffusionClassifier.DiffusionClassifier import DiffusionClassifier
from models.unets import get_guided_diffusion_unet
from tqdm import tqdm
import argparse
from models import FixNoiseModel


parser = argparse.ArgumentParser()
parser.add_argument("--begin", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--steps", type=int, default=50)
args = parser.parse_args()
begin, end, steps = args.begin, args.end, args.steps
# # TODO:注意这个step并不是总时间个数，而是跳跃的step
#
# noe_loader, val_loader_imagenet_o, imagenet_o_mask = get_imagenet_o_loader(mean=(0, 0, 0), std=(1, 1, 1))
#
# noe_loader = [item for i, item in enumerate(noe_loader) if begin <= i < end]
# val_loader_imagenet_o = [item for i, item in enumerate(val_loader_imagenet_o) if begin <= i < end]
#
# model = DiffusionClassifier(
#     unet=get_guided_diffusion_unet(resolution=256, cond=False),
#     beta=torch.linspace(0.1 / 1000, 20 / 1000, 1000),
#     ts=torch.arange(0, 1000, step=steps),
# )
# model = FixNoiseModel(model)
# model.eval().requires_grad_(False).cuda()
# # TODO: OOD检测应该使用FixNoise，避免对不同样本不公平。
#
# @torch.no_grad()
# def get_confidence(net, loader):
#     confidence = []
#     with torch.no_grad():
#         for data, target in tqdm(loader):
#             data, target = data.cuda(), target.cuda()
#             output = net.m.unet_loss_without_grad(data)
#             # print(net.m.unet_loss_without_grad(data))
#             # print(net.m.unet_loss_without_grad(data))
#             # print(net.m.unet_loss_without_grad(data))
#             # print(net.m.unet_loss_without_grad(data))
#             confidence.append(output.item())
#     return np.array(confidence)
#
#
# print(get_confidence(model, val_loader_imagenet_o))
# print(get_confidence(model, noe_loader))
with open("imagenet-o.txt") as f:
    # x = f.read().split("[")
    # in_d = x[1].split()
    # in_d[-1] = in_d[-1][:-1]
    # in_d = np.array([float(i) for i in in_d])
    # out_d = x[2].split()
    # out_d[-1] = out_d[-1][:-1]
    # out_d = np.array([float(i) for i in out_d])
    print(calibration_tools.get_measures(np.random.randn(512), np.random.randn(512)))
