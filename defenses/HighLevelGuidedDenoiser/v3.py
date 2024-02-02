import torch
import numpy as np
from .inception import get_net, Conv, Bottleneck

config = dict()
config["flip"] = True
config["loss_idcs"] = [1]
config["net_type"] = "inceptionresnetv2"
input_size = [299, 299]
block = Conv
fwd_out = [64, 128, 256, 256, 256]
num_fwd = [2, 3, 3, 3, 3]
back_out = [64, 128, 256, 256]
num_back = [2, 3, 3, 3]
n = 1
hard_mining = 0
loss_norm = False


def get_model():
    net = get_net(input_size, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining, loss_norm)
    return config, net
