import torch
from torch import nn
from torchvision import transforms
from models.unets import get_NCSNPP

class UniformDiffusionSampler():
    def __init__(self,
                 unet: nn.Module = None,
                 img_shape=(3, 32, 32),
                 T=100,
                 mean=0.5,
                 std=0.5,
                 eps=16 / 255,
                 ):
        if unet is None:
            unet = get_NCSNPP()
        self.device = torch.device('cuda')
        self.unet = unet
        self.alpha_bar = torch.linspace(0, eps, T, device=self.device)
        self.T = T
        self.to_img = transforms.ToPILImage()
        self.i = 0
        self.unet.eval().requires_grad_(False).to(device=self.device)
        self.img_shape = img_shape
        self.state_size = img_shape[0] * img_shape[1] * img_shape[2]
        self.mean = mean
        self.std = std

    def convert(self, x):
        x = x * self.std + self.mean
        img = self.to_img(x[0])
        img.save(f'./what/{self.i}.png')
        self.i += 1
        return x

    def uniform_noise(self, shape):
        return (torch.rand(shape, device=self.device) - 0.5) * 2

    def sample(self, x):
        x = (x - self.mean) / self.std
        for t in range(self.T - 1, 0, -1):
            tensor_t = torch.zeros((x.shape[0]), device=self.device) + t
            predict = self.unet(x, tensor_t)[:, :3, :, :]
            x = x - self.alpha_bar[t].view(-1, 1, 1, 1) * predict
            if t > 1:
                noise = self.uniform_noise(x.shape) * self.alpha_bar[t].view(-1, 1, 1, 1)
            else:
                noise = 0
            x = x + noise
        return self.convert(x)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)
