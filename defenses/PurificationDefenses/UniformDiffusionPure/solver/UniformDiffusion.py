import torch
from torch import nn, Tensor
from tqdm import tqdm
from torchvision import transforms
import random
from math import isnan


class UniformDiffusionSolver():
    def __init__(self,
                 unet: nn.Module,
                 # beta=torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda')),
                 device=torch.device('cuda'),
                 T=100,
                 eps=16 / 255):
        self.device = device
        self.unet = unet
        # self.beta = beta
        # alpha = (1 - beta)
        # self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.alpha_bar = torch.linspace(0, eps, T, device=self.device)
        self.T = T

        # training schedule
        self.unet_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=2e-4)

        self.init()
        self.transform = lambda x: (x - 0.5) * 2

    def init(self):
        # init
        self.unet.eval().requires_grad_(False).to(self.device)

    def uniform_noise(self, shape):
        return (torch.rand(shape, device=self.device) - 0.5) * 2

    def train(self, train_loader, total_epoch=1000,
              p_uncondition=1,
              fp16=False,):
        self.unet.train()
        self.unet.requires_grad_(True)
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        for epoch in range(1, total_epoch + 1):
            epoch_loss = 0
            pbar = tqdm(train_loader)
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.cuda(), y.cuda()
                # some preprocess
                x = self.transform(x)
                # train
                x, y = x.to(self.device), y.to(self.device)
                t = torch.randint(self.T, (x.shape[0],), device=self.device)
                tensor_t = t
                noise = self.uniform_noise(x.shape)
                noised_x = x + self.alpha_bar[t].view(-1, 1, 1, 1) * noise
                target = noise
                if fp16:
                    with autocast():
                        if random.random() < p_uncondition:
                            pre = self.unet(noised_x, tensor_t)[:, :3, :, :]
                        else:
                            pre = self.unet(noised_x, tensor_t, y)[:, :3, :, :]
                        loss = self.unet_criterion(pre, target)
                else:
                    if random.random() < p_uncondition:
                        pre = self.unet(noised_x, tensor_t)[:, :3, :, :]
                    else:
                        pre = self.unet(noised_x, tensor_t, y)[:, :3, :, :]
                    loss = self.unet_criterion(pre, target)
                self.optimizer.zero_grad()
                if fp16:
                    raise NotImplementedError
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_value_(self.unet.parameters(), 0.1)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                epoch_loss += loss.item()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'step {step}, loss {epoch_loss / step}')
            print(f'epoch {epoch}, loss {epoch_loss / len(train_loader)}')
            if not isnan(epoch_loss):
                torch.save(self.unet.state_dict(), 'unet.pt')

        self.init()
