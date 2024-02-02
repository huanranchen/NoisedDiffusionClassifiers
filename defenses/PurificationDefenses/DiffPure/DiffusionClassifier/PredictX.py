import torch
from torch import Tensor
from .DiffusionClassifier import DiffusionClassifier

__all__ = ['PredictXDiffusionClassifier', 'PredictXDotProductDiffusionClassifier']


class PredictXDiffusionClassifier(DiffusionClassifier):
    def __init__(self, *args, **kwargs):
        super(PredictXDiffusionClassifier, self).__init__(*args, **kwargs)
        self.unet.load_state_dict(torch.load('./ema_new.pt'))
        self.weight = self.alpha_bar / (1 - self.alpha_bar)

    @torch.no_grad()
    def unet_loss_without_grad(self,
                               x: Tensor,
                               y: int or Tensor = None,
                               ) -> Tensor:
        """
        :param x: in range [0, 1]
        """
        # diffusion training loss
        t = self.ts
        y = torch.tensor([y], device=self.device) if type(y) is int else y
        y = y.repeat(t.numel()) if y is not None else None
        x = x.repeat(t.numel(), 1, 1, 1)
        x = self.transform(x)
        tensor_t = t
        noise = torch.randn_like(x)
        noised_x = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * x + \
                   torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * noise
        pre = self.unet(noised_x, tensor_t, y)[:, :3, :, :]
        x0 = (noised_x - torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * pre
              ) / torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        # loss = self.unet_criterion(x0, x)
        loss = self.weight * torch.mean((x0 - x) ** 2, dim=[1, 2, 3])  # N @ N
        loss = torch.mean(loss)
        return loss


class PredictXDotProductDiffusionClassifier(DiffusionClassifier):
    def __init__(self, *args, **kwargs):
        super(PredictXDotProductDiffusionClassifier, self).__init__(*args, **kwargs)
        self.unet.load_state_dict(torch.load('./ema_new.pt'))
        self.weight = self.alpha_bar / (1 - self.alpha_bar)

    @torch.no_grad()
    def unet_loss_without_grad(self,
                               x: Tensor,
                               y: int or Tensor = None,
                               ) -> Tensor:
        """
        :param x: in range [0, 1]
        """
        # diffusion training loss
        t = self.ts
        y = torch.tensor([y], device=self.device) if type(y) is int else y
        y = y.repeat(t.numel()) if y is not None else None
        x = x.repeat(t.numel(), 1, 1, 1)
        x = self.transform(x)
        tensor_t = t
        noise = torch.randn_like(x)
        noised_x = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * x + \
                   torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * noise
        pre = self.unet(noised_x, tensor_t, y)[:, :3, :, :]
        x0 = (noised_x - torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * pre
              ) / torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)  # N, C, H, D
        # loss = torch.mean(x0 ** 2, dim=[1, 2, 3]) - 2 * torch.mean((x0 * x).view(t.numel(), -1), dim=-1)  # h(x0)^T x
        # loss = - 2 * torch.mean((x0 * x).view(t.numel(), -1), dim=-1)
        loss = torch.mean(x0 ** 2, dim=[1, 2, 3])
        loss = torch.mean(self.weight * loss)
        return loss
