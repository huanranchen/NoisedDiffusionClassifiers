from .DiffusionClassifier import RobustDiffusionClassifier
import torch
from torch import nn
from ..model import get_unet
from ..sampler import DDIM
from .EDMDC import EDMEulerIntegralDC, EDMGaussQuadratureDC
from models.unets.EDM import EDMPrecond

__all__ = ["DiffusionClassifierForRandomizedSmoothing", "EDMEulerIntegralClassifierForRandomizedSmoothing"]


class DiffusionClassifierForRandomizedSmoothingShift(nn.Module):
    """
    这是有问题的，你这样的话，第二次噪声基本和RS噪声直接正交，预测肯定不准，大家一起偏了45°。建议好好推导
    """

    def __init__(self, sigma=0.25, *args, **kwargs):
        """
        explanation for computing t:
        Sigma need to be multiplied by 2 because RS add noise at [0, 1] while diffusion working at [-1, 1]
        Because alpha_t = 1 / (sigma ** 2 + 1), we can compute t by:
        alpha[t] >= 1 / (sigma ** 2 + 1) > alpha[t+1]
        """
        super(DiffusionClassifierForRandomizedSmoothingShift, self).__init__()
        self.rdc = RobustDiffusionClassifier(*args, **kwargs)
        self.dc = self.rdc.diffusion_classifier
        self.device = self.dc.device
        self.t = torch.max(self.dc.alpha_bar < 1 / ((sigma * 2) ** 2 + 1), dim=0)[1] - 1
        self.dc.ts = torch.arange(start=self.t + 1, end=self.dc.T, step=1).to(self.device)
        """
        Need to change alpha, cause the input is in p_t(x)
        """
        self.dc.alpha_bar = self.dc.alpha_bar / self.dc.alpha_bar[self.t]

        self._init()

    def _init(self):
        print(f"Randomized Smoothing for Robust diffusion classifier use noise scale {self.t}")

    def forward(self, x, *args, **kwargs):
        x = self.rdc(x, *args, **kwargs)
        return x


class DiffusionClassifierForRandomizedSmoothing(nn.Module):
    """
    单步去噪，然后再diffusion classifier
    """

    def __init__(self, sigma=0.25, *args, **kwargs):
        """
        explanation for computing t:
        Sigma need to be multiplied by 2 because RS add noise at [0, 1] while diffusion working at [-1, 1]
        Because alpha_t = 1 / (sigma ** 2 + 1), we can compute t by:
        alpha[t] >= 1 / (sigma ** 2 + 1) > alpha[t+1]
        """
        super(DiffusionClassifierForRandomizedSmoothing, self).__init__()
        self.rdc = RobustDiffusionClassifier(*args, **kwargs)
        self.dc = self.rdc.diffusion_classifier
        self.device = self.dc.device
        self.alpha_bar = self.dc.alpha_bar
        self.t = torch.max(self.alpha_bar < 1 / ((sigma * 2) ** 2 + 1), dim=0)[1] - 1
        self.unet = self.dc.unet
        self.purify_unet = get_unet()[0]
        self.purifier = DDIM(self.purify_unet)
        self._init()
        self.i = 0

    def _init(self):
        self.eval().requires_grad_(False)
        self.dc.ts = torch.arange(start=self.t + 1, end=self.dc.T, step=1, device=self.device)
        # self.dc.ts = torch.cat([self.dc.ts] * 2, dim=0)
        print(f"Randomized Smoothing for Robust diffusion classifier use noise scale {self.t}")

    def _one_step_denoise(self, x):
        x = (x - 0.5) * 2
        x = self.alpha_bar[self.t] * x
        # tensor_t = torch.zeros((x.shape[0],), device=self.device) + self.t
        # predict = self.purify_unet(x, tensor_t)[:, :3, :, :]
        # x0 = (x - torch.sqrt(1 - self.alpha_bar[self.t]) * predict) / torch.sqrt(self.alpha_bar[self.t])
        # x0 = x0 / 2 + 0.5
        x0 = self.purifier.purify(x, add_noise=False, noise_level=self.t, normalize=False)
        # save_image(x0[0], f'./debug/{self.i}.png')
        # self.i += 1
        return x0

    def forward(self, x, *args, **kwargs):
        denoised_x = self._one_step_denoise(x)
        x = self.rdc(denoised_x, *args, **kwargs)
        return x


class EDMEulerIntegralClassifierForRandomizedSmoothing(nn.Module):
    """
    单步去噪，然后再EDM diffusion classifier
    """

    def __init__(self, sigma=0.25):
        self.sigma = sigma * 2
        super(EDMEulerIntegralClassifierForRandomizedSmoothing, self).__init__()
        network_kwargs = dict(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
            augment_dim=9,
        )
        self.cond_edm = EDMPrecond(img_resolution=32, img_channels=3, label_dim=10, **network_kwargs)
        self.dc = EDMEulerIntegralDC(self.cond_edm, timesteps=torch.linspace(self.sigma, 3, 1001))
        self.uncond_edm = EDMPrecond(img_resolution=32, img_channels=3, label_dim=0, **network_kwargs)
        self._init()
        self.i = 0

    def _init(self):
        self.cond_edm.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_cond.pt"))
        self.uncond_edm.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
        self.eval().requires_grad_(False)
        print(f"EDM classifier (RS) use noise scale {self.sigma} (in range [-1, 1])")

    def _one_step_denoise(self, x):
        """
        x: In range (0, 1)
        """
        x = (x - 0.5) * 2
        # x = x + torch.randn_like(x) * self.sigma
        x0 = self.uncond_edm(x, sigma=torch.zeros((x.shape[0],), device=x.device) + self.sigma)
        x0 = x0 / 2 + 0.5
        return x0

    def forward(self, x, *args, **kwargs):
        denoised_x = self._one_step_denoise(x)
        x = self.dc(denoised_x, *args, **kwargs)
        return x


class EDMGaussQuadratureClassifierForRandomizedSmoothing(nn.Module):
    """
    单步去噪，然后再EDM diffusion classifier
    """

    def __init__(self, sigma=0.25):
        self.sigma = sigma * 2
        super(EDMGaussQuadratureClassifierForRandomizedSmoothing, self).__init__()
        network_kwargs = dict(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
            augment_dim=9,
        )
        self.cond_edm = EDMPrecond(img_resolution=32, img_channels=3, label_dim=10, **network_kwargs)
        self.dc = EDMGaussQuadratureDC(self.cond_edm, t0=self.sigma)
        self.uncond_edm = EDMPrecond(img_resolution=32, img_channels=3, label_dim=0, **network_kwargs)
        self._init()
        self.i = 0

    def _init(self):
        self.cond_edm.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_cond.pt"))
        self.uncond_edm.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
        self.eval().requires_grad_(False)
        print(f"EDM classifier (RS) use noise scale {self.sigma} (in range [-1, 1])")

    def _one_step_denoise(self, x):
        """
        x: In range (0, 1)
        """
        x = (x - 0.5) * 2
        # x = x + torch.randn_like(x) * self.sigma
        x0 = self.uncond_edm(x, sigma=torch.zeros((x.shape[0],), device=x.device) + self.sigma)
        x0 = x0 / 2 + 0.5
        return x0

    def forward(self, x, *args, **kwargs):
        denoised_x = self._one_step_denoise(x)
        x = self.dc(denoised_x, *args, **kwargs)
        return x
