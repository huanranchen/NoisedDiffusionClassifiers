from .diffpure import DiffusionPure
import torch


__all__ = ["DiffPureForRandomizedSmoothing", "CarliniDiffPureForRS", "XiaoDiffPureForRS"]


class DiffPureForRandomizedSmoothing(DiffusionPure):
    def __init__(self, *args, sigma=0.25, **kwargs):
        """
        explanation for computing t:
        Sigma need to be multiplied by 2 because RS add noise at [0, 1] while diffusion working at [-1, 1]
        Because alpha_t = 1 / (sigma ** 2 + 1), we can compute t by:
        alpha[t] >= 1 / (sigma ** 2 + 1) > alpha[t+1]
        """
        super(DiffPureForRandomizedSmoothing, self).__init__(*args, **kwargs)
        self.t = torch.max(self.diffusion.alpha_bar < 1 / ((sigma * 2) ** 2 + 1), dim=0)[1] - 1
        print(f"Diffusion timestep for sigma={sigma} is {self.t}")

    def forward(self, x, *args, **kwargs):
        x = self.pre_transforms(x)
        x = self.diffusion(x, *args, noise_level=(self.t + 1), scale=True, add_noise=False, **kwargs)
        x = self.post_transforms(x)
        x = self.model(x)
        return x


class XiaoDiffPureForRS(DiffusionPure):
    def __init__(self, *args, sigma=0.25, num_steps=10, ensemble_time=40, **kwargs):
        """
        Xiao et al. DensePure.
        Must be DDPM sampler! The reason of why we do not use SDE sampler is the standard discretization of SDE
        will cause the large error from DDPM when dt is large.
        """
        super(XiaoDiffPureForRS, self).__init__(*args, mode="ddpm", **kwargs)
        self.t = torch.max(self.diffusion.alpha_bar < 1 / ((sigma * 2) ** 2 + 1), dim=0)[1] - 1
        print(f"Diffusion timestep for sigma={sigma} is {self.t}")
        self.diffusion.stride = (self.t + 1) // num_steps
        print(f"Modifying DDPM sampler stride to {self.diffusion.stride} to achieve {num_steps} steps")
        self.ensemble_time = ensemble_time

    def forward(self, x, *args, **kwargs):
        B, C, H, D = x.shape
        x = x.view(B, 1, C, H, D).repeat(1, self.ensemble_time, 1, 1, 1).view(B * self.ensemble_time, C, H, D)
        x = self.pre_transforms(x)
        x = self.diffusion(x, *args, noise_level=(self.t + 1), scale=True, add_noise=False, **kwargs)
        x = self.post_transforms(x)
        x = self.model(x)
        num_classes = x.shape[1]
        x = x.view(B, self.ensemble_time, num_classes).mean(1)
        return x


class CarliniDiffPureForRS(DiffusionPure):
    def __init__(self, *args, sigma=0.25, **kwargs):
        super(CarliniDiffPureForRS, self).__init__(*args, **kwargs)
        self.t = torch.max(self.diffusion.alpha_bar < 1 / ((sigma * 2) ** 2 + 1), dim=0)[1] - 1
        print(f"Diffusion timestep for sigma={sigma} is {self.t}")

    def convert(self, x):
        x = (x + 1) * 0.5
        return torch.clamp(x, min=0, max=1)

    def one_step_denoise(self, x, *args, **kwargs):
        x = (x - 0.5) * 2
        noise_level = self.t + 1
        alpha_t = self.diffusion.alpha_bar[noise_level - 1]
        x_t = torch.sqrt(alpha_t) * x
        tensor_t = torch.zeros((x.shape[0],), device=self.device, dtype=torch.int) + noise_level
        epsilon = self.diffusion.unet(x_t, tensor_t)
        x0 = (x_t - torch.sqrt(1 - alpha_t) * epsilon) / torch.sqrt(alpha_t)
        return self.convert(x0)

    def forward(self, x, *args, **kwargs):
        x = self.pre_transforms(x)
        x = self.one_step_denoise(x, *args, **kwargs)
        x = self.post_transforms(x)
        x = self.model(x)
        return x
