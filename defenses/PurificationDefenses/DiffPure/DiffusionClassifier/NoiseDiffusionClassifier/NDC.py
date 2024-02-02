from ..EDMDC import EDMEulerIntegralDC
from torch import nn, Tensor
import torch

try:
    from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
except ImportError:
    print("warning: You haven't install torchcubicspline. You are not able to use LearnWeightEPNDC")

__all__ = [
    "CorrectAPNDC",
    "CorrectAPNDCCondY",
    "EPNDC",
    "APNDCEnsemble",
    "LearnWeightEPNDC",
]


class CorrectAPNDC(EDMEulerIntegralDC):
    def __init__(self, uncond_edm: nn.Module, cond_edm: nn.Module, sigma=0.25, steps=1001, *args, **kwargs):
        self.sigma = sigma * 2
        super(CorrectAPNDC, self).__init__(cond_edm, timesteps=torch.linspace(self.sigma, 3, steps), *args, **kwargs)
        self.uncond_edm = uncond_edm

    @torch.no_grad()
    def unet_loss_without_grad(
        self,
        x: Tensor,
        y: int or Tensor = None,
        batch_size=256,
        generator: torch.Generator = None,
    ) -> Tensor:
        """
        Calculate the diffusion loss
        x should be in range [0, 1]
        """
        x0 = (self.one_step_denoise(x, sigma=self.sigma) - 0.5) * 2
        result = 0
        count = 0
        y = torch.tensor([y], device=self.device) if type(y) is int else y
        all_i, all_i_plus_one, sigma_t = self.timesteps[1:-1], self.timesteps[2:], self.timesteps[0]
        for sigma_i, sigma_i_plus_one in zip(all_i.split(batch_size, dim=0), all_i_plus_one.split(batch_size, dim=0)):
            B = sigma_i.shape[0]
            weight = (
                (sigma_i_plus_one**2 + self.sigma_data**2)
                / (sigma_i_plus_one * self.sigma_data) ** 2
                * self.p_x[count : count + B]
                * self.dt[count : count + B]
            )
            count += B
            now_y = y.repeat(sigma_i_plus_one.numel()) if y is not None else None
            now_x = self.transform(x)
            now_x = now_x.repeat(sigma_i_plus_one.numel(), 1, 1, 1)
            noise = torch.randn(*now_x.shape, generator=generator, device=now_x.device, dtype=self.precision)
            noised_x = now_x + noise * torch.sqrt(sigma_i_plus_one**2 - sigma_t**2).view(-1, 1, 1, 1)
            pre = self.unet(noised_x, sigma_i_plus_one, now_y)
            loss = torch.sum(weight * torch.mean((pre - x0.repeat(pre.shape[0], 1, 1, 1)) ** 2, dim=[1, 2, 3]))
            result = result + loss
        return result

    def one_step_denoise(self, x: Tensor, normalize=True, sigma=0.5, y=None) -> Tensor:
        """
        x: In range (0, 1)
        """
        x = (x - 0.5) * 2 if normalize else x
        x0 = self.uncond_edm(x, torch.zeros((x.shape[0],), device=x.device, dtype=self.precision) + sigma, y)
        x0 = x0 / 2 + 0.5 if normalize else x0
        return x0


class CorrectAPNDCCondY(CorrectAPNDC):
    def __init__(self, *args, **kwargs):
        super(CorrectAPNDCCondY, self).__init__(*args, **kwargs)
        self.cond_edm = self.unet

    @torch.no_grad()
    def unet_loss_without_grad(
        self,
        x: Tensor,
        y: int or Tensor = None,
        batch_size=256,
        generator: torch.Generator = None,
    ) -> Tensor:
        """
        Calculate the diffusion loss
        x should be in range [0, 1]
        """
        result = 0
        count = 0
        y = torch.tensor([y], device=self.device) if type(y) is int else y
        x0 = (self.one_step_denoise(x, sigma=self.sigma, y=y) - 0.5) * 2
        all_i, all_i_plus_one, sigma_t = self.timesteps[1:-1], self.timesteps[2:], self.timesteps[0]
        for sigma_i, sigma_i_plus_one in zip(all_i.split(batch_size, dim=0), all_i_plus_one.split(batch_size, dim=0)):
            B = sigma_i.shape[0]
            weight = (
                (sigma_i_plus_one**2 + self.sigma_data**2)
                / (sigma_i_plus_one * self.sigma_data) ** 2
                * self.p_x[count : count + B]
                * self.dt[count : count + B]
            )
            count += B
            now_y = y.repeat(sigma_i_plus_one.numel()) if y is not None else None
            now_x = self.transform(x)
            now_x = now_x.repeat(sigma_i_plus_one.numel(), 1, 1, 1)
            noise = torch.randn(*now_x.shape, generator=generator, device=now_x.device, dtype=self.precision)
            noised_x = now_x + noise * torch.sqrt(sigma_i_plus_one**2 - sigma_t**2).view(-1, 1, 1, 1)
            pre = self.unet(noised_x, sigma_i_plus_one, now_y)
            loss = torch.sum(weight * torch.mean((pre - x0.repeat(pre.shape[0], 1, 1, 1)) ** 2, dim=[1, 2, 3]))
            result = result + loss
        return result

    def one_step_denoise(self, x: Tensor, normalize=True, sigma=0.5, y=None) -> Tensor:
        """
        x: In range (0, 1)
        """
        x = (x - 0.5) * 2 if normalize else x
        x0 = self.cond_edm(x, torch.zeros((x.shape[0],), device=x.device, dtype=self.precision) + sigma, y)
        x0 = x0 / 2 + 0.5 if normalize else x0
        return x0


class EPNDC(EDMEulerIntegralDC):
    """
    Exact Posterior Noised Diffusion Classifier
    """

    def __init__(self, cond_edm: nn.Module, sigma=0.25, steps=1001, weight=None, *args, **kwargs):
        self.sigma = sigma * 2
        super(EPNDC, self).__init__(cond_edm, timesteps=torch.linspace(self.sigma, 3, steps), *args, **kwargs)
        self.steps = steps
        self.eval().requires_grad_(False)
        self.weight = weight

    @torch.no_grad()
    def unet_loss_without_grad(
        self, x: Tensor, y: int or Tensor = None, batch_size=1001, generator: torch.Generator = None
    ) -> Tensor:
        """
        Calculate the diffusion loss
        x should be in range [0, 1]
        """
        result = 0
        count = 5
        y = torch.tensor([y], device=self.device) if type(y) is int else y
        all_i, all_i_plus_one, sigma_t = self.timesteps[count:-1], self.timesteps[count + 1 :], self.timesteps[0]
        for sigma_i, sigma_i_plus_one in zip(all_i.split(batch_size, dim=0), all_i_plus_one.split(batch_size, dim=0)):
            B = sigma_i.shape[0]
            weight = (
                (sigma_i**2 + self.sigma_data**2)
                / (sigma_i * self.sigma_data) ** 2
                * self.p_x[count : count + B]
                * self.dt[count : count + B]
            )
            weight = (
                weight
                * sigma_i_plus_one**2
                * (sigma_i_plus_one**2 - sigma_t**2)
                / 4
                / (sigma_i_plus_one - sigma_i) ** 2
                / (sigma_i**2 - sigma_t**2)
            )
            count += B
            now_y = y.repeat(sigma_i.numel()) if y is not None else None
            now_x = self.transform(x)
            now_x = now_x.repeat(sigma_i.numel(), 1, 1, 1)
            noise = torch.randn(*now_x.shape, generator=generator, device=now_x.device)
            noised_x = now_x + noise * torch.sqrt(sigma_i_plus_one**2 - sigma_t**2).view(-1, 1, 1, 1)
            pre = self.unet(noised_x, sigma_i_plus_one, now_y)
            p_mean = (
                (sigma_i_plus_one**2 - sigma_i**2).view(B, 1, 1, 1) * pre + sigma_i.view(B, 1, 1, 1) ** 2 * noised_x
            ) / sigma_i_plus_one.view(B, 1, 1, 1) ** 2
            q_mean = (
                (sigma_i_plus_one**2 - sigma_i**2).view(B, 1, 1, 1) * now_x
                + (sigma_i**2 - sigma_t**2).view(B, 1, 1, 1) * noised_x
            ) / (sigma_i_plus_one**2 - sigma_t**2).view(B, 1, 1, 1)
            loss = torch.mean((p_mean - q_mean) ** 2, dim=[1, 2, 3])
            loss = torch.sum(weight * loss)
            result = result + loss
        return result


class APNDCEnsemble(EDMEulerIntegralDC):
    """
    APNDC_ensemble.
    The noise is added from x0.
    """

    def __init__(self, uncond_edm: nn.Module, cond_edm: nn.Module, sigma=0.25, steps=1001, *args, **kwargs):
        self.sigma = sigma * 2
        super(APNDCEnsemble, self).__init__(cond_edm, timesteps=torch.linspace(self.sigma, 3, steps), *args, **kwargs)
        self.uncond_edm = uncond_edm

    def get_one_instance_prediction(self, x: Tensor) -> Tensor:
        """
        :param x: 1, C, H, D
        :return D
        """
        x = self.one_step_denoise(x, sigma=self.sigma)
        logit = super().get_one_instance_prediction(x)
        return logit

    def one_step_denoise(self, x: Tensor, normalize=True, sigma=0.5, y=None) -> Tensor:
        """
        x: In range (0, 1)
        """
        x = (x - 0.5) * 2 if normalize else x
        x0 = self.uncond_edm(x, torch.zeros((x.shape[0],), device=x.device, dtype=self.precision) + sigma, y)
        x0 = x0 / 2 + 0.5 if normalize else x0
        return x0


class LearnWeightEPNDC(EPNDC):
    def __init__(self, *args, num_params=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Parameter(torch.randn(num_params))
        self.params_t = self.timesteps[:: (self.timesteps.numel() // num_params)]
        # self.linear = nn.Parameter(torch.randn(self.steps - 2))

    @torch.no_grad()
    def unet_loss_without_grad(self, x: Tensor, y: int or Tensor = None, batch_size=1024) -> Tensor:
        """
        Calculate the diffusion loss
        x should be in range [0, 1]
        """
        result = []
        count = 0
        y = torch.tensor([y], device=self.device) if type(y) is int else y
        all_i, all_i_plus_one, sigma_t = self.timesteps[:-1], self.timesteps[1:], self.timesteps[0]
        for sigma_i, sigma_i_plus_one in zip(all_i.split(batch_size, dim=0), all_i_plus_one.split(batch_size, dim=0)):
            B = sigma_i.shape[0]
            weight = (
                (sigma_i**2 + self.sigma_data**2)
                / (sigma_i * self.sigma_data) ** 2
                * self.p_x[count : count + B]
                * self.dt[count : count + B]
            )
            weight = weight * (2 * sigma_i_plus_one**4) * sigma_i**2 / (sigma_i_plus_one**2 - sigma_i**2) ** 2
            count += B
            now_y = y.repeat(sigma_i.numel()) if y is not None else None
            now_x = self.transform(x)
            now_x = now_x.repeat(sigma_i.numel(), 1, 1, 1)
            noise = torch.randn_like(now_x)
            noised_x = now_x + noise * sigma_i_plus_one.view(-1, 1, 1, 1)
            pre = self.unet(noised_x, sigma_i_plus_one, now_y)
            p_mean = (
                (sigma_i_plus_one**2 - sigma_i**2).view(B, 1, 1, 1) * pre + sigma_i.view(B, 1, 1, 1) ** 2 * noised_x
            ) / sigma_i_plus_one.view(B, 1, 1, 1) ** 2
            q_mean = (
                (sigma_i_plus_one**2 - sigma_i**2).view(B, 1, 1, 1) * now_x
                + (sigma_i**2 - sigma_t**2).view(B, 1, 1, 1) * noised_x
            ) / (sigma_i_plus_one**2 - sigma_t**2).view(B, 1, 1, 1)
            loss = weight * torch.mean((p_mean - q_mean) ** 2, dim=[1, 2, 3])
            result.append(loss)
        result = torch.cat(result, dim=0)
        return result

    def get_one_instance_prediction(self, x: Tensor) -> Tensor:
        """
        :param x: 1, C, H, D
        :return D
        """
        loss = []
        for class_id in self.target_class:
            loss.append(self.unet_loss_without_grad(x, class_id))
        loss = torch.stack(loss)  # num_classes, T
        loss = loss * -1  # convert into logit where greatest is the target
        coeff = natural_cubic_spline_coeffs(self.params_t, self.linear.unsqueeze(1))
        spline = NaturalCubicSpline(coeff)
        weight = spline.evaluate(self.timesteps[:-1]).squeeze()
        loss = loss @ weight
        return loss

    def train(self, mode: bool = True) -> nn.Module:
        if hasattr(self, "linear"):
            self.linear.requires_grad_(mode)
        self.unet.eval().requires_grad_(False)
        return self
