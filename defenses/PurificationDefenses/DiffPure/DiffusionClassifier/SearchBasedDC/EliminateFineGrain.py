import torch
import math
from torch import Tensor
from pprint import pprint
from typing import List
from ..EDMDC import EDMEulerIntegralDC

__inf__ = 999


class EDMEulerIntegralEliminateFineGrainDC(EDMEulerIntegralDC):
    def __init__(self, *args, eliminate_ratio=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.eliminate_ratio = eliminate_ratio

    def get_one_instance_prediction(self, x: Tensor) -> Tensor:
        """
        :param x: 1, C, H, D
        :return D
        """
        logit = self.elimination_and_fine_grain_algorithm(x) * -1
        return logit

    def unet_loss_given_timesteps_and_candidate(self, x: Tensor, t: Tensor, y: Tensor, batch_size=32) -> Tensor:
        """
        :param x: Should be (1, C, H, D)
        :param t: (T, )
        :param y: (K, )
        :param batch_size: batch size
        :param generator: for control randomness.
        return loss (K, ). Averaged on T.
        """
        x = self.transform(x)
        K, T, C, H, D = y.shape[0], t.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # timesteps
        timesteps = t.repeat(K)  # K*T
        # x, y and dt
        # labels = y.view(-1, 1).repeat(1, T).view(-1)  # K*T
        labels = y.view(-1, 1).expand(-1, T).reshape(-1)  # K*T
        # x = x.repeat(K * T, 1, 1, 1)
        x = x.expand(K * T, -1, -1, -1)
        # noises
        # noises = torch.randn(T, C, H, D).repeat(K, 1, 1, 1) if self.share_noise else torch.randn(T * K, C, H, D)
        noises = torch.randn(1, T, C, H, D).expand(K, -1, -1, -1, -1).reshape(K * T, C, H, D)
        dt = self.dt[0]  # this is euler integral. dt is same.
        # dt = 0.1316
        result = []
        for sigma, now_x, now_y, noise in zip(
            timesteps.split(batch_size, dim=0),
            x.split(batch_size, dim=0),
            labels.split(batch_size, dim=0),
            noises.split(batch_size, dim=0),
        ):
            noise = noise.to(self.device)
            p_t = (
                1
                / (sigma * self.P_std * math.sqrt(2 * math.pi))
                * torch.exp(-((torch.log(sigma) - self.P_mean) ** 2) / 2 * self.P_std**2)
            )
            weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2 * p_t * dt
            noised_x = now_x + noise * sigma.view(-1, 1, 1, 1)
            pre = self.unet(noised_x, sigma, now_y)
            loss = weight * torch.mean((pre - now_x) ** 2, dim=[1, 2, 3])
            result.append(loss)
        result = torch.cat(result, dim=0)  # K*T
        result = result.view(K, T).mean(1)
        return result

    def dynamic_elimination_algorithm(self, x: Tensor):
        candidates = set(self.target_class)
        diffusion_losses = torch.zeros((len(candidates),), device=self.device)
        count = 0
        for now_timestep in self.timesteps.split(1, dim=0):  # TODO：T应该按重要性排
            pprint(candidates, compact=True, width=150)  # TODO：试试建模一下t的不确定性。假设检验
            if len(candidates) == 1:
                break
            count += 1
            y = torch.tensor(list(candidates), device=self.device)
            loss = self.unet_loss_given_timesteps_and_candidate(x, now_timestep, y)
            diffusion_losses[y] += loss
            loss = diffusion_losses[y]  # calculate total loss for abandon
            # print(loss)
            # indices = torch.sort(loss)[1]  # from small loss to large loss
            # abandoned boundary
            # abandoned_boundary = math.ceil(indices.numel() / 2)
            # abandoned_boundary = indices.numel() - 1
            # abandoned = indices[abandoned_boundary:]  # ceiling
            abandoned = (loss - torch.min(loss)) > 0.0005  # TODO: 这里threshold应该随着t累积
            # For abandoned, loss += inf, remove from candidate
            abandoned_y = y[abandoned]
            diffusion_losses[abandoned_y] += __inf__
            candidates = candidates - set(abandoned_y.cpu().numpy().tolist())
        print(count)
        return diffusion_losses

    def elimination_and_fine_grain_algorithm(self, x: Tensor):
        candidates = set(self.target_class)
        diffusion_losses = torch.zeros((len(candidates),), device=self.device)
        # elimination
        timesteps = torch.linspace(1e-4, 3, 50, device=self.device)
        for now_timestep in timesteps.split(1, dim=0):  # TODO：T应该按重要性排
            # pprint(candidates, compact=True, width=150)  # TODO：试试建模一下t的不确定性。假设检验
            if len(candidates) <= 10:
                break
            y = torch.tensor(list(candidates), device=self.device)
            loss = self.unet_loss_given_timesteps_and_candidate(x, now_timestep, y)
            diffusion_losses[y] += loss
            loss = diffusion_losses[y]  # calculate total loss for abandon
            # print(loss)
            # indices = torch.sort(loss)[1]  # from small loss to large loss
            # abandoned boundary
            # abandoned_boundary = math.ceil(indices.numel() / 2)
            # abandoned_boundary = indices.numel() - 1
            # abandoned = indices[abandoned_boundary:]  # ceiling
            # abandoned = ((loss - torch.min(loss)) / torch.min(loss) > 0.05) | ((loss - torch.min(loss)) > 0.0005)
            abandoned = (loss - torch.min(loss)) > 0.0005
            # For abandoned, loss += inf, remove from candidate
            abandoned_y = y[abandoned]
            diffusion_losses[abandoned_y] += __inf__
            candidates = candidates - set(abandoned_y.cpu().numpy().tolist())
        # print(loss)
        # fine grain:
        printed_information = list(candidates)
        printed_information.sort()
        pprint(printed_information, compact=True, width=150)
        timesteps = torch.linspace(1e-4, 3, 126, device=self.device)
        y = torch.tensor(list(candidates), device=self.device)
        abandoned_y = set(self.target_class) - candidates
        abandoned_y = torch.tensor(list(abandoned_y), device=self.device)
        loss = self.unet_loss_given_timesteps_and_candidate(x, timesteps, y)
        diffusion_losses = torch.zeros_like(diffusion_losses)
        diffusion_losses[abandoned_y] += __inf__
        diffusion_losses[y] = loss
        return diffusion_losses


class EDMEulerIntegralEliminator(EDMEulerIntegralEliminateFineGrainDC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> List:
        return self.get_one_instance_prediction(x)

    def get_one_instance_prediction(self, x: Tensor) -> List:
        return self.eliminate(x)

    def eliminate(self, x: Tensor) -> List:
        candidates = set(self.target_class)
        diffusion_losses = torch.zeros((len(candidates),), device=self.device)
        # elimination
        timesteps = torch.linspace(1e-4, 3, 50, device=self.device)
        for now_timestep in timesteps.split(1, dim=0):  # TODO：T应该按重要性排
            # pprint(candidates, compact=True, width=150)  # TODO：试试建模一下t的不确定性。假设检验
            if len(candidates) <= 10:
                break
            y = torch.tensor(list(candidates), device=self.device)
            loss = self.unet_loss_given_timesteps_and_candidate(x, now_timestep, y)
            diffusion_losses[y] += loss
            loss = diffusion_losses[y]  # calculate total loss for abandon
            # print(loss)
            # indices = torch.sort(loss)[1]  # from small loss to large loss
            # abandoned boundary
            # abandoned_boundary = math.ceil(indices.numel() / 2)
            # abandoned_boundary = indices.numel() - 1
            # abandoned = indices[abandoned_boundary:]  # ceiling
            # abandoned = ((loss - torch.min(loss)) / torch.min(loss) > 0.05) | ((loss - torch.min(loss)) > 0.0005)
            abandoned = (loss - torch.min(loss)) > 0.0005
            # For abandoned, loss += inf, remove from candidate
            abandoned_y = y[abandoned]
            diffusion_losses[abandoned_y] += __inf__
            candidates = candidates - set(abandoned_y.cpu().numpy().tolist())
        if len(candidates) > 10:
            candidates = torch.sort(diffusion_losses, dim=0)[1].cpu().numpy().tolist()[:10]
        return list(candidates)
