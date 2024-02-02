import torch
from data import get_CIFAR10_test, get_CIFAR10_train
from defenses.PurificationDefenses.DiffPure import SBGC, EDM2VP, EDMEulerIntegralDC
from models.unets import get_edm_cifar_uncond
from matplotlib import pylab as plt
from tqdm import tqdm


loader = get_CIFAR10_test(batch_size=1)
unet = get_edm_cifar_uncond(use_fp16=True)
sbgc = SBGC(unet=EDM2VP(unet), mode="Skilling-Hutchinson trace estimator")
dc = EDMEulerIntegralDC(
    unet,
    timesteps=torch.linspace(1e-4, 3, 12600),
    mode="ELBO"
)
dc.share_noise = True
sbgc.eval().requires_grad_(False).cuda()
dc.eval().requires_grad_(False).cuda()

elbos, log_likelihoods = [], []
for i, (x, y) in enumerate(tqdm(loader)):
    if i > 128:
        break
    x, y = x.cuda(), y.cuda()
    log_likelihood = sbgc.compute_likelihood(x)[1].item()
    elbo = dc.unet_loss_without_grad(x).item() * 3072
    print(log_likelihood, elbo, f"{((elbo-log_likelihood) / log_likelihood) * 100}%")
    elbos.append(elbo)
    log_likelihoods.append(log_likelihood)

plt.plot(list(range(len(elbos))), elbos)
plt.plot(list(range(len(log_likelihoods))), log_likelihoods)
plt.legend(['ELBO', "log likelihood"])
plt.savefig("./elbo.png")
