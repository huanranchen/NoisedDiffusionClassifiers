import torch
from tester import test_acc
from data import get_CIFAR10_test, get_CIFAR10_train
from defenses.PurificationDefenses.DiffPure import SBGC
from models.unets import get_NCSNPP
from utils.ImageHandling import save_image, get_image

loader = get_CIFAR10_test(batch_size=1)
loader = [item for i, item in enumerate(loader) if i < 100]
unet = get_NCSNPP(grad_checkpoint=False)
unet.load_state_dict(torch.load("mle_ema_new.pt"), strict=False)
# unet.load_state_dict(torch.load('no_uncond_ema_2815.pt'))
# model = SBGC(unet=unet, mode='Skilling-Hutchinson trace estimator')
model = SBGC(unet=unet, mode="huanran approximator")

for cfg in range(11):
    print("-" * 100)
    print(f"now cfg = {cfg}")
    model.cfg = cfg
    test_acc(model, loader)
# model.cfg = 0
# # test_acc(model, loader)
# with torch.no_grad():
#     model.compute_likelihood(next(iter(loader))[0].cuda())
# # latent = model.compute_likelihood(x)[2]
# #     reverse = model.sample(latent)
# # print(torch.sum((reverse - x) ** 2))
# # save_image(x, 'ori.png')
# # save_image(reverse, 'reverse.png')
