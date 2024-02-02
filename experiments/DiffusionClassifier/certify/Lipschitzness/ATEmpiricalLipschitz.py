from models import Wang2023Better
from optimizer import GradientNormMaximizer
from data import get_CIFAR10_test
import torch


def cw_loss(x, y):
    p = x[:, y]
    n = x.clone()
    n[:, y] -= float("inf")
    n = torch.max(n, dim=1)[0]
    loss = p - n
    return loss.sum()


model = Wang2023Better(pretrained=True).eval().requires_grad_(False).cuda()
loader = get_CIFAR10_test(batch_size=1)
in_lr, out_lr = 0.01, 1 / 255
iterations = 100
eps = 1e-6
for x, y in loader:
    x, y = x.cuda(), y.cuda()
    print("ground truth is:", y)
    ori_x = x.clone()
    x.requires_grad_(True)
    inner_optim = torch.optim.SGD([x], lr=in_lr)
    outer_optim = torch.optim.Adam([x], lr=out_lr)
    optim = GradientNormMaximizer([x], inner_optim, outer_optim, lr=in_lr)
    for _ in range(iterations):
        # Gradient norm maximizer
        # pre = model(x)
        # loss = cw_loss(pre, y)
        # optim.zero_grad()
        # loss.backward()
        # print(torch.norm(x.grad, p=2).item(), torch.norm(x.grad, p="fro").item(), loss.item())
        # print(model(x))
        # optim.first_step()
        # pre = model(x)
        # loss = cw_loss(pre, y)
        # optim.zero_grad()
        # loss.backward()
        # optim.second_step()
        # with torch.no_grad():
        #     torch.clamp_(x, min=0, max=1)
        #     torch.clamp_(x, min=ori_x-8/255, max=ori_x+8/255)

        # During optimization
        pre = model(x)
        loss = cw_loss(pre, y)
        optim.zero_grad()
        loss.backward()
        outer_optim.step()
        mask = (x > 0 + eps) * (x < 1 - eps) * (torch.abs(x - ori_x) < 8 / 255 - eps)
        print(
            loss.item(),
            torch.norm(x.grad).item(),
            torch.norm(x.grad[mask]).item(),
            torch.norm(x - ori_x).item(),
            torch.sum(mask),
        )
        print(model(x))
        with torch.no_grad():
            torch.clamp_(x, min=0, max=1)
            torch.clamp_(x, min=ori_x - 8 / 255, max=ori_x + 8 / 255)
