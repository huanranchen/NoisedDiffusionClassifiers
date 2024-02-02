from data import get_imagenet_loader
from models import resnet50, BaseNormModel
from defenses.AdvTrain import ClassifierSolver
import torch
from optimizer import IdentityScheduler

loader = get_imagenet_loader(split="train", batch_size=128, augment=True, shuffle=True)
test_loader = get_imagenet_loader(split="val", batch_size=128)
cnn = BaseNormModel(resnet50()).cuda()
solver = ClassifierSolver(
    cnn,
    optimizer=lambda x: torch.optim.Adam(x.parameters(), lr=1e-3),
    scheduler=IdentityScheduler,
    writer_name="resnet50_noaug_shuffle",
)
solver.train(loader, 100, eval_loader=test_loader)
