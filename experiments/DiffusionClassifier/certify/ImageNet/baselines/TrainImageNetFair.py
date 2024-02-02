from data import get_imagenet_loader
from models import resnet50, BaseNormModel
from defenses.AdvTrain import ClassifierSolver
import torch
from optimizer import IdentityScheduler
from torchvision import transforms

loader = get_imagenet_loader(split="train", batch_size=128, augment=True, shuffle=True)
test_loader = get_imagenet_loader(split="val", batch_size=128)
fair_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
cnn = BaseNormModel(resnet50(pretrained=False), transform=fair_transform).cuda()
solver = ClassifierSolver(
    cnn,
    optimizer=lambda x: torch.optim.Adam(x.parameters(), lr=1e-4),
    scheduler=IdentityScheduler,
    writer_name="resnet50_fair_scratch",
)
solver.train(loader, 100, eval_loader=test_loader)
