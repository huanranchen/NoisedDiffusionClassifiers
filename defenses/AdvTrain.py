from attacks import AdversarialInputAttacker, IdentityAttacker
import torch
import os
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from collections import OrderedDict
from tqdm import tqdm
from typing import Callable, List
from optimizer import default_optimizer, default_lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tester import test_robustness
from utils.plot import Landscape4Model

__all__ = ["AdversarialTraining", "ClassifierSolver"]


class AdversarialTraining:
    def __init__(
        self,
        attacker: AdversarialInputAttacker,
        model: nn.Module,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        optimizer: Callable = default_optimizer,
        scheduler: Callable = default_lr_scheduler,
        writer_name: str = None,
        pre_processor: Callable = lambda *args: args
    ):
        self.attacker = attacker
        self.student = model
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer(self.student)
        if writer_name is not None:
            self.writer_name = writer_name
            self.init(writer_name)
        self.scheduler = scheduler(self.optimizer)
        self.pre_processor = pre_processor

    def init(self, name: str):
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")
        self.writer = SummaryWriter(f"./runs/{name}")

    def train_an_epoch(
        self,
        loader: DataLoader,
        eval_loader: DataLoader,
        test_attacker: AdversarialInputAttacker,
        fp16: bool,
        scaler: GradScaler,
        epoch: int,
        save_advs: bool = True,
    ) -> OrderedDict:
        result = OrderedDict()
        result["advs"] = []
        train_loss = 0
        train_acc = 0
        pbar = tqdm(loader)
        self.student.requires_grad_(True).train()
        for step, (x, y) in enumerate(pbar, 1):
            x, y = self.pre_processor(x, y)
            x, y = x.to(self.device), y.to(self.device)
            adv_x = self.attacker(x.clone(), y)
            if save_advs:
                result["advs"].append((adv_x.cpu(), y.cpu()))
            x = adv_x
            if fp16:
                with autocast():
                    student_out = self.student(x)  # N, 60
                    _, pre = torch.max(student_out, dim=1)
                    loss = self.criterion(student_out, y)
            else:
                student_out = self.student(x)  # N, 60
                _, pre = torch.max(student_out, dim=1)
                loss = self.criterion(student_out, y)

            train_acc += (torch.sum(pre == y).item()) / y.shape[0]
            train_loss += loss.item()
            self.optimizer.zero_grad()

            if fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                self.optimizer.step()
            if step % 10 == 0:
                pbar.set_postfix_str(f"loss={train_loss / step}, acc={train_acc / step}")

        train_loss /= len(loader)
        train_acc /= len(loader)

        self.scheduler.step(train_loss)

        print(f"epoch {epoch}, loss = {train_loss}, acc = {train_acc}")
        torch.save(self.student.state_dict(), f"./checkpoints/student_{self.writer_name}.pth")
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("acc/train", train_acc, epoch)
        self.writer.add_scalar("hyper/lr", self.optimizer.param_groups[0]["lr"], epoch)
        result["epoch_info"] = {"train_loss": train_loss, "train_acc": train_acc}

        if eval_loader is not None:
            self.student.eval().requires_grad_(False)
            if test_attacker is None:
                test_attacker = self.attacker
            valid_acc = test_robustness(test_attacker, eval_loader, [self.student])
            valid_acc = valid_acc[0]
            self.writer.add_scalar("acc/eval", valid_acc, epoch)
            result["epoch_info"]["eval_acc"] = valid_acc

        # return values
        result["model"] = self.student
        return result

    @torch.no_grad()
    def draw_landscape(self, advs: List[Tensor], model: nn.Module):
        def get_loss(m: nn.Module) -> float:
            loss = 0
            for x, y in advs:
                loss += self.criterion(m(x.to(self.device)), y.to(self.device))
            loss /= len(advs)
            return loss

        drawer = Landscape4Model(model, get_loss)
        drawer.synthesize_coordinates()
        drawer.draw()

    def train(
        self,
        loader: DataLoader,
        total_epoch: int = 2000,
        fp16: bool = False,
        eval_loader: DataLoader = None,
        test_attacker: AdversarialInputAttacker = None,
        draw_landscape: bool = False,
    ):
        scaler = GradScaler()
        for epoch in range(1, total_epoch + 1):
            results = self.train_an_epoch(loader, eval_loader, test_attacker, fp16, scaler, epoch, draw_landscape)
            if draw_landscape:
                self.draw_landscape(results["advs"], results["model"])


class ClassifierSolver(AdversarialTraining):
    def __init__(self, *args, **kwargs):
        super().__init__(IdentityAttacker(), *args, **kwargs)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
