import lightning as L
import torch
import torchmetrics.classification
from torch import nn, optim


class LitMulticlassClassifier(L.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int, lr: float = 0.0):
        super().__init__()

        self.model = model
        self.lr = lr

        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, target = batch
        pred = self.model(x)
        loss = self.criterion(pred, target)
        return loss, pred, target

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, target = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_accuracy(pred, target)
        self.log(
            "train/accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, target = self.step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_accuracy(pred, target)
        self.log(
            "val/accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, target = self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_accuracy(pred, target)
        self.log(
            "test/accuracy",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(
        self,
    ) -> optim.Optimizer:
        return optim.Adam(self.model.parameters(), lr=self.lr)
