import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim import Adam


class Model(LightningModule):
    def __init__(self, model: nn.Module, loss=F.mse_loss, lambda_=0.0001, gamma=0.01, seq_length=224, batch_size=16):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.batch_sz = batch_size

        self.loss = loss
        self.lambda_ = lambda_  # controls smoothness penalty
        self.gamma = gamma  # controls area penalty

        self.length_in = seq_length

        self.model = model
        self.model.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear or type(m) == nn.ConvTranspose1d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.view(-1, 1, 224)
        x = self.model(x)
        return x.view(-1, 224)

    def training_step(self, batch, batch_idx):
        inputs, targets, ids = batch
        predictions = self(inputs)
        train_loss = self._loss(inputs, predictions, targets)
        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        return {"loss": train_loss, "preds": predictions, "targets": targets, "inputs": inputs, "ids": ids}

    def validation_step(self, batch, batch_idx):
        inputs, targets, ids = batch
        predictions = self(inputs)
        val_loss = self._loss(inputs, predictions, targets)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        return {"loss": val_loss, "preds": predictions, "targets": targets, "inputs": inputs, "ids": ids}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        inputs, targets, _ = batch
        predictions = self(inputs)
        pred_loss = self._loss(inputs, predictions, targets)
        return {"loss": pred_loss, "preds": predictions, "targets": targets}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=0.005, weight_decay=0)
        return optimizer

    def _loss(self, inputs, predictions, targets):
        loss = self.loss(predictions, targets)
        smoothness_loss = self._smoothness(predictions.detach())
        area_loss = self._area_loss(inputs.cpu(), predictions.detach().cpu(), targets.cpu())
        return (1.0 - self.lambda_ - self.gamma) * loss + self.lambda_ * smoothness_loss + self.gamma * area_loss

    @staticmethod
    def _smoothness(preds):
        dy = preds[:, 1:] - preds[:, :-1]
        d2y = dy[:, 1:] - dy[:, :-1]
        return torch.mean(torch.sum(d2y**2)) / 1_000

    @staticmethod
    def _area_loss(inputs, targets, predictions):
        target_diff = torch.sum(targets - inputs, axis=1)
        pred_diff = torch.sum(predictions - inputs, axis=1)
        return F.mse_loss(pred_diff, target_diff) / 10_000
