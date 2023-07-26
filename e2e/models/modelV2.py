import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim import Adam


class ModelV2(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss=F.mse_loss,
        theta=0.1,
        lambda_=0.01,
        gamma=0.01,
        in_len=224,
        out_len=100,
        batch_size=16,
        lr=0.001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.batch_sz = batch_size

        self.loss = loss
        self.lr = lr
        self.lambda_ = lambda_  # controls smoothness penalty
        self.gamma = gamma  # controls area penalty
        self.theta = theta  # controls footprint penalty

        self.length_in = in_len
        self.length_out = out_len

        self.model = model
        self.model.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear or type(m) == nn.ConvTranspose1d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.view(-1, 1, self.length_in)
        x = self.model(x)
        return x.view(-1, self.length_out)

    def training_step(self, batch, batch_idx):
        inputs, targets, ids, fp = batch
        predictions = self(inputs)
        train_loss = self._loss(inputs, predictions, targets, fp)
        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        return {
            "loss": train_loss,
            "preds": predictions,
            "targets": targets,
            "inputs": inputs,
            "ids": ids,
            "footprint": fp,
        }

    def validation_step(self, batch, batch_idx):
        inputs, targets, ids, fp = batch
        predictions = self(inputs)
        val_loss = self._loss(inputs, predictions, targets, fp)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        return {
            "loss": val_loss,
            "preds": predictions,
            "targets": targets,
            "inputs": inputs,
            "ids": ids,
            "footprint": fp,
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        inputs, targets, ids, fp = batch
        predictions = self(inputs)
        pred_loss = self._loss(inputs, predictions, targets, fp)
        return {"loss": pred_loss, "preds": predictions, "targets": targets}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=0)
        return optimizer

    def _loss(self, inputs, predictions, targets, fp):
        loss = self.loss(predictions, targets)
        footprint_loss = self._footprint_loss(predictions, targets, fp)
        # smoothness_loss = self._smoothness(predictions.detach())
        # area_loss = self._area_loss(predictions, targets)
        # return (1.0 - self.lambda_ - self.gamma) * loss + self.lambda_ * smoothness_loss + self.gamma * area_loss
        return (1 - self.theta - self.gamma) * loss + self.theta * footprint_loss  # + self.gamma * area_loss

    def _footprint_loss(self, preds, targets, fp):
        ldiff = preds[:, fp[0]] - targets[:, fp[1]]
        rdiff = preds[:, fp[0]] - targets[:, fp[1]]
        return torch.mean(ldiff**2 + rdiff**2)

    @staticmethod
    def _smoothness(preds):
        dy = preds[:, 1:] - preds[:, :-1]
        d2y = dy[:, 1:] - dy[:, :-1]
        return torch.mean(torch.sum(d2y**2)) / 100

    @staticmethod
    def _area_loss(inputs, targets, predictions):
        # calculate are between inputs and predictions
        diff_true = targets - inputs
        x = torch.linspace(0, (119 * 0.1), 119)

        diff_pred = predictions - inputs
        area_true = torch.trapz(diff_true, x)
        area_pred = torch.trapz(diff_pred, x)
        return torch.mean(torch.abs(area_true - area_pred))
