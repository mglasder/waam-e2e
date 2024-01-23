import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fp.helpers import timing


class Model(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss=F.mse_loss,
        in_len=224,
        batch_size=16,
        lr=0.001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.batch_sz = batch_size

        self.loss = loss
        self.lr = lr

        self.length_in = in_len
        self.length_out = 2

        self.model = model
        self.model.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    def forward(self, x):
        x = x.view(-1, 1, self.length_in)
        x = self.model(x)
        return x.view(-1, 1, self.length_out)

    def _step(self, inputs, targets):
        pred = self(inputs)
        loss = self._loss(pred, targets)
        return pred, loss

    def _loss(self, predictions, targets):
        return self.loss(predictions, targets)

    def training_step(self, batch, batch_idx):
        inputs, fp, ids = batch

        predictions, train_loss = self._step(inputs, fp)

        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        return {
            "loss": train_loss,
            "preds": predictions,
            "targets": fp,
            "inputs": inputs,
            "ids": ids,
            "footprint": fp,
        }

    def validation_step(self, batch, batch_idx):
        inputs, fp, ids = batch

        predictions, val_loss = self._step(inputs, fp)

        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        return {
            "loss": val_loss,
            "preds": predictions,
            "targets": fp,
            "inputs": inputs,
            "ids": ids,
            "footprint": fp,
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        inputs, fp, ids = batch
        predictions, pred_loss = self._step(inputs, fp)
        return {"loss": pred_loss, "preds": predictions, "targets": fp}

    @timing.time_it
    def predict_with_uncertainty(self, x, num_samples=30):
        self.model.train()  # Set the model to training mode to enable dropout
        with torch.no_grad():
            results = torch.zeros((num_samples,) + (x.shape[0], self.length_out))

            for i in range(num_samples):
                pred = self.forward(x)
                results[i, :] = pred.reshape(1, -1, self.length_out)

        self.model.eval()  # Set the model back to evaluation mode
        mean_prediction = results.mean(dim=0)
        prediction_std = results.std(dim=0)
        return mean_prediction, prediction_std

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5, verbose=True)

        if scheduler:
            # Every metric logged with log() or log_dict() in LightningModule
            # is a candidate for the monitor key.
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        else:
            return optimizer


class ModelV2(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss=F.mse_loss,
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

        self.length_in = in_len
        self.length_out = out_len

        self.model = model
        self.model.apply(self._init_weights)

        self.xs = self.xs = torch.arange(0, 90).to(self.device) / (90 - 1)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear or type(m) == nn.ConvTranspose1d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.view(-1, 1, self.length_in)
        x = self.model(x)
        return x.view(-1, 3)

    @staticmethod
    def polynomial3(a, b, c, d, xs):
        y = a * xs**3 + b * xs**2 + c * xs + d
        return y

    @staticmethod
    def polynomial4(a, b, c, d, e, xs):
        y = a * xs**4 + b * xs**3 + c * xs**2 + d * xs + e
        return y

    def _step(self, inputs, targets):
        left = inputs[:, 0][:, None]
        right = inputs[:, -1][:, None]
        m = right - left
        y_corr = self.xs.flipud().to(self.device) * m - right
        inputs_ = inputs + y_corr

        x = torch.concatenate((inputs_, m), dim=1)

        params = self(x)
        params_ = params.split(1, dim=1)
        a = params_[0]
        b = params_[1]
        c = params_[2]
        pred = self.polynomial4(a, b, c, -(a + b + c), 0, self.xs.to(self.device))
        out = pred - y_corr
        loss = self._loss(out, targets)
        return out, loss

    def training_step(self, batch, batch_idx):
        inputs, targets, ids, fp = batch

        predictions, train_loss = self._step(inputs, targets)

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

        predictions, val_loss = self._step(inputs, targets)

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

        predictions, pred_loss = self._step(inputs, targets)

        return {"loss": pred_loss, "preds": predictions, "targets": targets}

    @timing.time_it
    def predict_with_uncertainty(self, x, num_samples=30):
        self.model.train()  # Set the model to training mode to enable dropout
        with torch.no_grad():
            results = torch.zeros((num_samples,) + (x.shape[0], self.length_out))

            for i in range(num_samples):
                left = x[:, 0][:, None]
                right = x[:, -1][:, None]
                m = right - left
                y_corr = self.xs.flipud().to(self.device) * m - right
                x_ = x + y_corr

                input_ = torch.concatenate((x_, m), dim=1)

                params = self.forward(input_.clone())
                params_ = params.split(1, dim=1)
                a = params_[0]
                b = params_[1]
                c = params_[2]
                pred = self.polynomial4(a, b, c, -(a + b + c), 0, self.xs.to(self.device))
                results[i, :] = pred - y_corr

        self.model.eval()  # Set the model back to evaluation mode
        mean_prediction = results.mean(dim=0)
        prediction_std = results.std(dim=0)
        return mean_prediction, prediction_std

    def _loss(self, predictions, targets):
        loss = self.loss(predictions, targets)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)

        if scheduler:
            # Every metric logged with log() or log_dict() in LightningModule
            # is a candidate for the monitor key.
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        else:
            return optimizer
