import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from e2e.helpers import timing


class ModelV2(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        device,
        loss=F.mse_loss,
        in_len=224,
        out_len=100,
        n_outputs=5,
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
        self.n_outputs = n_outputs

        self.model = model
        self.model.apply(self._init_weights)

        self.xs = torch.arange(0, 90).to(self.device) / (90 - 1)
        self.ts = torch.linspace(0, 1, 90, dtype=torch.float32)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear or type(m) == nn.ConvTranspose1d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.view(-1, 1, self.length_in)
        x = self.model(x)
        return x.view(-1, self.n_outputs)

    @staticmethod
    def polynomial3(a, b, c, d, xs):
        y = a * xs**3 + b * xs**2 + c * xs + d
        return y

    @staticmethod
    def polynomial4(a, b, c, d, e, xs):
        y = a * xs**4 + b * xs**3 + c * xs**2 + d * xs + e
        return y

    @staticmethod
    def _hermite(p0, p1, m0, m1, ts):
        t3 = ts**3
        t2 = ts**2
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + ts
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    def _step(self, inputs, targets, fp):
        left = inputs[:, 0][:, None]
        right = inputs[:, -1][:, None]
        m = right - left
        y_corr = self.xs.flipud().to(self.device) * m - right
        inputs_ = inputs + y_corr

        width = torch.abs(fp[:, 1] - fp[:, 0]).unsqueeze(1) * 0.1

        x = torch.concatenate((inputs_, m / width), dim=1)

        params = self(x)
        params_ = params.split(1, dim=1)

        m0 = params_[0]  # .relu() + 0.1
        m1 = params_[1]  # .relu() + 0.1)

        ts = self.ts.to(self.device)
        pred = self._hermite(0, 0, m0, m1, ts)

        out = pred - y_corr
        return out

    def training_step(self, batch, batch_idx):
        inputs, targets, ids, fp = batch

        predictions = self._step(inputs, targets, fp)
        train_loss = self._loss(predictions, targets)

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

        predictions = self._step(inputs, targets, fp)
        val_loss = self._loss(predictions, targets)

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

        predictions = self._step(inputs, targets, fp)
        pred_loss = self._loss(predictions, targets)

        return {"loss": pred_loss, "preds": predictions, "targets": targets}

    @timing.time_it
    def predict_with_uncertainty(self, x, fp=None, num_samples=30, reduction="mean"):
        self.model.train()  # Set the model to training mode to enable dropout
        with torch.no_grad():
            results = torch.zeros((num_samples,) + (x.shape[0], self.length_out))

            for i in range(num_samples):
                out = self._step(inputs=x, targets=None, fp=fp)
                results[i, :] = out

        self.model.eval()  # Set the model back to evaluation mode

        if reduction == "mean":
            mean_prediction = results.mean(dim=0)
            prediction_std = results.std(dim=0)
            return mean_prediction, prediction_std
        else:
            return results

    def _loss(self, predictions, targets):
        loss = self.loss(predictions, targets)
        # diff = torch.abs(predictions - targets)
        # area_loss = (diff.sum(dim=1) * 0.1 - 14.6)**2

        return loss  # + area_loss.mean()

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=True)
        if scheduler:
            # Every metric logged with log() or log_dict() in LightningModule
            # is a candidate for the monitor key.
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        else:
            return optimizer
