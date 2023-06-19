import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim import Adam


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class CNN1D(LightningModule):
    def __init__(self, loss=F.mse_loss, seq_length=224, batch_size=16):
        super().__init__()
        self.save_hyperparameters()

        self.batch_sz = batch_size

        self.loss = loss
        self.length_in = seq_length
        length_out = seq_length
        k = 56  # kernel_size
        s = 8  # stride
        p = (k - s) // 2  # padding, k is odd
        p = 0
        out_sz1 = ((seq_length + 2 * p - k) // s) + 1
        # out_sz2 = ((out_sz1 + 2 * p - k) // s) + 1

        # TODO: finish model implementation
        self.model = nn.Sequential(
            # nn.ReLU(),
            Block(in_channels=1, out_channels=1, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            # Block(in_channels=16, out_channels=1, kernel_size=k, stride=s, padding=p),
            # nn.ReLU(),
            nn.BatchNorm1d(num_features=1),
            nn.Linear(in_features=out_sz1, out_features=224),
        )
        self.model.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.view(-1, 1, 224)
        x = self.model(x)
        return x.view(-1, 224)

    def training_step(self, batch, batch_idx):
        inputs, targets, ids = batch
        predictions = self(inputs)
        train_loss = self.loss(predictions, targets)
        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        return {"loss": train_loss, "preds": predictions, "targets": targets, "inputs": inputs, "ids": ids}

    def validation_step(self, batch, batch_idx):
        inputs, targets, ids = batch
        predictions = self(inputs)
        val_loss = self.loss(predictions, targets)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        return {"loss": val_loss, "preds": predictions, "targets": targets, "inputs": inputs, "ids": ids}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        inputs, targets, _ = batch
        preds = self(inputs)
        pred_loss = self.loss(preds, targets)
        return {"loss": pred_loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=0.01, weight_decay=0)
        return optimizer
