import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim import Adam


class CNN1D(LightningModule):
    def __init__(self, loss=F.mse_loss, seq_length=224, batch_size=16):
        super().__init__()
        self.save_hyperparameters()

        self.batch_sz = batch_size

        self.loss = loss
        self.length_in = seq_length
        length_out = seq_length
        k = 224  # kernel_size
        d = 1  # dilation
        s = 1  # stride
        p = (k - 1) // 2  # padding, k is odd

        # TODO: finish model implementation
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_features=1),
            nn.Conv1d(in_channels=1, out_channels=224, kernel_size=k, stride=s, dilation=d, padding=0),
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k, stride=s, dilation=d, padding=p),
            # nn.Conv1d(in_channels=1, out_channels=224, kernel_size=k, stride=s, dilation=d, padding=p),
            # nn.Conv1d(in_channels=224, out_channels=1, kernel_size=k, stride=s, dilation=d, padding=p),
        )
        # ...

    def forward(self, x):
        # TODO: view or transpose?
        x = self.model(x.view(-1, 1, 224))
        return x.view(-1, 224)

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        predictions = self(inputs)
        train_loss = self.loss(predictions, targets)
        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        return {"loss": train_loss, "preds": predictions, "targets": targets}

    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        predictions = self(inputs)
        val_loss = self.loss(predictions, targets)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        return {"loss": val_loss, "preds": predictions, "targets": targets}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        inputs, targets, _ = batch
        preds = self(inputs)
        pred_loss = self.loss(preds, targets)
        return {"loss": pred_loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=0.005, weight_decay=0)
        return optimizer
