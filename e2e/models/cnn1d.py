import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim import Adam


class CNN1D(LightningModule):
    def __init__(self, loss=F.mse_loss):
        super().__init__()
        self.save_hyperparameters()

        self.loss = loss

        # TODO: finish model implementation
        self.model = nn.Conv1d(in_channels=224, out_channels=224, kernel_size=24, stride=1, padding=0)
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
        optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
