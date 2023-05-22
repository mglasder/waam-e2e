from lightning import LightningModule
from torch import nn
from torch.optim import Adam


class CNN1D(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # TODO: finish model implementation
        self.model = nn.Conv1d(in_channels=100, out_channels=100)
        # ...

    def forward(self, x):
        x = self(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        predictions = self(inputs)
        train_loss = self.lossfunc(predictions, targets)
        return {"loss": train_loss, "preds": predictions, "targets": targets}

    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        predictions = self(inputs)
        val_loss = self.lossfunc(predictions, targets)
        return {"loss": val_loss, "preds": predictions, "targets": targets}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        imgs, targets, _ = batch
        preds = self(imgs)
        pred_loss = self.lossfunc(preds, targets)
        return {"loss": pred_loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
