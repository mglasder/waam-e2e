import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from e2e.helpers import timing


class ModelPoints(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss=F.mse_loss,
        in_len=90,
        out_len=90,
        batch_size=64,
        lr=0.001,
        gamma=0.25,
        target_area=11.4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.batch_sz = batch_size

        self.loss = loss
        self.lr = lr

        self.target_area = target_area
        self.gamma = gamma

        self.length_in = in_len
        self.length_out = out_len

        self.model = model
        # self.model.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) is nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.model(x)
        return x

    def _step(self, inputs, targets, fp):
        """
        Defines pre- and post processing of model input for train, val, test and predict steps,
        and for predict_with_uncertainty.
        """
        pred = self(inputs)
        return pred

    def training_step(self, batch, batch_idx):
        inputs, targets, ids, fp = batch

        predictions = self._step(inputs, targets, fp)
        area = self._calculate_area(inputs, predictions)
        train_loss = (1 - self.gamma) * self._loss(predictions, targets) + self.gamma * (
            (area - self.target_area) ** 2
        ).mean()

        self.log("train_area", area.mean(), prog_bar=False, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_sz)
        return {
            "loss": train_loss,
            "preds": predictions,
            "targets": targets,
            "inputs": inputs,
            "ids": ids,
            "footprint": fp,
        }

    def _calculate_area(self, inpt_batch: torch.tensor, trgt_batch: torch.tensor) -> torch.tensor:
        """
        :param inpt_batch: torch.tensor of shape (batch_size, 2, length_out) representing the input batch.
        :param trgt_batch: torch.tensor of shape (batch_size, 2, length_out) representing the target batch.
        :return: torch.tensor of shape (batch_size,) representing the calculated area for each sample in the batch.

        This method calculates the area of a closed curve defined by the input and target batches using Green's theorem.
        The input batch and target batch should have the same shape. The shape of the input and target
        batch should be (batch_size, 2, length_out), where length_out is the length of the curve.

        The method first concatenates the input and target batches along the last dimension to obtain a closed curve
        tensor. Then, it extracts the X and Z coordinates of the curve. The X_next and Z_next coordinates are obtained
        by rolling the X and Z tensors along the second dimension. The integral is then calculated using the formula:
        X_next * Z - Z_next * X. The area is obtained by taking the absolute value of the integral and multiplying it
        by 0.5.

        The resulting area tensor has a shape of (batch_size,) and represents the calculated area for each sample in the batch.

        Example usage:

        inpt = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
        trgt = torch.tensor([[[7, 8, 9], [10, 11, 12]]])
        area = _calculate_area(inpt, trgt)
        print(area)  # Output: tensor([ 4., 15.])

        """
        assert inpt_batch.shape == trgt_batch.shape and trgt_batch.shape[1:] == (2, self.length_out)

        closed_curve = torch.cat([inpt_batch, torch.flip(trgt_batch, dims=(2,))], dim=2)
        X = closed_curve[:, 1, :]
        Z = closed_curve[:, 0, :]
        X_next = torch.roll(X, shifts=(0, -1), dims=(0, 1))
        Z_next = torch.roll(Z, shifts=(0, -1), dims=(0, 1))
        integral = torch.sum(X_next * Z - Z_next * X, dim=1)
        area = 0.5 * torch.abs(integral)
        return area

    def validation_step(self, batch, batch_idx):
        inputs, targets, ids, fp = batch

        predictions = self._step(inputs, targets, fp)
        area = self._calculate_area(inputs, predictions)
        val_loss = (1 - self.gamma) * self._loss(predictions, targets) + self.gamma * (
            (area - self.target_area) ** 2
        ).mean()

        self.log("val_area", area.mean(), prog_bar=False, on_epoch=True, on_step=False, batch_size=self.batch_sz)
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
        """
        :param predictions: A tensor containing the predicted values. Its shape should be (batch_size, 2, length_out).
        :param targets: A tensor containing the target values. Its shape should be (batch_size, 2, length_out).
        :return: A scalar tensor representing the batch mean distance between the predictions and targets.

        This method calculates the Euclidean distance between the predictions and targets for each sample in the batch.
        It then computes the mean distance over the last dimension (individual points) and finally calculates the mean
        distance over the batch dimension.
        The resulting batch mean distance is returned as a scalar tensor.
        """
        assert predictions.shape == targets.shape and predictions.shape[1:] == (2, self.length_out)

        # compute the Euclidean distances
        distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
        # compute the mean over the last dimension (individual points)
        mean_distances = torch.mean(distances, dim=1)
        # compute the mean over the batch dimension
        batch_mean_distance = torch.mean(mean_distances)

        return batch_mean_distance

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
