import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from e2e.helpers import timing


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        # TODO: get this right
        second_derivative = outputs[:, :-2] - 2 * outputs[:, 1:-1] + outputs[:, 2:]
        smoothness_loss = torch.mean(second_derivative**2, dim=1)
        return torch.mean(smoothness_loss)


class SmoothnessLossMid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, fp_idx):
        # start_index, end_index = fp_idx[:, 0], fp_idx[:, 1]
        # smoothness_loss = torch.mean(
        #     (outputs[:, start_index : end_index - 1] - outputs[:, start_index + 1 : end_index]) ** 2
        # )
        # return smoothness_loss

        batch_size = outputs.size(0)
        max_len = outputs.size(1)

        # Create a range of indices
        idx = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1).to(outputs.device)

        # Create masks for start and end indices
        start_mask = idx >= fp_idx[:, 0:1]
        end_mask = idx < fp_idx[:, 1:2]
        mask = start_mask & end_mask

        # Apply mask to get the relevant outputs
        masked_outputs = outputs * mask.float()

        # Calculate second derivative
        second_derivative = masked_outputs[:, :-2] - 2 * masked_outputs[:, 1:-1] + masked_outputs[:, 2:]

        # Apply mask to second derivative
        second_derivative_masked = second_derivative * mask[:, :-2].float()

        # Calculate smoothness loss
        smoothness_loss = torch.mean(second_derivative_masked**2, dim=1)

        return torch.mean(smoothness_loss)


class SmoothnessLossPiecewise(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.n = n

    def forward(self, outputs, fp_idx):
        batch_size = outputs.size(0)
        max_len = outputs.size(1)

        # Create a range of indices
        idx = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1).to(outputs.device)

        # Create masks for start and end indices
        left_mask = idx < fp_idx[:, 0:1]
        middle_mask = (idx >= fp_idx[:, 0:1]) & (idx < fp_idx[:, 1:2])
        right_mask = idx >= fp_idx[:, 1:2]

        # Calculate second derivative
        second_derivative = outputs[:, : -2 * self.n] - 2 * outputs[:, self.n : -self.n] + outputs[:, 2 * self.n :]

        # Apply masks to second derivative
        second_derivative_left = second_derivative * left_mask[:, : -2 * self.n].float()
        second_derivative_middle = second_derivative * middle_mask[:, : -2 * self.n].float()
        second_derivative_right = second_derivative * right_mask[:, : -2 * self.n].float()

        # Calculate smoothness loss
        smoothness_loss_left = torch.mean(second_derivative_left**2, dim=1)
        smoothness_loss_middle = torch.mean(second_derivative_middle**2, dim=1)
        smoothness_loss_right = torch.mean(second_derivative_right**2, dim=1)

        # Combine the losses
        smoothness_loss = smoothness_loss_left + smoothness_loss_middle + smoothness_loss_right

        return torch.mean(smoothness_loss)


class AreaLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.1

    def forward(self, inputs, outputs, targets, fp_idx):
        batch_size = outputs.size(0)
        max_len = outputs.size(1)

        # Create a range of indices
        idx = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1).to(outputs.device)

        # Create masks for start and end indices
        start_mask = idx >= fp_idx[:, 0:1]
        end_mask = idx < fp_idx[:, 1:2]
        mask = start_mask & end_mask

        # Apply mask to get the relevant outputs and targets
        masked_outputs = outputs * mask.float()
        masked_targets = targets * mask.float()

        # Calculate areas
        area_outputs = torch.sum(masked_outputs, dim=1)
        area_targets = torch.sum(masked_targets, dim=1)

        # Calculate absolute difference in areas
        loss = torch.abs(area_outputs - area_targets)
        return torch.mean(loss) * self.scaling_factor


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
        # self.model.apply(self._init_weights)

        self._smoothness_loss = SmoothnessLoss()
        self._area_loss = AreaLoss()

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear or type(m) == nn.ConvTranspose1d:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.view(-1, 1, self.length_in)
        x = self.model(x)
        return x.view(-1, self.length_out)

    def training_step(self, batch, batch_idx):
        inputs, targets, ids, fp = batch

        diff = targets.detach() - inputs.detach()

        pred_diff = self(inputs)
        train_loss = self._loss(inputs, pred_diff, diff, fp)

        predictions = inputs + pred_diff

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
        diff = targets.detach() - inputs.detach()
        pred_diff = self(inputs)
        val_loss = self._loss(inputs, pred_diff, diff, fp)
        predictions = inputs + pred_diff
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
        diff = targets.detach() - inputs.detach()
        pred_diff = self(inputs)
        pred_loss = self._loss(inputs, pred_diff, diff, fp)
        predictions = inputs + pred_diff
        return {"loss": pred_loss, "preds": predictions, "targets": targets}

    @timing.time_it
    def predict_with_uncertainty(self, x, num_samples=30):
        self.model.train()  # Set the model to training mode to enable dropout
        with torch.no_grad():
            results = torch.zeros((num_samples,) + (x.shape[0], self.length_out))

            for i in range(num_samples):
                y_pred_diff = self.forward(x)
                results[i, :] = x + y_pred_diff

        self.model.eval()  # Set the model back to evaluation mode
        mean_prediction = results.mean(dim=0)
        prediction_std = results.std(dim=0)
        return mean_prediction, prediction_std

    def _loss(self, inputs, predictions, targets, fp):
        loss = self.loss(predictions, targets)
        fploss = self._footprint_loss(predictions, targets, fp)
        # smoothness_loss = self._smoothness(predictions.detach())
        # area_loss = self._area_loss(predictions, targets)
        # return (1.0 - self.lambda_ - self.gamma) * loss + self.lambda_ * smoothness_loss + self.gamma * area_loss

        fploss = self.theta * self._footprint_loss(predictions, targets, fp)
        sloss = self.lambda_ * self._smoothness_loss(predictions)
        aloss = self.gamma * self._area_loss(inputs, predictions, targets, fp)

        return (1 - self.theta - self.gamma - self.lambda_) * loss + fploss + sloss + aloss

    def _footprint_loss(self, preds, targets, fp):
        ldiff = preds[:, fp[0]] - targets[:, fp[1]]
        rdiff = preds[:, fp[0]] - targets[:, fp[1]]
        return torch.mean(ldiff**2 + rdiff**2)

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
