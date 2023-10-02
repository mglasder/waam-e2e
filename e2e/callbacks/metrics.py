import numpy as np
import wandb
from lightning import Callback
from scipy.spatial.distance import cdist


class FootprintAvgAbsValErrorLogger(Callback):
    def __init__(self, footprint_is_absolute=True):
        """Average absolute error of footprint prediction on validation data set."""
        self.footprint_is_absolute = footprint_is_absolute

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        preds = outputs["preds"]
        targets = outputs["targets"]
        fp_idx = outputs["footprint"]

        if self.footprint_is_absolute:
            left_errors = preds[:, 0] - targets[:, 0]
            right_errors = preds[:, 1] - targets[:, 1]

        else:
            left_errors = preds[:, fp_idx[0]] - targets[:, fp_idx[1]]
            right_errors = preds[:, fp_idx[0]] - targets[:, fp_idx[1]]

        trainer.logger.log_metrics({"val/footprint_error_left": left_errors.abs().mean()})
        trainer.logger.log_metrics({"val/footprint_error_right": right_errors.abs().mean()})


class ModHausdorffLogger(Callback):
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        preds = outputs["preds"].detach().cpu().numpy()
        targets = outputs["targets"].detach().cpu().numpy()
        fp_idx = outputs["footprint"].detach().cpu().numpy()

        # this is not really correct as the resolution is not 0.1 anymore
        x = np.arange(0, preds.shape[1] / 10, 0.1)

        # median of min distances
        modhaussdorff = []
        for b in range(outputs["preds"].shape[0]):
            left = fp_idx[b, 0]
            right = fp_idx[b, 1]
            dist = cdist(list(zip(x, targets[b, left:right])), list(zip(x, preds[b, left:right])))
            modhaussdorff.append((np.median(np.min(dist, axis=1))))

        trainer.logger.log_metrics({"val/mod_haussdorff": float(np.mean(modhaussdorff))})


class LogModelParametersAndGradients(Callback):
    def on_after_backward(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            trainer.logger.experiment.log({f"Gradients/{name}": wandb.Histogram(param.grad)})
            trainer.logger.experiment.log({f"Parameters/{name}": wandb.Histogram(param.data)})
