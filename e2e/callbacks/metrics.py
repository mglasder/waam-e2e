import numpy as np
from lightning import Callback
from scipy.spatial.distance import cdist


class FootprintAvgAbsValErrorLogger(Callback):
    """Average absolute error of footprint prediction on validation data set."""

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
        x = np.arange(0, len(preds) / 10, 0.1)

        # median of min distances
        modhaussdorff = []
        for b in range(outputs["preds"].shape[0]):
            left = fp_idx[b, 0]
            right = fp_idx[b, 1]
            dist = cdist(list(zip(x, targets[b, left:right])), list(zip(x, preds[b, left:right])))
            modhaussdorff.append((np.median(np.min(dist, axis=1))))

        trainer.logger.log_metrics({"val/mod_haussdorff": float(np.mean(modhaussdorff))})
