import random
from typing import Any, Optional

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from matplotlib import pyplot as plt


class PredictionPlotting(Callback):
    def __init__(self, epochs: list[int] = [], subset_size: Optional[int] = None):
        self.epochs = epochs
        self._subset_sz = subset_size

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int):
        final_epoch = trainer.max_epochs - 1
        if self._should_log_predictions(trainer.current_epoch, final_epoch):
            self._log_predictions(trainer, outputs, stage="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        final_epoch = trainer.max_epochs - 1
        if self._should_log_predictions(trainer.current_epoch, final_epoch):
            self._log_predictions(trainer, outputs, stage="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        final_epoch = trainer.max_epochs - 1
        # if self._should_log_predictions(trainer.current_epoch, final_epoch):
        self._log_predictions(trainer, outputs, stage="test")

    def _should_log_predictions(self, current_epoch: int, final_epoch: int) -> bool:
        return (current_epoch == final_epoch) or (current_epoch in self.epochs)

    def _log_predictions(self, trainer: Trainer, outputs: Any, stage: str) -> None:
        images, captions = [], []
        loss = self._convert_to_numpy(outputs["loss"])

        selected_indices = self._select_indices(len(outputs["preds"]), stage=stage)

        for idx in selected_indices:
            pred, trgt, inpt, id_ = (
                outputs["preds"][idx],
                outputs["targets"][idx],
                outputs["inputs"][idx],
                outputs["ids"][idx],
            )
            figure, caption = self._plot_example(pred, trgt, inpt, id_, loss, trainer.current_epoch, stage=stage)
            images.append(figure)
            captions.append(caption)
            plt.close()

        trainer.logger.log_image(
            key=f"{stage}/preds_plots",
            images=images,
            caption=captions,
        )

    def _select_indices(self, batch_size: int, stage: str) -> list[int]:
        if self._subset_sz is None or self._subset_sz >= batch_size or stage == "val":
            return list(range(batch_size))
        else:
            return random.sample(range(batch_size), self._subset_sz)

    def _plot_example(self, pred, trgt, inpt, id_, batch_loss, epoch, stage):
        pred_np, trgt_np, inpt_np = map(self._convert_to_numpy, [pred, trgt, inpt])

        fig, ax = plt.subplots()
        ax.plot(trgt_np[1, :], trgt_np[0, :], marker="s", markersize=4, color="blue", label="target (after)")
        ax.plot(pred_np[1, :], pred_np[0, :], marker="*", markersize=4, color="red", label="pred")
        ax.plot(inpt_np[1, :], inpt_np[0, :], marker=".", markersize=4, color="black", label="input (before)")

        caption = f"{stage} sample: {id_} - epoch: {epoch} \n batch_loss: {batch_loss:.4f}"
        ax.set_title(caption)
        # ax.aspect("equal")
        ax.legend()

        return fig, caption

    @staticmethod
    def _convert_to_numpy(data: torch.tensor) -> np.ndarray:
        """Move to cpu & convert to numpy."""
        return data.cpu().detach().numpy() if data.requires_grad else data.cpu().numpy()
