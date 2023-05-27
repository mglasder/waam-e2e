from typing import Any

from lightning import Callback, LightningModule, Trainer
from matplotlib import pyplot as plt


class PredictionPlotting(Callback):
    def __init__(self, epochs: list[int] = []):
        self.epochs = epochs

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        if (trainer.current_epoch == trainer.max_epochs - 1) or (trainer.current_epoch in self.epochs):
            images = []
            captions = []
            loss = outputs["loss"].cpu().detach().numpy()
            for p, t, i, id_ in zip(outputs["preds"], outputs["targets"], outputs["inputs"], outputs["ids"]):
                fig, caption = self._plot_example(p, t, i, id_, loss, trainer.current_epoch, "train")
                images.append(fig)
                captions.append(caption)
                plt.close()

            trainer.logger.log_image(
                key="train/preds_plots",
                images=images,
                caption=captions,
            )

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if (trainer.current_epoch == trainer.max_epochs - 1) or (trainer.current_epoch in self.epochs):
            images = []
            captions = []
            loss = outputs["loss"].cpu().detach().numpy()
            for p, t, i, id_ in zip(outputs["preds"], outputs["targets"], outputs["inputs"], outputs["ids"]):
                fig, caption = self._plot_example(p, t, i, id_, loss, trainer.current_epoch, "val")
                images.append(fig)
                captions.append(caption)
                plt.close()

            trainer.logger.log_image(
                key="val/preds_plots",
                images=images,
                caption=captions,
            )

    @staticmethod
    def _plot_example(pred, trgt, inpt, id_, batch_loss, epoch, stage):
        fig, ax = plt.subplots()
        ax.plot(pred.cpu().detach().numpy(), color="red", label="pred")
        ax.plot(trgt.cpu().numpy(), color="blue", label="target (after)")
        ax.plot(inpt.cpu().detach().numpy(), color="black", label="input (before)")
        caption = f"{stage} sample: {id_} - epoch: {epoch} \n batch_loss: {batch_loss:.4f}"
        ax.set_title(caption)
        ax.legend()
        return fig, caption
