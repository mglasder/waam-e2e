from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from e2e.callbacks.plotting import PredictionPlotting
from e2e.data.datamodule import ShapePredictionDataModule
from e2e.data.dataset import ShapeDataset
from e2e.models.cnn1d import CNN1D

NAS_DATA_DIR = Path("/Volumes/hornets/homes/mglasder/datasets/TrainingDev")
MAC_DATA_DIR = Path("/Users/magnus/datasets/WAAM/TrainingDev")

BATCH_SIZE = 16
MAX_EPOCHS = 200
N_WORKERS = 4

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


def main():
    model = CNN1D(batch_size=BATCH_SIZE)
    logger = WandbLogger(project="waam-e2e-pre", log_model="all")

    datamodule = ShapePredictionDataModule(
        batch_size=BATCH_SIZE,
        data_dir=MAC_DATA_DIR,
        workers=N_WORKERS,
        dataset=ShapeDataset(),
    )

    callbacks = [PredictionPlotting(epochs=[], subset_size=5)]

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        enable_checkpointing=False,
        accelerator="mps",
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
