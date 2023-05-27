from pathlib import Path

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from e2e.callbacks.plotting import PredictionPlotting
from e2e.data.datamodule import ShapePredictionDataModule
from e2e.data.dataset import ShapeDataset
from e2e.models.cnn1d import CNN1D

NAS_DATA_DIR = Path("/Volumes/hornets/homes/mglasder/datasets/TrainingDev")
MAC_DATA_DIR = Path("/Users/magnus/datasets/WAAM/TrainingDev")

BATCH_SIZE = 16


def main():
    model = CNN1D(batch_size=BATCH_SIZE)
    logger = WandbLogger(project="waam-e2e-pre", log_model="all", log_every_n_steps=10)

    datamodule = ShapePredictionDataModule(
        batch_size=BATCH_SIZE,
        data_dir=MAC_DATA_DIR,
        dataset=ShapeDataset(),
    )

    callbacks = [PredictionPlotting()]

    trainer = Trainer(
        max_epochs=20,
        logger=logger,
        enable_checkpointing=False,
        accelerator="mps",
        callbacks=callbacks,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
