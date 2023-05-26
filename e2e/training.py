from pathlib import Path

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from e2e.datamodules.height_seq import ShapeDataset, ShapePredictionDataModule
from e2e.models.cnn1d import CNN1D

NAS_DATA_DIR = Path("/Volumes/hornets/homes/mglasder/datasets/TrainingDev")
MAC_DATA_DIR = Path("/Users/magnus/datasets/WAAM/TrainingDev")

BATCH_SIZE = 16


def main():
    model = CNN1D(batch_size=BATCH_SIZE)
    logger = WandbLogger(project="waam-e2e-pre", log_model=False)
    logger = False

    datamodule = ShapePredictionDataModule(batch_size=BATCH_SIZE, data_dir=MAC_DATA_DIR, dataset=ShapeDataset())

    trainer = Trainer(max_epochs=50, logger=logger, enable_checkpointing=False, gpu=False)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
