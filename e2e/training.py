from pathlib import Path

from lightning import Trainer

from e2e.datamodules.height_seq import ShapeDataset, ShapePredictionDataModule
from e2e.models.cnn1d import CNN1D

DATA_DIR = Path("/Volumes/hornets/homes/mglasder/datasets/TrainingDev")


def main():
    model = CNN1D()

    datamodule = ShapePredictionDataModule(data_dir=DATA_DIR, dataset=ShapeDataset())

    trainer = Trainer(max_epochs=10, logger=False)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
