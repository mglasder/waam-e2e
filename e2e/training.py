from lightning import Trainer

from e2e.datamodules.height_seq import ShapeDataset, ShapePredictionDataModule
from e2e.models.cnn1d import CNN1D


def main():
    model = CNN1D()
    datamodule = ShapePredictionDataModule(dataset=ShapeDataset())

    trainer = Trainer()
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
