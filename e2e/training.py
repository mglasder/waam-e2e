from pathlib import Path

import torch
from lightning import Trainer

from e2e.callbacks.plotting import PredictionPlotting
from e2e.data.datamodule import ShapePredictionDataModule
from e2e.data.dataset import ShapeDataset
from e2e.models.model import Model
from e2e.models.resnet import ResNet1D, ResNetConfig

NAS_DATA_DIR_DEV = Path("/Volumes/hornets/homes/mglasder/datasets/TrainingDev")
MAC_DATA_DIR_DEV = Path("/Users/magnus/datasets/WAAM/TrainingDev")
VM_DATA_DIR = Path("/home/magnus/datasets/waam/30_processing_results/ImageGenerator")
VM_DATA_DIR_DEV = Path("/home/magnus/datasets/waam/TrainingDev")

BATCH_SIZE = 16
MAX_EPOCHS = 10
N_WORKERS = 1
DEVICE = "cpu"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


def main():
    # unet = UNet1D(enconf=EncoderConfig(), deconf=DecoderConfig())
    # unet.to(DEVICE)

    resnet = ResNet1D(conf=ResNetConfig())

    model = Model(model=resnet, batch_size=BATCH_SIZE)

    # logger = WandbLogger(project="waam-e2e-pre", log_model="all")
    logger = None

    datamodule = ShapePredictionDataModule(
        batch_size=BATCH_SIZE,
        data_dir=MAC_DATA_DIR_DEV,
        workers=N_WORKERS,
        dataset=ShapeDataset(),
        split=[0.5, 0.5, 0],
        # train_val_sets=[EXP.CONSTANT_EX3, EXP.CONSTANT_EX4, EXP.RANDOM_EX3, EXP.RANDOM_EX5],
        train_val_sets="all",
        separate_test_set=None,
    )

    callbacks = []
    if logger is not None:
        callbacks.append(PredictionPlotting(epochs=[], subset_size=5))

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        enable_checkpointing=False,
        accelerator=DEVICE,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
