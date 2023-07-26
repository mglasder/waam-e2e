from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from e2e.callbacks.metrics import FootprintAvgAbsValErrorLogger, ModHausdorffLogger
from e2e.callbacks.plotting import PredictionPlotting
from e2e.data.datamodule import ShapePredictionDataModule
from e2e.data.dataset import ShapeDataset
from e2e.data.loader import EXPERIMENT as EXP
from e2e.models.modelV2 import ModelV2
from e2e.models.simpleconv import SimpleConv

NAS_DATA_DIR_DEV = Path("/Volumes/hornets/homes/mglasder/datasets/TrainingDev")
MAC_DATA_DIR_DEV = Path("/Users/magnus/datasets/WAAM/TrainingDev")
VM_DATA_DIR = Path("/home/magnus/datasets/waam/30_processing_results/ImageGenerator")
VM_DATA_DIR_DEV = Path("/home/magnus/datasets/waam/TrainingDev")

BATCH_SIZE = 16
MAX_EPOCHS = 50
N_WORKERS = 16
DEVICE = "cuda"
# footprint
THETA = 0.2
# smoothness
LAMBDA = 0.0
# area
GAMMA = 0.0
DEV_RUN = False
LOGGING = True
TARGET_LENGTH = 120
LR = 0.0005
P = 0

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


def main():
    # unet = UNet1D(enconf=EncoderConfig(), deconf=DecoderConfig())
    # unet.to(DEVICE)

    conv = SimpleConv(in_channels=1, out_channels=TARGET_LENGTH, kernel_size=120, p=P)
    conv.to(DEVICE)

    model = ModelV2(
        model=conv,
        batch_size=BATCH_SIZE,
        lr=LR,
        theta=THETA,
        lambda_=LAMBDA,
        gamma=GAMMA,
        in_len=TARGET_LENGTH,
        out_len=TARGET_LENGTH,
    )

    if DEV_RUN:
        print("THIS IS A DEV RUN! Used dataset and split are adjusted accordingly.")
        split = [0.5, 0.5, 0]
        train_val_sets = "all"
        separate_test_set = None

    else:
        split = [0.7, 0.3, 0]
        train_val_sets = [EXP.CONSTANT_EX3, EXP.CONSTANT_EX4, EXP.RANDOM_EX3, EXP.RANDOM_EX5]
        separate_test_set = None

    if LOGGING:
        logger = WandbLogger(project="waam-e2e-pre", log_model="all")
    else:
        logger = None

    datamodule = ShapePredictionDataModule(
        batch_size=BATCH_SIZE,
        data_dir=VM_DATA_DIR,
        workers=N_WORKERS,
        dataset=ShapeDataset(segment_length=TARGET_LENGTH),
        split=split,
        train_val_sets=train_val_sets,
        separate_test_set=separate_test_set,
    )

    callbacks = []
    if logger is not None:
        # callbacks.append(EarlyStopping(monitor="val_loss", patience=10, min_delta=0.001, mode="min"))
        # callbacks.append(PredictionPlotting(epochs=[]))
        callbacks.append(PredictionPlotting(epochs=[]))
        callbacks.append(FootprintAvgAbsValErrorLogger())
        callbacks.append(ModHausdorffLogger())

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
