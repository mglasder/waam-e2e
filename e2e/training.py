from pathlib import Path

import requests
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import ReduceLROnPlateau
from lightning.pytorch.loggers import WandbLogger

from e2e.autogit.autogit import git_add_commit_with
from e2e.callbacks.metrics import FootprintAvgAbsValErrorLogger, ModHausdorffLogger
from e2e.callbacks.plotting import PredictionPlotting
from e2e.data.datamodule import ShapePredictionDataModule
from e2e.data.dataset import ShapeDataset, ResampledFootprintDataset
from e2e.data.loader import EXPERIMENT as EXP
from e2e.mcpredict import McUncertainty
from e2e.models.modelV2 import ModelV2
from e2e.models.recurrent import RNN

NAS_DATA_DIR_DEV = Path("/Volumes/hornets/homes/mglasder/datasets/TrainingDev")
MAC_DATA_DIR_DEV = Path("/Users/magnus/datasets/WAAM/TrainingDev")
VM_DATA_DIR = Path("/home/magnus/datasets/waam/30_processing_results/ImageGenerator")
VM_DATA_DIR_DEV = Path("/home/magnus/datasets/waam/TrainingDev")

BATCH_SIZE = 16
MAX_EPOCHS = 100
N_WORKERS = 16
DEVICE = "cuda"
# footprint
THETA = 0.0
# smoothness
LAMBDA = 0.5
# area
GAMMA = 0.0
INPUT_LENGTH = 90
TARGET_LENGTH = 90
LR = 0.0005
P = 0.5

DEV_RUN = False
LOGGING = True
AUTOCOMMIT = True
AUTOCOMMIT_IP = "172.31.1.8"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


def main():
    # unet = UNet1D(enconf=EncoderConfig(), deconf=DecoderConfig())
    # unet.to(DEVICE)

    # conv = SimpleConv(in_channels=1, out_channels=TARGET_LENGTH, kernel_size=120, p=P)
    # conv.to(DEVICE)

    # mlp = MLP(n_features=TARGET_LENGTH, p=P)
    # mlp.to(DEVICE)

    rnn = RNN(p=P, n_input_features=INPUT_LENGTH, n_output_features=TARGET_LENGTH, n_hidden=TARGET_LENGTH, n_layers=40)
    rnn.to(DEVICE)

    model = ModelV2(
        model=rnn,
        batch_size=BATCH_SIZE,
        lr=LR,
        theta=THETA,
        lambda_=LAMBDA,
        gamma=GAMMA,
        in_len=INPUT_LENGTH,
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

    if AUTOCOMMIT and LOGGING and not DEV_RUN:
        # TODO: wrap everything in its own class
        # get wandb run name
        run_name = logger.experiment.name

        # check whether remote or local machine
        if Path("/home/magnus").exists():
            message = f"""{run_name}"""
            response = requests.post(f"http://{AUTOCOMMIT_IP}:3000/execute", json={"message": message})
            commit_hash = response.json()["result"]

        else:
            commit_hash = git_add_commit_with(message=f"{run_name}")

        print(f"Commit hash:, {commit_hash}")
        logger.experiment.config.update({"commit": commit_hash})

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
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=10,
                monitor="val_loss",
                mode="min",
                auto_insert_metric_name=True,
                save_on_train_epoch_end=False,
            )
        )
        callbacks.append(PredictionPlotting(epochs=[]))
        # callbacks.append(FootprintAvgAbsValErrorLogger())
        # callbacks.append(ModHausdorffLogger())
        # add learning rate scheduler ReduceLROnPlateau
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer=model.optimizers(), mode="min", patience=5, factor=0.1, verbose=True
        # )
        # callbacks.append(scheduler)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        enable_checkpointing=True,
        accelerator=DEVICE,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model=model, datamodule=datamodule)

    val_data_loader = datamodule.val_dataloader()
    train_data_loader = datamodule.train_dataloader()

    mc = McUncertainty(model, train_data_loader, val_data_loader, logger=logger)
    mc.predict()
    mc.calibrate(strategy="mlp")
    mc.plot_predictions("train", log=True, take=30)
    mc.plot_predictions("val", log=True, take=30)


if __name__ == "__main__":
    main()
