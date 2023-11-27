import os
from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np
import requests
import torch
import wandb
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from e2e.autogit.autogit import git_add_commit_with
from e2e.callbacks.metrics import FootprintAvgAbsValErrorLogger, ModHausdorffLogger
from e2e.data.datamodule import ShapePredictionDataModule
from e2e.data.loader import EXPERIMENT as EXP
from e2e.data.loader import SampleLoader
from e2e.data.resampled import ResampledShapeDataset
from e2e.mcpredict import McUncertainty
from e2e.models.modelV2 import ModelV2
from e2e.models.recurrent import LSTM

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# NAS_DATA_DIR_DEV = Path("/Volumes/hornets/homes/mglasder/datasets/TrainingDev")
MAC_DATA_DIR_DEV = Path("/Users/magnus/datasets/WAAM/TrainingDev")
VM_DATA_DIR = Path("/home/magnus/datasets/waam/30_processing_results/ImageGenerator")
VM_DATA_DIR_DEV = Path("/home/magnus/datasets/waam/TrainingDev")

DATASET = VM_DATA_DIR

SEED = 2345078
BATCH_SIZE = 64
MAX_EPOCHS = 100
N_WORKERS = 3
DEVICE = "cuda"
# footprint
THETA = 0.4
# smoothness
LAMBDA = 0.0
# surface energy
GAMMA = 0.0

INPUT_LENGTH = 90
TARGET_LENGTH = 90
N_OUTPUTS = TARGET_LENGTH  # predict z-values directly
LR = 0.001
P = 0.6

MIRROR = True
DEV_RUN = False
LOGGING = True
AUTOCOMMIT = False
AUTOCOMMIT_IP = "172.31.1.8"

PROJECT = "waam-e2e-pre"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


def main(note: str = ""):
    wandb.init(project=PROJECT)
    config = wandb.config

    config.update({"n_lstm_hidden": 90, "n_lstm_layers": 1})
    print("Config: \n")
    pprint(config)

    lstm = LSTM(
        p=P,
        n_input_features=INPUT_LENGTH,
        n_output_features=TARGET_LENGTH,
        n_outputs=N_OUTPUTS,
        n_hidden=config["n_lstm_hidden"],
        n_layers=config["n_lstm_layers"],
    )
    lstm.to(DEVICE)

    model = ModelV2(
        model=lstm,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        lr=LR,
        in_len=INPUT_LENGTH + 1,
        out_len=TARGET_LENGTH,
        n_outputs=N_OUTPUTS,
        mode="pure",
    )
    model.to(DEVICE)

    if DEV_RUN:
        print("THIS IS A DEV RUN! Used dataset and split are adjusted accordingly.")
        split = [0.5, 0.5, 0]
        train_val_sets = "all"
        separate_test_set = None

    else:
        split = [0.7, 0.3, 0]
        train_val_sets = [EXP.CONSTANT_EX3, EXP.CONSTANT_EX4, EXP.RANDOM_EX3, EXP.RANDOM_EX5]
        separate_test_set = EXP.RANDOM_EX6

    if LOGGING:
        logger = WandbLogger(project=PROJECT, log_model="all")
        logger.watch(model.model)
        run_name = logger.experiment.name
        logger.experiment.notes = note

    else:
        logger = None
        run_name = None

    if AUTOCOMMIT and LOGGING and not DEV_RUN:
        # TODO: wrap everything in its own class
        # check whether remote or local machine
        if Path("/home/magnus").exists():
            message = f"""{run_name}: {note}"""
            response = requests.post(f"http://{AUTOCOMMIT_IP}:3000/execute", json={"message": message})
            commit_hash = response.json()["result"]

        else:
            commit_hash = git_add_commit_with(message=f"{run_name}")

        print(f"Commit hash:, {commit_hash}")
        logger.experiment.config.update({"commit": commit_hash})

    loader = SampleLoader(sample_dir=DATASET)

    datamodule = ShapePredictionDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATASET,
        workers=N_WORKERS,
        dataset=ResampledShapeDataset(mirror=MIRROR, segment_length=TARGET_LENGTH),
        split=split,
        data_fraction=1.0,
        train_val_sets=train_val_sets,
        separate_test_set=separate_test_set,
        seed=SEED,
        sample_loader=loader,
    )

    callbacks = []
    if logger is not None:
        # callbacks.append(EarlyStopping(monitor="val_loss", patience=10, min_delta=0.001, mode="min"))
        # callbacks.append(PredictionPlotting(epochs=[]))
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=5,
                monitor="val_loss",
                mode="min",
                auto_insert_metric_name=True,
                save_on_train_epoch_end=False,
            )
        )
        # callbacks.append(PredictionPlotting(epochs=[]))
        callbacks.append(FootprintAvgAbsValErrorLogger(footprint_is_absolute=True))
        callbacks.append(ModHausdorffLogger())
        # callbacks.append(LogModelParametersAndGradients())

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        enable_checkpointing=True,
        accelerator=DEVICE,
        callbacks=callbacks,
        log_every_n_steps=5,
    )
    trainer.fit(model=model, datamodule=datamodule)

    val_data_loader = datamodule.val_dataloader()
    train_data_loader = datamodule.train_dataloader()
    test_data_loader = datamodule.test_dataloader()

    # get best model path
    model_path = trainer.checkpoint_callback.best_model_path
    print(model_path)
    best_model = ModelV2.load_from_checkpoint(model=lstm, checkpoint_path=model_path)
    best_model.to("cpu")

    mc = McUncertainty(best_model, train_data_loader, val_data_loader, test_dataloader=test_data_loader, logger=logger)
    mc.predict()
    mc.calibrate(strategy="temperature_scaling")
    # mc.plot_predictions("train", log=LOGGING, take=20)
    mc.plot_predictions("val", log=LOGGING, take=20)
    mc.plot_predictions("test", log=LOGGING, take=None)
    #
    names = ["mean_predictions", "uncertainties_calib", "xs", "ys", "ids", "temperatures"]
    # date and time up to seconds
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(Path(f"outputs/mc/{now}-{run_name}"))

    # create directory
    for name in names:
        items = np.array(mc._uncertainty_preds["test"][name])
        if name == "ids":
            np.savetxt(f"outputs/mc/{now}-{run_name}/{name}.csv", items, delimiter=",", fmt="%s")
        else:
            np.savetxt(f"outputs/mc/{now}-{run_name}/{name}.csv", items, delimiter=",")

    logger.experiment.finish()

    # run_e2e_prediction(best_model, DEVICE)


if __name__ == "__main__":
    # if LOGGING:
    #     note = input("Enter run note: ")
    # else:
    #     note = ""
    #
    # main(note=note)

    sweep_config = yaml.safe_load((open("sweep-config.yaml", "r")))
    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    wandb.agent(sweep_id, function=main, project=PROJECT)
