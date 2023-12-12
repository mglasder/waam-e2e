from pathlib import Path

import requests
import torch
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from e2e.autogit.autogit import git_add_commit_with
from e2e.callbacks.plotting import PredictionPlotting
from e2e.data.datamodule import ShapePredictionDataModule
from e2e.data.loader import EXPERIMENT as EXP
from e2e.data.loader import SampleLoader
from e2e.data.resampled import ResampledShapePointsDataset
from e2e.models.modelV2 import ModelPoints
from e2e.models.recurrent import ShapePointsModel

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# NAS_DATA_DIR_DEV = Path("/Volumes/hornets/homes/mglasder/datasets/TrainingDev")
MAC_DATA_DIR_DEV = Path("/Users/magnus/datasets/WAAM/TrainingDev")
VM_DATA_DIR = Path("/home/magnus/datasets/waam/30_processing_results/ImageGenerator")
VM_DATA_DIR_DEV = Path("/home/magnus/datasets/waam/TrainingDev")

DATASET = VM_DATA_DIR

SEED = 2345078
BATCH_SIZE = 64
MAX_EPOCHS = 50
N_WORKERS = 2
DEVICE = "cuda"

INPUT_LENGTH = 100
TARGET_LENGTH = 100

LR = 0.001
P = 0.0

MIRROR = True
DEV_RUN = False
LOGGING = True
AUTOCOMMIT = True
AUTOCOMMIT_IP = "172.31.1.8"
CHECKPOINTING_ENABLED = False

PROJECT = "waam-e2e-shape-points"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


def auto_commit(run_name: str, AUTOCOMMIT_IP: str, note: str):
    if Path("/home/magnus").exists():
        message = f"""{run_name}: {note}"""
        response = requests.post(f"http://{AUTOCOMMIT_IP}:3000/execute", json={"message": message})
        commit_hash = response.json()["result"]
    else:
        commit_hash = git_add_commit_with(message=f"{run_name}")
    print(f"Commit hash:, {commit_hash}")
    return commit_hash


def main(note: str = ""):
    wandb.init(project=PROJECT)
    config = wandb.config

    # add global params to config
    config["seed"] = SEED
    config["input_length"] = INPUT_LENGTH
    config["target_length"] = TARGET_LENGTH
    config["batch_size"] = BATCH_SIZE

    points = ShapePointsModel(
        p=P,
        n_input_features=INPUT_LENGTH,
        n_output_features=TARGET_LENGTH,
    )
    points.to(DEVICE)

    model = ModelPoints(
        model=points,
        batch_size=BATCH_SIZE,
        lr=LR,
        in_len=INPUT_LENGTH,
        out_len=TARGET_LENGTH,
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
        logger.watch(model.model, log="all")  # log gradients and params
        run_name = logger.experiment.name
        logger.experiment.notes = note

    else:
        logger = None
        run_name = None

    if AUTOCOMMIT and LOGGING and not DEV_RUN:
        commit_hash = auto_commit(run_name, AUTOCOMMIT_IP, note)
        logger.experiment.config.update({"commit": commit_hash})

    loader = SampleLoader(sample_dir=DATASET)

    datamodule = ShapePredictionDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATASET,
        workers=N_WORKERS,
        dataset=ResampledShapePointsDataset(mirror=MIRROR, segment_length=TARGET_LENGTH),
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
        if CHECKPOINTING_ENABLED:
            callbacks.append(
                ModelCheckpoint(
                    every_n_epochs=5,
                    monitor="val_loss",
                    mode="min",
                    auto_insert_metric_name=True,
                    save_on_train_epoch_end=False,
                )
            )
        callbacks.append(PredictionPlotting(epochs=[0, 10, 20, 30, 40]))
        # callbacks.append(FootprintAvgAbsValErrorLogger(footprint_is_absolute=True))
        # callbacks.append(ModHausdorffLogger())
        # callbacks.append(LogModelParametersAndGradients())

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        enable_checkpointing=CHECKPOINTING_ENABLED,
        accelerator=DEVICE,
        callbacks=callbacks,
        log_every_n_steps=5,
    )
    trainer.fit(model=model, datamodule=datamodule)

    # val_data_loader = datamodule.val_dataloader()
    # train_data_loader = datamodule.train_dataloader()
    # test_data_loader = datamodule.test_dataloader()
    #
    # # get best model path
    # model_path = trainer.checkpoint_callback.best_model_path
    # print(model_path)
    # best_model = ModelPoints.load_from_checkpoint(model=lstm, checkpoint_path=model_path)
    # best_model.to("cpu")
    #
    # mc = McUncertainty(best_model, train_data_loader, val_data_loader, test_dataloader=test_data_
    # loader, logger=logger)
    # mc.predict()
    # mc.calibrate(strategy="temperature_scaling")
    # # mc.plot_predictions("train", log=LOGGING, take=20)
    # mc.plot_predictions("val", log=LOGGING, take=20)
    # mc.plot_predictions("test", log=LOGGING, take=None)
    # #
    # names = ["mean_predictions", "uncertainties_calib", "xs", "ys", "ids", "temperatures"]
    # # date and time up to seconds
    # now = datetime.now().strftime("%Y%m%d-%H%M%S")
    # os.mkdir(Path(f"outputs/mc/{now}-{run_name}"))
    #
    # # create directory
    # for name in names:
    #     items = np.array(mc._uncertainty_preds["test"][name])
    #     if name == "ids":
    #         np.savetxt(f"outputs/mc/{now}-{run_name}/{name}.csv", items, delimiter=",", fmt="%s")
    #     else:
    #         np.savetxt(f"outputs/mc/{now}-{run_name}/{name}.csv", items, delimiter=",")
    #
    # logger.experiment.finish()

    # run_e2e_prediction(best_model, DEVICE)


if __name__ == "__main__":
    if LOGGING:
        note = input("Enter run note: ")
    else:
        note = ""

    main(note=note)

    # sweep_config = yaml.safe_load((open("sweep-config.yaml", "r")))
    # sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    # wandb.agent(sweep_id, function=main, project=PROJECT)
