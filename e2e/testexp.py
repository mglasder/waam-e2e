from pathlib import Path

import numpy as np
import torch

from e2e.data.datamodule import ShapePredictionDataModule
from e2e.data.resampled import ResampledShapeDataset
from e2e.helpers import timing
from e2e.models.mlp import ResMLP
from e2e.models.model import Model
from e2e.models.modelV2 import ModelV2

timing.ENABLE_TIMING = True
from e2e.data.loader import EXPERIMENT as EXP
from e2e.models.recurrent import LSTM
from e2e.prediction.end_to_end import DataLoader, ModelHandler, Plotter, Predictor

PROJECT_NAME = "waam-e2e-pre"
RUN_ID = "q2btbt08"

# "oqfrcpcw"
# "oqfrcpcw"
# "isn0h9ws"  # first_in_row + 2x
# "xtuq679p"    # first_in_row + 4x

# GLOBAL
VERSION = "best"
VM_DATA_DIR = Path("/home/magnus/datasets/waam/30_processing_results/ImageGenerator")
MAC_DATA_DIR = Path("/Users/magnus/datasets/WAAM/test")


# MLP
MLP_PROJECT_NAME = "waam-e2e-footprint"
MLP_RUN_ID = "y21szkna"

MLP_INPUT_LENGTH = 224
MLP_HIDDEN_LENGTH_FACTOR = 2
MLP_TARGET_LENGTH = 2
MLP_N_LAYERS = 3
MLP_P = 0.5

# LSTM
LSTM_PROJECT_NAME = "waam-e2e-pre"
LSTM_RUN_ID = "hyxp1rgr"

LSTM_INPUT_LENGTH = 90
LSTM_TARGET_LENGTH = 90
LSTM_P = 0.5


def main():
    # configure models
    mlp = ResMLP(
        p=MLP_P,
        n_input_features=MLP_INPUT_LENGTH,
        n_hidden=MLP_INPUT_LENGTH * MLP_HIDDEN_LENGTH_FACTOR,
        n_output_features=MLP_TARGET_LENGTH,
        n_layers=MLP_N_LAYERS,
    )

    lstm = LSTM(
        p=LSTM_P,
        n_input_features=LSTM_INPUT_LENGTH,
        n_output_features=LSTM_TARGET_LENGTH,
        n_hidden=LSTM_TARGET_LENGTH * 3,
        n_layers=10,
    )

    # load dataset
    data_loader = DataLoader(MAC_DATA_DIR)
    cross_section_samples = data_loader.load_samples([EXP.RANDOM_EX6])
    dataset = ResampledShapeDataset(mirror=False, segment_length=MLP_INPUT_LENGTH).create(cross_section_samples)

    # instantiate models
    mlp_handler = ModelHandler(MLP_PROJECT_NAME, MLP_RUN_ID, VERSION, model=Model, module=mlp)
    # lstm_handler = ModelHandler(LSTM_PROJECT_NAME, LSTM_RUN_ID, VERSION, model=ModelV2, module=lstm)

    base = Path("/Users/magnus/repos/waam-e2e/e2e/")
    model_path = base / Path("./waam-e2e-pre/mbw4tdhh/checkpoints/epoch=199-step=2400.ckpt")

    lstm_handler = ModelHandler(
        LSTM_PROJECT_NAME, LSTM_RUN_ID, VERSION, model=ModelV2, module=lstm, model_path=model_path
    )

    predictor = Predictor(MLP_INPUT_LENGTH, mlp_handler, lstm_handler, cross_section_samples)

    init_input = torch.tensor(dataset[0][0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).clone().detach()

    alt_input = torch.zeros_like(init_input)

    # for i in range(1):
    #     s = dataset[i][0]
    #     plt.plot(s)
    #
    # plt.plot(alt_input.squeeze().numpy())
    # plt.show()

    datamodule = ShapePredictionDataModule(
        batch_size=1,
        data_dir=MAC_DATA_DIR,
        workers=1,
        dataset=ResampledShapeDataset(mirror=False, segment_length=MLP_INPUT_LENGTH),
        split=[0.7, 0.3, 0],
        train_val_sets=[],
        separate_test_set=EXP.RANDOM_EX6,
        seed=2345078,
    )

    datamodule.setup("test")

    test_loader = datamodule.test_dataloader()

    for s in test_loader:
        break

    in_sample = torch.tensor(s[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).clone().detach()

    UNCERT_THRESH = 1000.0
    predictions, labels, ground_truth = predictor.predict(
        in_sample,
        uncertainty_threshold=UNCERT_THRESH,
        mode="e2e",
        predict_footprint=False,
    )
    Plotter.plot_e2e(predictions, labels, ground_truth, title=f"E2E uncert:{UNCERT_THRESH}")

    print(f"fallback rate to alternative prediction: {np.sum(labels)/len(labels)*100:.2f}%")

    # predictions, labels, ground_truth = predictor.predict(init_input, uncertainty_threshold=4.5, mode="hybrid")
    # Plotter.plot_e2e(predictions, labels, ground_truth, title="Hybrid")


if __name__ == "__main__":
    main()
