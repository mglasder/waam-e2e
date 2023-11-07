from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

from e2e.data.resampled import ResampledE2EDataset
from e2e.helpers import timing
from e2e.helpers.wandb import download_model
from e2e.models.mlp import ResMLP
from e2e.models.model import Model
from e2e.models.modelV2 import ModelV2

timing.ENABLE_TIMING = True
import matplotlib.pyplot as plt

from e2e.data.loader import EXPERIMENT as EXP
from e2e.models.recurrent import LSTM
from e2e.prediction.end_to_end import DataLoader

plt.style.use(["science", "ieee", "grid"])
plt.rcParams.update({"font.size": 12})


def plot_outline_with_uncertainty_envelope(mean_outline, std_outline):
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(30 * cm, 10 * cm), dpi=600)

    xa = np.arange(0, len(mean_outline)) / 10
    ax.plot(xa, mean_outline, color="blue", alpha=1, ls="-", linewidth=1)

    ax.set_ylim(0, mean_outline.max() + 2)
    ax.set_xlim(0, xa.max())
    ax.set_aspect("equal")

    ax.fill_between(
        xa,
        mean_outline + std_outline,
        mean_outline - std_outline,
        color="red",
        alpha=0.3,
        label="$\pm 1\sigma$",
    )

    ax.fill_between(
        xa,
        mean_outline,
        color="lightgrey",
        hatch="\\",
        edgecolor="black",
        alpha=0.3,
    )

    # add legend below graph
    ax.plot([], [], color="blue", alpha=1, linewidth=1, label="mean prediction")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
        fancybox=False,
        shadow=False,
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    return fig


def run_e2e_prediction_with_uncertainty(footprint_predictor, shape_predictor, dataset, N_PREDICTIONS=30):
    base_input = torch.zeros_like(torch.tensor(dataset.x_sections[0].slice_based_before.ys), dtype=torch.float32)
    torchpositions = dataset.torchpositions

    mid_idx = 224 // 2
    curr_base = base_input

    dataset.labels.append(0)

    with torch.no_grad():
        for STEP in range(len(torchpositions)):
            dataset.predictions.append(curr_base.numpy())

            curr_torch = torchpositions[STEP]
            curr_input = curr_base[curr_torch - 112 : curr_torch + 112].unsqueeze(0).clone()

            # predict footprint
            footprints = footprint_predictor.predict_with_uncertainty(
                curr_input,
                num_samples=N_PREDICTIONS,
                reduction="none",
            )

            # chose one from footprints randomly
            footprint = footprints[np.random.randint(0, N_PREDICTIONS)]
            left_fp = footprint[:, 0].int().item()
            right_fp = footprint[:, 1].int().item()

            # offset
            z_offset = curr_input[:, 112].clone().item()
            curr_input -= z_offset

            # predict shape
            segment = curr_input[:, left_fp:right_fp]
            len_segment = segment.size(1)
            xs_original = np.arange(0, len_segment) / 10
            f = interp1d(xs_original, segment, kind="linear")
            xs_90 = np.linspace(0, (len_segment - 1) / 10, 90)
            curr_input_segment = f(xs_90)

            shapes = shape_predictor.predict_with_uncertainty(
                torch.tensor(curr_input_segment, dtype=torch.float32).view(-1, 90),
                num_samples=N_PREDICTIONS,
                reduction="none",
            )

            # chose one from shapes randomly
            shape = shapes[np.random.randint(0, N_PREDICTIONS)]

            # sample to original length
            f = interp1d(xs_90, shape, kind="linear")
            shape_prediction = f(xs_original)
            shape_prediction += z_offset

            next_base = curr_base.clone()
            next_base[curr_torch - (mid_idx - left_fp) : curr_torch + (right_fp - mid_idx)] = torch.tensor(
                shape_prediction.copy(), dtype=torch.float32
            )
            curr_base = next_base.clone()
        dataset.predictions.append(curr_base.numpy())
        return dataset


if __name__ == "__main__":
    MAC_DATA_DIR = Path("/Users/magnus/datasets/WAAM/test")
    MLP_INPUT_LENGTH = 224
    data_loader = DataLoader(MAC_DATA_DIR)
    cross_section_samples = data_loader.load_samples([EXP.RANDOM_EX6])
    dataset = ResampledE2EDataset(mirror=False, segment_length=MLP_INPUT_LENGTH).create(cross_section_samples)
    # GLOBAL
    VERSION = "best"
    DEVICE = "cpu"

    # MLP
    MLP_PROJECT_NAME = "waam-e2e-footprint"
    MLP_RUN_ID = "y21szkna"

    MLP_INPUT_LENGTH = 224
    MLP_HIDDEN_LENGTH_FACTOR = 2
    MLP_TARGET_LENGTH = 2
    MLP_N_LAYERS = 3
    MLP_P = 0.5

    # LSTM
    LSTM_INPUT_LENGTH = 90
    LSTM_TARGET_LENGTH = 90
    LSTM_P = 0.9

    N_PREDICTIONS = 30

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

    # instantiate models

    footprint_path = download_model(project=MLP_PROJECT_NAME, run_id=MLP_RUN_ID, version=VERSION)

    footprint_predictor = Model.load_from_checkpoint(
        model=mlp, checkpoint_path=footprint_path, map_location=torch.device(DEVICE)
    )

    base = Path("/Users/magnus/repos/waam-e2e/e2e/")
    lstm_path = base / Path("./waam-e2e-pre/mbw4tdhh/checkpoints/epoch=199-step=2400.ckpt")
    shape_predictor = ModelV2.load_from_checkpoint(
        model=lstm, checkpoint_path=lstm_path, map_location=torch.device(DEVICE)
    )
    shape_predictor.to("cpu")

    dataset = run_e2e_prediction_with_uncertainty(footprint_predictor, shape_predictor, dataset, N_PREDICTIONS=30)
    outline = np.array(dataset.predictions).max(axis=0)[200:901]
