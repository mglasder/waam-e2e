from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from e2e.data.loader import EXPERIMENT as EXP
from e2e.data.resampled import ResampledE2EDataset
from e2e.helpers.wandb import download_model
from e2e.models.mlp import ResMLP
from e2e.models.model import Model
from e2e.models.modelV2 import ModelV2
from e2e.models.recurrent import LSTM
from e2e.prediction.end_to_end import DataLoader

plt.style.use(["science", "ieee", "grid"])
plt.rcParams.update({"font.size": 12})
plt.rcParams.update({"legend.frameon": "False"})
# plt.rcParams.update({"mathtext.default": "regular"})


def setup():
    VERSION = "best"
    MAC_DATA_DIR = Path("/Users/magnus/datasets/WAAM/test")

    # MLP
    MLP_PROJECT_NAME = "waam-e2e-footprint"
    MLP_RUN_ID = "y21szkna"

    MLP_INPUT_LENGTH = 224
    MLP_HIDDEN_LENGTH_FACTOR = 2
    MLP_TARGET_LENGTH = 2
    MLP_N_LAYERS = 3
    MLP_P = 0.5

    DEVICE = "cpu"

    LSTM_INPUT_LENGTH = 90
    LSTM_TARGET_LENGTH = 90
    LSTM_P = 0.9

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

    footprint_path = download_model(project=MLP_PROJECT_NAME, run_id=MLP_RUN_ID, version=VERSION)

    footprint_predictor = Model.load_from_checkpoint(
        model=mlp, checkpoint_path=footprint_path, map_location=torch.device(DEVICE)
    )

    data_loader = DataLoader(MAC_DATA_DIR)
    cross_section_samples = data_loader.load_samples([EXP.RANDOM_EX6])
    dataset = ResampledE2EDataset(mirror=False, segment_length=MLP_INPUT_LENGTH).create(cross_section_samples)

    base = Path("/Users/magnus/repos/waam-e2e/e2e/")
    lstm_path = base / Path("./waam-e2e-pre/mbw4tdhh/checkpoints/epoch=199-step=2400.ckpt")
    shape_predictor = ModelV2.load_from_checkpoint(
        model=lstm, checkpoint_path=lstm_path, map_location=torch.device(DEVICE)
    )
    shape_predictor.to("cpu")

    return dataset, footprint_predictor, shape_predictor


def predict_footprint(base, footprint_predictor):
    N_PREDICTIONS = 10
    fp_preds = []

    footprint = footprint_predictor.predict_with_uncertainty(base, num_samples=N_PREDICTIONS, reduction="none")
    fp_preds.append(footprint.cpu().numpy())

    fp_preds = np.array(fp_preds).reshape(-1, 2)
    return fp_preds


def propagate_to_shape(fp_preds, base, shape_predictor):
    from scipy.interpolate import interp1d

    shape_predictions = []
    shape_uncertainties = []

    curr_input = base.squeeze().cpu().numpy().copy()

    for footprint in fp_preds:
        footprint = footprint.reshape(-1, 2)

        left_fp = int(np.round(footprint[:, 0]))
        right_fp = int(np.round(footprint[:, 1]))

        z_offset = curr_input[112]
        curr_input -= z_offset

        u = np.zeros_like(curr_input)

        # cut out footprint ~67 points
        segment = curr_input[left_fp:right_fp].copy()
        len_segment = len(segment)
        xs_original = np.arange(0, len_segment) / 10
        f = interp1d(xs_original, segment, kind="linear")

        # upsample to 90 points
        xs_90 = np.linspace(0, (len_segment - 1) / 10, 90)
        curr_input_segment = f(xs_90)

        # predict -> output is 90 points
        pred, uncertainty = shape_predictor.predict_with_uncertainty(
            torch.tensor(curr_input_segment, dtype=torch.float32).view(-1, 90),
            num_samples=30,
        )
        pred = pred.detach().cpu().squeeze().numpy()

        uncertainty_score = np.sum(uncertainty.detach().cpu().numpy())

        # downsample to ~67 points
        f = interp1d(xs_90, pred, kind="linear")
        shape_prediction = f(xs_original)
        shape_prediction += z_offset

        fu = interp1d(xs_90, uncertainty.cpu().squeeze().numpy(), kind="linear")
        u[left_fp:right_fp] = fu(xs_original)

        # put back into workpiece segment of total 224 points
        pred_out = curr_input.copy()
        pred_out[left_fp:right_fp] = shape_prediction

        shape_uncertainties.append(u)
        shape_predictions.append(pred_out)

    shape_predictions = np.array(shape_predictions)
    shape_uncertainties = np.array(shape_uncertainties)

    return shape_predictions, shape_uncertainties


def get_true_next(tp, fp, curr_input, target):
    fp_left, fp_right = tp - fp.left_idx, fp.right_idx - tp

    left = 112 - fp_left
    right = 112 + fp_right
    xs_224 = np.arange(0, 224) * 0.1
    true_next = curr_input.copy()
    ft = interp1d(xs_224, target, kind="linear")
    x_target = np.linspace(0, (224 - 1) / 10, right - left)
    true_next[left:right] = ft(x_target)


def plot(base, target, shape_predictions):
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(18 * cm, 10 * cm), dpi=600)

    start = 70
    end = 150

    x = (np.arange(0, 224) * 0.1)[start:end]
    inpt = base.squeeze()
    ax.plot(x, inpt[start:end], color="black", ls="-", alpha=1, label="input $W_i$")
    ax.plot(x, true_next[start:end], color="green", ls="--", alpha=1, label="target (measured) $W_{i+1}$")

    # for shape in shape_predictions:
    #     ax.plot(x, shape[start:end], color="blue", ls="-",alpha=0.3)

    ax.plot(
        x,
        shape_predictions.mean(axis=0)[start:end],
        color="blue",
        ls="-",
        alpha=1,
        label="prediction (mean), $\hat{F}$ and $\hat{S}$",
    )
    ax.fill_between(
        x,
        (shape_predictions.mean(axis=0) - shape_predictions.std(axis=0))[start:end],
        (shape_predictions.mean(axis=0) + shape_predictions.std(axis=0))[start:end],
        color="red",
        alpha=0.2,
        label="uncertainty $U(\hat{S}(x))$",
    )

    ax.axis("equal")
    ax.set_xlim(7, 15)
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$z$ [mm]")

    # add legend below plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.5))
    plt.subplots_adjust(bottom=0.3)
    plt.show()


def get_data(BEAD_NR, dataset):
    target = dataset.targets[BEAD_NR].clone().cpu().numpy()
    tp = dataset.torchpositions[BEAD_NR]
    fp = dataset.true_footprints[BEAD_NR]
    base = dataset.ground_truth[BEAD_NR][tp - 112 : tp + 112]
    return base, target, tp, fp


if __name__ == "__main__":
    BEAD_NR = 6
    dataset, footprint_predictor, shape_predictor = setup()

    for i in [0, 5, 10, 15, 20]:
        base, target, tp, fp = get_data(BEAD_NR, dataset)

        base = torch.tensor(base, dtype=torch.float32).view(-1, 224)

        predicted_footprints = predict_footprint(base, footprint_predictor)
        shape_predictions, shape_uncertainties = propagate_to_shape(predicted_footprints, base, shape_predictor)

        fp_left, fp_right = tp - fp.left_idx, fp.right_idx - tp

        left = 112 - fp_left
        right = 112 + fp_right
        xs_224 = np.arange(0, 224) * 0.1
        true_next = base.squeeze().cpu().numpy().copy()
        ft = interp1d(xs_224, target, kind="linear")
        x_target = np.linspace(0, (224 - 1) / 10, right - left)
        true_next[left:right] = ft(x_target)

        plot(base, true_next, shape_predictions)
