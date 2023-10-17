from datetime import datetime
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
from e2e.data.loader import EXPERIMENT as EXP
from e2e.models.recurrent import LSTM
from e2e.prediction.end_to_end import DataLoader, Plotter


def run_e2e_prediction(shape_predictor, DEVICE="cpu", FOOTPRINT_PREDICT=False, U_THRESHOLD=1000.0):
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
    LSTM_INPUT_LENGTH = 90
    LSTM_TARGET_LENGTH = 90
    LSTM_P = 0.5

    N_PREDICTIONS = 150

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
    data_loader = DataLoader(VM_DATA_DIR)
    cross_section_samples = data_loader.load_samples([EXP.RANDOM_EX6])
    dataset = ResampledE2EDataset(mirror=False, segment_length=MLP_INPUT_LENGTH).create(cross_section_samples)

    # instantiate models

    footprint_path = download_model(project=MLP_PROJECT_NAME, run_id=MLP_RUN_ID, version=VERSION)

    footprint_predictor = Model.load_from_checkpoint(
        model=mlp, checkpoint_path=footprint_path, map_location=torch.device(DEVICE)
    )

    if not shape_predictor:
        base = Path("/home/magnus/repos/waam-e2e/e2e/")
        lstm_path = base / Path("./waam-e2e-pre/ma9hzh4c/checkpoints/epoch=199-step=2000.ckpt")
        shape_predictor = ModelV2.load_from_checkpoint(
            model=lstm, checkpoint_path=lstm_path, map_location=torch.device(DEVICE)
        )
        shape_predictor.to("cpu")

    base_input = torch.zeros_like(torch.tensor(dataset.x_sections[0].slice_based_before.ys), dtype=torch.float32)
    mid_idx = 224 // 2
    curr_base = base_input

    dataset.labels.append(0)

    with torch.no_grad():
        for STEP in range(len(dataset)):
            dataset.predictions.append(curr_base.numpy())

            curr_torch = dataset.torchpositions[STEP]

            # if STEP == 16:
            #     print("taking ground truth input for step 16")
            #     ground_truth_base = torch.tensor(dataset.ground_truth[STEP], dtype=torch.float32).clone()
            #     offset = ground_truth_base[200:220].mean()
            #     ground_truth_base = ground_truth_base - offset
            #     curr_base = ground_truth_base

            curr_input = curr_base[curr_torch - 112 : curr_torch + 112].unsqueeze(0).clone()
            # predict footprint
            if FOOTPRINT_PREDICT:
                footprint, fp_uncert = footprint_predictor.predict_with_uncertainty(
                    curr_input, num_samples=N_PREDICTIONS
                )
                left_fp = footprint[:, 0].int().item()
                right_fp = footprint[:, 1].int().item()
                uncertainty_score = np.sum(fp_uncert.detach().cpu().numpy())
                print(f"footprint-uncertainty-score@{STEP}: {uncertainty_score}")

            else:
                true_fp = dataset.true_footprints[STEP]
                left_fp = 112 - np.abs(curr_torch - true_fp.left_idx)
                right_fp = 112 + np.abs(curr_torch - true_fp.right_idx)
                footprint = torch.tensor([left_fp, right_fp], dtype=torch.float32).unsqueeze(0)
                print(f"footprint-uncertainty-score@{STEP}: 0.0 (true footprint)")

            z_offset = curr_input[:, 112].clone().item()
            curr_input -= z_offset

            # predict shape
            segment = curr_input[:, left_fp:right_fp]
            len_segment = segment.size(1)
            xs_original = np.arange(0, len_segment) / 10
            f = interp1d(xs_original, segment, kind="linear")
            xs_90 = np.linspace(0, (len_segment - 1) / 10, 90)
            curr_input_segment = f(xs_90)

            # add gaussian noise to curr_input_segment
            # if STEP > 0:
            #     noise = 0.0 * np.random.randn(90)
            #     noise[0] = 0
            #     noise[-1] = 0
            #
            #     curr_input_segment += noise

            pred, uncertainty = shape_predictor.predict_with_uncertainty(
                torch.tensor(curr_input_segment, dtype=torch.float32).view(-1, 90),
            )
            pred = pred.detach().cpu().squeeze().numpy()
            uncertainty_score = np.sum(uncertainty.detach().cpu().numpy())
            print(f"shape-uncertainty-score@{STEP}: {uncertainty_score}")

            if uncertainty_score >= U_THRESHOLD and STEP < len(dataset) - 1:
                print("taking ground truth prediction")
                ground_truth_base = torch.tensor(dataset.ground_truth[STEP + 1], dtype=torch.float32).clone()
                offset = ground_truth_base[180:220].mean()
                ground_truth_base = ground_truth_base - offset
                dataset.labels.append(1)
                curr_base = ground_truth_base

            else:
                dataset.labels.append(0)

                f = interp1d(xs_90, pred, kind="linear")
                shape_prediction = f(xs_original)
                shape_prediction += z_offset

                next_base = curr_base.clone()
                next_base[curr_torch - (mid_idx - left_fp) : curr_torch + (right_fp - mid_idx)] = torch.tensor(
                    shape_prediction.copy(), dtype=torch.float32
                )
                curr_base = next_base.clone()

            print("\n")

        # dataset.predictions.append(curr_base.numpy())
        # dataset.labels.append(0)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        Plotter.plot_e2e(
            dataset.predictions, dataset.labels, dataset.ground_truth, title=f"E2E uncert:{U_THRESHOLD}, {now}"
        )


if __name__ == "__main__":
    run_e2e_prediction(None, DEVICE="cpu", FOOTPRINT_PREDICT=True, U_THRESHOLD=1000.0)
