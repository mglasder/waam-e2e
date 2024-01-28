import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from scipy.interpolate import interp1d

from e2e.data.resampled import ResampledE2EDataset
from e2e.helpers import timing
from e2e.helpers.geometry import find_nearest_point
from e2e.helpers.resample import interp_equidistant, interp_xsampling
from e2e.models.modelV2 import ModelPoints
from fp.models.mlp import ResMLPpoints
from fp.models.model import Model

timing.ENABLE_TIMING = False
from e2e.models.recurrent import ShapePointsModel
from e2e.prediction.end_to_end import Plotter

f32 = torch.float32


def interp_and_torch_stack(
    interp_func: Callable,
    curve: Union[np.ndarray, torch.tensor],
    num_points: int,
) -> torch.tensor:

    x_re, z_re = interp_func(x=curve[1, :], y=curve[0, :], num_points=num_points)
    curve_re = torch.stack([torch.tensor(z_re, dtype=f32), torch.tensor(x_re, dtype=f32)])

    return curve_re


def run_e2e_prediction(DEVICE="cpu"):
    # VM_DATA_DIR = Path("/home/magnus/datasets/waam/30_processing_results/ImageGenerator")
    MAC_DATA_DIR = Path("/Users/magnus/datasets/WAAM/test")

    footprint_points_model = ResMLPpoints(
        p=0.0,
        n_input_features=224,
        n_output_features=2,
    )

    shape_points_model = ShapePointsModel(
        p=0.0,
        n_input_features=50,
        n_output_features=4,
    )

    # instantiate models
    # TODO: always add wandb id
    repos = Path("/Users/magnus/repos/")
    footprint_path = repos / Path(
        "waam-footprint/fp/waam-footprint-ps-idx/byl1e1dk/checkpoints/epoch=84-step=425.ckpt",  # zany-dragon-29
        # "waam-footprint/fp/waam-footprint-ps-idx/ykaeuujz/checkpoints/epoch=12-step=39.ckpt",  # devout-eon-46 (resampled points input)
    )

    footprint_predictor = Model.load_from_checkpoint(
        model=footprint_points_model, checkpoint_path=footprint_path, map_location=torch.device(DEVICE)
    )

    # shape
    # TODO: always add wandb id
    shape_points_model_path = repos / Path(
        # "waam-e2e/e2e/waam-e2e-shape-points/5ty9skjv/checkpoints/epoch=99-step=900.ckpt" # spring-haze-73
        # "waam-e2e/e2e/waam-e2e-shape-points/3wum6bz5/checkpoints/epoch=199-step=1000.ckpt"  # sage-darkness-97
        # "waam-e2e/e2e/waam-e2e-shape-points/lroch2o3/checkpoints/epoch=179-step=900.ckpt"  # pretty-salad-110 (bezier)
        "waam-e2e/e2e/waam-e2e-shape-points/oghf9rs9/checkpoints/epoch=179-step=900.ckpt",  # ... (bezier+area)
    )
    shape_predictor = ModelPoints.load_from_checkpoint(
        model=shape_points_model,
        checkpoint_path=shape_points_model_path,
        map_location=torch.device(DEVICE),
    )

    # get data
    dataset = ResampledE2EDataset(mirror=False, segment_length=224)
    v_seam_toolpath = np.loadtxt("/Users/magnus/repos/WAAM-process-model/v-seam-toolpath-x-idx.txt", delimiter=",")
    v_seam_substrate = np.loadtxt(
        "/Users/magnus/repos/WAAM-process-model/v-seam-substrate-resampled.txt", delimiter=","
    )

    rect_block_substrate = np.loadtxt("/Users/magnus/repos/WAAM-process-model/rect-block-substrate.txt", delimiter=",")

    rect_block_toolpath = np.loadtxt(
        "/Users/magnus/repos/WAAM-process-model/rect-block-toolpath-x-idx.txt", delimiter=","
    )

    # set input
    torchpositions = v_seam_toolpath.astype("int")
    base_input = v_seam_substrate[:, 1]

    # some setup
    dataset.torchpositions = torchpositions
    dataset.ids = list(range(len(dataset.torchpositions)))
    mid_idx = 224 // 2
    len_base = len(base_input)

    # TODO: make sure x resample is consistent and accurate
    dataset.labels.append(0)
    xs_sample_substrate = np.linspace(0, (len_base - 1) / 10, len_base)

    curr_base = torch.tensor(np.array([base_input.flatten(), xs_sample_substrate]), dtype=torch.float32)
    curr_base[1, :] -= curr_base[1, 0].clone()

    f32 = torch.float32

    with torch.no_grad():
        for STEP in range(len(dataset)):
            tic = time.time()
            dataset.predictions.append(curr_base.numpy())

            curr_torch = dataset.torchpositions[STEP]
            curr_W = curr_base[:, curr_torch - 112 : curr_torch + 112].clone()

            shift_x = curr_W[1, 0].clone()
            shift_z = curr_W[0, 112].clone()
            curr_W[1, :] -= shift_x
            curr_W[0, :] -= shift_z

            # predict footprint
            footprint = footprint_predictor.forward(curr_W)
            left_fp = footprint[0, 0, 0].int().item()
            right_fp = footprint[0, 0, 1].int().item()
            dataset.fp_predictions.append(np.array([left_fp, right_fp]))
            print(f"pred. footprint @ {STEP}: {left_fp}, {right_fp}")

            curr_W[1, :] += shift_x
            curr_W[0, :] += shift_z

            # predict shape
            F_hat = curr_W[:, left_fp:right_fp].clone()
            shift_x = curr_W[1, 112].clone()
            shift_z = curr_W[0, 112].clone()
            F_hat[1, :] -= shift_x
            F_hat[0, :] -= shift_z

            F_hat_re = interp_and_torch_stack(interp_equidistant, F_hat, num_points=50)
            S_hat = shape_predictor.forward(F_hat_re.unsqueeze(0)).squeeze()
            S_hat_re = interp_and_torch_stack(interp_equidistant, S_hat, num_points=F_hat.size(1))

            S_hat_re[1, :] += shift_x
            S_hat_re[0, :] += shift_z

            # update workpiece
            next_base = curr_base.clone()
            next_base[:, curr_torch - (mid_idx - left_fp) : curr_torch + (right_fp - mid_idx)] = S_hat_re

            # resample equidistant
            next_base_re = interp_and_torch_stack(interp_xsampling, next_base, num_points=len_base).squeeze()
            curr_base = next_base_re.clone()

            toc = time.time()
            print(f"prediction-time @ {STEP}: {(toc - tic)*1000:.4f} ms")
            dataset.prediction_times.append(toc - tic)
            dataset.labels.append(0)

        dataset.predictions.append(curr_base.numpy())

        print(f"total time: {sum(dataset.prediction_times)*1000:.3f} ms")
        return dataset


if __name__ == "__main__":
    dataset = run_e2e_prediction(DEVICE="cpu")

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    Plotter.plot_e2e_points(dataset.predictions, dataset.labels, None, title=f"E2E v-groove, {now}")
