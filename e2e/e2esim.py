import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

from e2e.data.resampled import ResampledE2EDataset
from e2e.helpers import timing
from e2e.helpers.geometry import find_nearest_point
from e2e.helpers.resample import interp_equidistant
from e2e.models.modelV2 import ModelPoints
from fp.models.mlp import ResMLPpoints
from fp.models.model import Model

timing.ENABLE_TIMING = False
from e2e.models.recurrent import ShapePointsModel
from e2e.prediction.end_to_end import Plotter


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
        n_output_features=50,
    )

    # instantiate models
    # TODO: always add wandb id
    repos = Path("/Users/magnus/repos/")
    footprint_path = repos / Path(
        "waam-footprint/fp/waam-footprint-ps-idx/byl1e1dk/checkpoints/epoch=84-step=425.ckpt",  # zany-dragon-29
    )

    footprint_predictor = Model.load_from_checkpoint(
        model=footprint_points_model, checkpoint_path=footprint_path, map_location=torch.device(DEVICE)
    )

    # shape
    # TODO: always add wandb id
    shape_points_model_path = repos / Path(
        # "waam-e2e/e2e/waam-e2e-shape-points/5ty9skjv/checkpoints/epoch=99-step=900.ckpt" # spring-haze-73
        "waam-e2e/e2e/waam-e2e-shape-points/2egmfpd6/checkpoints/epoch=179-step=360.ckpt"  # swept-flower-88
    )
    shape_predictor = ModelPoints.load_from_checkpoint(
        model=shape_points_model, checkpoint_path=shape_points_model_path, map_location=torch.device(DEVICE)
    )
    shape_predictor.to("cpu")

    # get data
    dataset = ResampledE2EDataset(mirror=False, segment_length=224)
    v_seam_toolpath = np.loadtxt("/Users/magnus/repos/WAAM-process-model/v-seam-toolpath-x-idx.txt", delimiter=",")
    v_seam_substrate = np.loadtxt(
        "/Users/magnus/repos/WAAM-process-model/v-seam-substrate-resampled.txt", delimiter=","
    )

    # rect_block_substrate = np.loadtxt("/Users/magnus/repos/WAAM-process-model/rect-block-substrate.txt", delimiter=",")
    #
    # rect_block_toolpath = np.loadtxt(
    #     "/Users/magnus/repos/WAAM-process-model/rect-block-toolpath-x-idx.txt", delimiter=","
    # )

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
            left_fp = np.abs(footprint[0, 0, 0].int().item())
            right_fp = np.abs(footprint[0, 0, 1].int().item())
            dataset.fp_predictions.append(np.array([left_fp, right_fp]))
            print(f"pred. footprint @ {STEP}: {left_fp}, {right_fp}")

            curr_W[1, :] += shift_x
            curr_W[0, :] += shift_z

            # predict shape
            F_hat = curr_W[:, left_fp:right_fp].clone()

            shift_x = curr_W[1, curr_torch - left_fp].clone()
            shift_z = curr_W[0, curr_torch - left_fp].clone()

            F_hat[1, :] -= shift_x
            F_hat[0, :] -= shift_z

            len_F = F_hat.size(1)

            x_re, z_re = interp_equidistant(x=F_hat[1, :], y=F_hat[0, :], num_points=50)
            F_hat_re = torch.stack([torch.tensor(z_re, dtype=f32), torch.tensor(x_re, dtype=f32)])

            S_hat = shape_predictor.forward(F_hat_re.view(-1, 1, 100)).squeeze()

            pred_x_re, pred_z_re = interp_equidistant(x=S_hat[1, :], y=S_hat[0, :], num_points=len_F)
            S_hat_re = torch.stack([torch.tensor(pred_z_re, dtype=f32), torch.tensor(pred_x_re, dtype=f32)])

            S_hat_re[1, :] += shift_x
            S_hat_re[0, :] += shift_z

            # update workpiece
            next_base = curr_base.clone()
            next_base[:, curr_torch - (mid_idx - left_fp) : curr_torch + (right_fp - mid_idx)] = S_hat_re

            # resample equidistant
            next_x_re, next_z_re = interp_equidistant(x=next_base[1, :], y=next_base[0, :], num_points=len_base)
            next_base_re = torch.tensor(
                np.array([next_z_re, next_x_re]),
                dtype=torch.float32,
            ).squeeze()

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
