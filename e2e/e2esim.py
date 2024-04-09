import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Union
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from e2e.data.resampled import ResampledE2EDataset
from e2e.helpers import timing
from e2e.helpers.geometry import find_nearest_point
from e2e.helpers.resample import interp_equidistant, interp_xsampling
from e2e.models.modelV2 import ModelPoints
from fp.models.mlp import MlpRelativeDistance
from fp.models.model import Model

timing.ENABLE_TIMING = False
from e2e.models.recurrent import ShapePointsModel
from e2e.prediction.end_to_end import Plotter
from scipy.spatial.distance import cdist
from e2e.data.sample import CrossSectionSample

f32 = torch.float32


def interp_and_torch_stack(
    interp_func: Callable,
    curve: Union[np.ndarray, torch.tensor],
    num_points: int,
) -> torch.tensor:

    x_re, z_re = interp_func(x=curve[1, :], y=curve[0, :], num_points=num_points)
    curve_re = torch.stack([torch.tensor(z_re, dtype=f32), torch.tensor(x_re, dtype=f32)])

    return curve_re


def get_index_euclidean(point: np.ndarray, list_of_points: np.ndarray) -> int:
    distances = cdist(point, list_of_points, "euclidean")
    idx = np.argmin(distances, axis=1)
    return idx[0]


def get_index_vertical_intersection(torch_x: float, list_of_points: np.ndarray) -> int:
    """
    Find the intersection of a series of points and a vertical line.

    :param list_of_points: numpy array representing points on the curve
    :param torch_x: x-coordinate of the vertical line
    :return: index of the intersection point in the curve
    """

    # Extract x-coordinates from points
    x_coords = list_of_points[:, 0]

    # Find the index of the point closest to the vertical line
    idx = np.abs(x_coords - torch_x).argmin()

    return idx


def get_footprint_index_from(radii, substrate_points):
    radius_left, radius_right = radii[0], radii[1]

    distances_left = torch.norm(substrate_points[:, substrate_points[1, :] <= 0], dim=0)
    distances_right = torch.norm(substrate_points[:, substrate_points[1, :] > 0], dim=0)

    differences_left = torch.abs(distances_left - radius_left)
    differences_right = torch.abs(distances_right - radius_right)

    _, left_idx = torch.topk(differences_left, 1, largest=False)
    _, right_idx = torch.topk(differences_right, 1, largest=False)
    right_idx += len(differences_left)

    return left_idx.item(), right_idx.item()


def get_xz_correction(substrate_points):
    shift_x = substrate_points[1, 112].clone()
    shift_z = substrate_points[0, 112].clone()
    return shift_x, shift_z


def run_e2e_prediction(experiment, DEVICE="cpu"):
    # VM_DATA_DIR = Path("/home/magnus/datasets/waam/30_processing_results/ImageGenerator")
    # MAC_DATA_DIR = Path("/Users/magnus/datasets/WAAM/test")

    print(f"Simulating experiment: {experiment} end-to-end.")

    footprint_points_model = MlpRelativeDistance(
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

    FOOTPRINT_MODEL_ID = "iiy649xa"

    repos = Path("/Users/magnus/repos/waam-footprint/fp/")
    footprint_path = (
        repos
        # / "waam-footprint/fp/waam-footprint-ps-idx"
        / (
            # "waam-footprint/fp/waam-footprint-ps-idx/byl1e1dk/checkpoints/epoch=84-step=425.ckpt",  # zany-dragon-29
            # "waam-footprint/fp/waam-footprint-ps-idx/ykaeuujz/checkpoints/epoch=12-step=39.ckpt",  # devout-eon-46 (resampled points input)
            # "waam-footprint/fp/waam-footprint-ps-idx/52rebqh8/checkpoints/epoch=55-step=168.ckpt"  # likely-shape-78 (xsampling, torch=(0,0), radius)
            # "5cqux5la/checkpoints/epoch=53-step=216.ckpt"  # dazzling-sponge-107 (xsamplig, ", finetuneing v-groove)
            # "2b1grw9n/checkpoints/epoch=74-step=300.ckpt"  # vital-universe-108 (same as above, but works better for some reason)
            f"waam-footprint-radii-equidistant/{FOOTPRINT_MODEL_ID}/checkpoints/epoch=75-step=304.ckpt"  # astral-dragon-39, finetuned on cubes sim
        )
    )

    footprint_predictor = Model.load_from_checkpoint(
        model=footprint_points_model, checkpoint_path=footprint_path, map_location=torch.device(DEVICE)
    )

    # shave footprint predictor as pickle
    with open(f"../../waam-rl/footprint_predictor_{FOOTPRINT_MODEL_ID}.pickle", "wb") as f:
        pickle.dump(footprint_predictor, f)

    # load again
    with open(f"../../waam-rl/footprint_predictor_{FOOTPRINT_MODEL_ID}.pickle", "rb") as f:
        footprint_predictor = pickle.load(f)

    # shape
    # TODO: always add wandb id
    shape_points_model_path = (
        repos
        / "waam-e2e/e2e"
        / Path(
            # "waam-e2e/e2e/waam-e2e-shape-points/lroch2o3/checkpoints/epoch=179-step=900.ckpt"  # pretty-salad-110 (bezier)
            # "waam-e2e-shape-points/l2973txr/checkpoints/epoch=139-step=700.ckpt"  # royal-thunder-140 (bezier+area)
            # "waam-e2e/e2e/waam-e2e-shape-points/wxoyse1p/checkpoints/epoch=199-step=1400.ckpt"  # dauntless-cherry-146 (bezier+area, finetuned v-groove)
            "waam-e2e-shape-points/pzxni5gu/checkpoints/epoch=159-step=1120.ckpt"  # pretty-spaceship-147 (", area: 11.4)
            # "waam-e2e-shape-points/wm0fngvp/checkpoints/epoch=199-step=1400.ckpt"  # dandy-frost-148 (", 11.2
        )
    )
    shape_predictor = ModelPoints.load_from_checkpoint(
        model=shape_points_model,
        checkpoint_path=shape_points_model_path,
        map_location=torch.device(DEVICE),
    )

    with open("../../waam-rl/shape_predictor_finetuned_114.pickle", "wb") as f:
        pickle.dump(shape_predictor, f)

    # load again
    with open("../../waam-rl/shape_predictor_finetuned_114.pickle", "rb") as f:
        shape_predictor = pickle.load(f)

    # get data
    dataset = ResampledE2EDataset(mirror=False, segment_length=224)
    v_seam_toolpath = np.loadtxt(
        f"/Users/magnus/repos/waam-eval/toolpaths/for_e2e/{experiment}-toolpath-real-coords.txt", delimiter=","
    )

    substrate_type = ("-").join(experiment.split("-")[:-1])
    v_seam_substrate = np.loadtxt(
        f"/Users/magnus/repos/waam-eval/substrates/{substrate_type}-substrate.txt", delimiter=","
    )

    # rect_block_substrate = np.loadtxt("/Users/magnus/repos/WAAM-process-model/rect-block-substrate.txt", delimiter=",")
    #
    # rect_block_toolpath = np.loadtxt(
    #     "/Users/magnus/repos/WAAM-process-model/rect-block-toolpath-x-idx.txt", delimiter=","
    # )

    # load first measurement after
    folder = Path("/Users/magnus/datasets/WAAM/v-groove-results/CrossSectionTorchHeightBasing")
    files = list(folder.glob("*.pickle"))

    samples = [CrossSectionSample.read_file(f) for f in files]
    samples = sorted(samples, key=lambda sample: sample.bead_id)
    # first = np.array(samples[0].slice_aligned_after.points).reshape(-1, 2)
    # first[:, 0] -= 100 - 1.45
    # first[:, 1] += 1.61
    # first_input = np.array([first[:, 1], first[:, 0]])

    # set input
    torchpositions = v_seam_toolpath  # np.repeat(500, 10).astype("int")
    base_input = v_seam_substrate[:, 1]

    # print(torchpositions)

    # some setup
    dataset.torchpositions = torchpositions[:]
    dataset.ids = list(range(len(dataset.torchpositions)))
    mid_idx = 224 // 2
    len_base = len(base_input)

    dataset.labels.append(0)
    # xs_sample_substrate = np.linspace(0, (len_base - 1) / 10, len_base)

    curr_base = torch.tensor(np.array([v_seam_substrate[:, 1], v_seam_substrate[:, 0]]), dtype=torch.float32)

    # left_edge_idx = get_index_vertical_intersection(first[0, 0], curr_base.T.flip(dims=(1,)).clone().cpu().numpy())
    # right_edge_idx = get_index_vertical_intersection(first[-1, 0], curr_base.T.flip(dims=(1,)).clone().cpu().numpy())

    # curr_base_new = torch.tensor(
    #     np.concatenate(
    #         [
    #             curr_base[:, :left_edge_idx],
    #             first_input,
    #             curr_base[:, right_edge_idx:],
    #         ],
    #         axis=1,
    #     ),
    #     dtype=torch.float32,
    # ).clone()
    #
    # curr_base = interp_and_torch_stack(interp_xsampling, curr_base_new, num_points=len_base).squeeze().clone()

    with torch.no_grad():
        for STEP in range(len(dataset)):
            tic = time.time()
            dataset.predictions.append(curr_base.numpy())

            curr_torch = dataset.torchpositions[STEP]
            curr_torch_idx = get_index_vertical_intersection(
                curr_torch[0], curr_base.T.flip(dims=(1,)).clone().cpu().numpy()
            )
            dataset.torchpositions_idx.append(curr_torch_idx)
            curr_W = curr_base[:, curr_torch_idx - 112 : curr_torch_idx + 112].clone()

            # curr_W = interp_and_torch_stack(interp_xsampling, curr_W, 224)

            # shift curr_W to torch position at (0,0)
            shift_x, shift_z = get_xz_correction(curr_W)
            curr_W[1, :] -= shift_x
            curr_W[0, :] -= shift_z

            # predict footprint
            radii = footprint_predictor.forward(curr_W).squeeze()
            left_fp, right_fp = get_footprint_index_from(radii, curr_W)
            F_hat = curr_W[:, left_fp:right_fp].clone()

            fpl, fpr = curr_W[:, left_fp].cpu().numpy(), curr_W[:, right_fp].cpu().numpy()

            fpl[1] += shift_x
            fpl[0] += shift_z

            fpr[1] += shift_x
            fpr[0] += shift_z

            dataset.fp_predictions.append([fpl, fpr])
            print(f"pred. footprint @ {STEP}: {left_fp}, {right_fp}")

            # predict shape
            F_hat_re = interp_and_torch_stack(interp_equidistant, F_hat, num_points=50).unsqueeze(0)
            S_hat = shape_predictor.forward(F_hat_re).squeeze()
            S_hat_re = interp_and_torch_stack(interp_xsampling, S_hat, num_points=F_hat.size(1))

            # shift S_hat to coordinate system of substrate
            S_hat_re[1, :] += shift_x
            S_hat_re[0, :] += shift_z

            # update workpiece
            next_base = curr_base.clone()
            next_base[:, curr_torch_idx - (mid_idx - left_fp) : curr_torch_idx + (right_fp - mid_idx)] = S_hat_re

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
    EXPERIMENT = "v-45-A"

    dataset = run_e2e_prediction(experiment=EXPERIMENT, DEVICE="cpu")

    # with open(f"{EXPERIMENT}-sim.pickle", "wb") as f:
    #     pickle.dump(dataset, f)
    #     print(f"saved dataset to {f.name}")

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    Plotter.plot_e2e_points(
        dataset.predictions,
        dataset.labels,
        dataset.torchpositions_idx,
        dataset.torchpositions,
        title=f"E2E {EXPERIMENT}, {now}",
    )
