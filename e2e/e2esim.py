import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

from e2e.data.resampled import ResampledE2EDataset
from e2e.helpers import timing
from e2e.models.modelV2 import ModelPoints
from fp.models.mlp import ResMLPpoints
from fp.models.model import Model

timing.ENABLE_TIMING = False
from e2e.models.recurrent import ShapePointsModel
from e2e.prediction.end_to_end import Plotter


def run_e2e_prediction(DEVICE="cpu"):
    # VM_DATA_DIR = Path("/home/magnus/datasets/waam/30_processing_results/ImageGenerator")
    MAC_DATA_DIR = Path("/Users/magnus/datasets/WAAM/test")

    N_PREDICTIONS = 30

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
    # TODO: wandb id: zany-dragon-29
    repos = Path("/Users/magnus/repos/")
    footprint_path = repos / Path("waam-footprint/fp/waam-footprint-ps-idx/byl1e1dk/checkpoints/epoch=84-step=425.ckpt")

    footprint_predictor = Model.load_from_checkpoint(
        model=footprint_points_model, checkpoint_path=footprint_path, map_location=torch.device(DEVICE)
    )

    # shape
    # TODO: wandb id: spring-haze-73
    shape_points_model_path = repos / Path(
        "waam-e2e/e2e/waam-e2e-shape-points/5ty9skjv/checkpoints/epoch=99-step=900.ckpt"
    )
    shape_predictor = ModelPoints.load_from_checkpoint(
        model=shape_points_model, checkpoint_path=shape_points_model_path, map_location=torch.device(DEVICE)
    )
    shape_predictor.to("cpu")

    # load dataset
    # data_loader = SampleLoader(sample_dir=MAC_DATA_DIR)
    # cross_section_samples = data_loader.load(which=[EXP.RANDOM_EX6])
    dataset = ResampledE2EDataset(mirror=False, segment_length=224)

    v_seam_toolpath = np.loadtxt("/Users/magnus/repos/WAAM-process-model/v-seam-toolpath-x-idx.txt", delimiter=",")
    v_seam_substrate = np.loadtxt(
        "/Users/magnus/repos/WAAM-process-model/v-seam-substrate-resampled.txt", delimiter=","
    )

    dataset.torchpositions = v_seam_toolpath.astype("int")
    dataset.ids = list(range(len(v_seam_toolpath)))

    base_input = v_seam_substrate[:, 1]
    mid_idx = 224 // 2
    curr_base = torch.tensor(base_input)

    dataset.labels.append(0)
    xs_sample_footprint = np.linspace(0, (224 - 1) / 10, 224)
    xs_sample_shape = np.linspace(0, (50 - 1) / 10, 50)

    with torch.no_grad():
        for STEP in range(len(dataset)):
            dataset.predictions.append(curr_base.numpy())

            curr_torch = dataset.torchpositions[STEP]
            curr_input = curr_base[curr_torch - 112 : curr_torch + 112].unsqueeze(0).clone()
            # predict footprint
            curr_input_points = torch.tensor(
                np.array([curr_input.numpy().flatten(), xs_sample_footprint]), dtype=torch.float32
            )

            tic = time.time()

            footprint = footprint_predictor.forward(curr_input_points)
            left_fp = footprint[0, 0, 0].int().item()
            right_fp = footprint[0, 0, 1].int().item()
            dataset.fp_predictions.append(np.array([left_fp, right_fp]))

            z_offset = curr_input[:, 112].clone().item()
            curr_input -= z_offset

            # predict shape
            segment = curr_input[:, left_fp:right_fp]
            len_segment = segment.size(1)
            xs_original = np.arange(0, len_segment) / 10
            f = interp1d(xs_original, segment, kind="linear")
            xs_90 = np.linspace(0, (len_segment - 1) / 10, 50)
            curr_input_segment = f(xs_90)

            curr_input_segment_points = torch.tensor(
                np.array([curr_input_segment.flatten(), xs_sample_shape]), dtype=torch.float32
            )

            pred = shape_predictor.forward(curr_input_segment_points.view(-1, 1, 100))
            toc = time.time()
            pred = pred.detach().cpu().squeeze().numpy()
            print(f"prediction-time@{STEP}: {toc - tic}")

            dataset.prediction_times.append(toc - tic)
            dataset.labels.append(0)

            pred_z = pred[0, :]
            pred_x = pred[1, :]
            f = interp1d(pred_x, pred_z, kind="linear")
            xs_resample = np.linspace(pred[1, 0], pred[1, -1], len_segment)
            shape_prediction = f(xs_resample)
            shape_prediction += z_offset

            next_base = curr_base.clone()
            next_base[curr_torch - (mid_idx - left_fp) : curr_torch + (right_fp - mid_idx)] = torch.tensor(
                shape_prediction.copy(), dtype=torch.float32
            )
            curr_base = next_base.clone()

        dataset.predictions.append(curr_base.numpy())

        print(f"total time: {sum(dataset.prediction_times)*1000:.27} ms")
        return dataset


if __name__ == "__main__":
    dataset = run_e2e_prediction(DEVICE="cpu")

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    Plotter.plot_e2e(dataset.predictions, dataset.labels, None, title=f"E2E v-groove, {now}")
