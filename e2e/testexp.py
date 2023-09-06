from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from e2e.data.dataset import ShapeDataset
from e2e.data.loader import SampleLoader
from e2e.helpers import timing

timing.ENABLE_TIMING = True
from e2e.data.loader import EXPERIMENT as EXP
from e2e.helpers.wandb import download_model
from e2e.models.modelV2 import ModelV2
from e2e.models.recurrent import LSTM

PROJECT_NAME = "waam-e2e-pre"
RUN_ID = "1v8m0ien"
VERSION = "best"
VM_DATA_DIR = Path("/home/magnus/datasets/waam/30_processing_results/ImageGenerator")
INPUT_LENGTH = 90
TARGET_LENGTH = 90
P = 0.5


HL = TARGET_LENGTH // 2


def debug_input_plot(x):
    fig, ax = plt.subplots(figsize=(10, 10))
    x_ = np.arange(0, len(x)) / 10
    ax.plot(x_, x)
    ax.set_aspect("equal")
    plt.show()


def update_curr_cross_section_with_pred(curr_cross_section_sample, pred):
    ys_before = curr_cross_section_sample.slice_based_before.ys
    ys_predicted = ys_before

    # update current cross section with prediction
    curr_torch_idx = curr_cross_section_sample.torchposition.global_y_idx
    ys_predicted[curr_torch_idx - HL : curr_torch_idx + HL] = pred

    return ys_predicted


def get_next_input(cross_section_predicted, next_cross_section_sample):
    next_torch_idx = next_cross_section_sample.torchposition.global_y_idx
    next_input = cross_section_predicted[next_torch_idx - HL : next_torch_idx + HL]
    return torch.tensor(next_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def main():
    model_path = download_model(project=PROJECT_NAME, run_id=RUN_ID, version=VERSION)
    lstm = LSTM(
        p=P, n_input_features=INPUT_LENGTH, n_output_features=TARGET_LENGTH, n_hidden=TARGET_LENGTH * 3, n_layers=10
    )
    model = ModelV2.load_from_checkpoint(model=lstm, checkpoint_path=model_path)
    model.eval()

    loader = SampleLoader(VM_DATA_DIR)
    cross_section_samples = loader.load(which=[EXP.RANDOM_EX6])
    cross_section_samples = sorted(cross_section_samples, key=lambda x: x.bead_id)
    dataset = ShapeDataset(mirror=False, segment_length=TARGET_LENGTH).create(cross_section_samples)

    curr_input = torch.tensor(dataset[0][0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    predictions = []
    labels = []

    for i in range(len(dataset) - 1):
        print(f"Predicting cross section {i+1} of {len(dataset) - 1}")

        curr_cross_section_sample = cross_section_samples[i]
        next_cross_section_sample = cross_section_samples[i + 1]

        # debug_input_plot(curr_input.detach().cpu().numpy())

        pred, uncertainty = model.predict_with_uncertainty(curr_input.to(model.device), num_samples=150)
        uncertainty_score = np.sum(uncertainty.detach().cpu().numpy())
        print(f"Uncertainty: {uncertainty_score}")
        pred = pred.detach().cpu().numpy()

        cross_section_predicted = update_curr_cross_section_with_pred(curr_cross_section_sample, pred)

        # HACK: if first bead in row, use actual input
        # is_first_in_row = (curr_cross_section_sample.welding_params["weld_bead_nr"] == 1) and (i > 0)
        if uncertainty_score > 3:
            cross_section_predicted = curr_cross_section_sample.slice_based_after.ys
            labels.append(1)
        else:
            labels.append(0)

        next_input = get_next_input(cross_section_predicted, next_cross_section_sample)

        predictions.append(cross_section_predicted)
        curr_input = next_input

    # TODO: plot e2e predictions and compare to ground truth
    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    for i in range(len(predictions)):
        pred = predictions[i]
        z_offset = np.mean(pred[100:200])
        y = pred - z_offset
        x = np.arange(0, len(y)) / 10

        color = "black" if labels[i] == 1 else "blue"

        ax.plot(x, y, color=color)
        ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
