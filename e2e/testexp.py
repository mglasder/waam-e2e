from pathlib import Path

import numpy as np
import torch

from e2e.data.dataset import ShapeDataset
from e2e.helpers import timing

timing.ENABLE_TIMING = True
from e2e.data.loader import EXPERIMENT as EXP
from e2e.models.recurrent import LSTM
from e2e.prediction.end_to_end import DataLoader, ModelHandler, Plotter, Predictor

PROJECT_NAME = "waam-e2e-pre"
RUN_ID = "95fanoij"

# "oqfrcpcw"
# "oqfrcpcw"
# "isn0h9ws"  # first_in_row + 2x
# "xtuq679p"    # first_in_row + 4x
VERSION = "best"
VM_DATA_DIR = Path("/home/magnus/datasets/waam/30_processing_results/ImageGenerator")
INPUT_LENGTH = 90
TARGET_LENGTH = 90
P = 0.5


def main():
    lstm = LSTM(
        p=P, n_input_features=INPUT_LENGTH, n_output_features=TARGET_LENGTH, n_hidden=TARGET_LENGTH * 3, n_layers=10
    )
    data_loader = DataLoader(VM_DATA_DIR)
    cross_section_samples = data_loader.load_samples([EXP.RANDOM_EX6])
    dataset = ShapeDataset(mirror=False, segment_length=TARGET_LENGTH).create(cross_section_samples)

    model_handler = ModelHandler(PROJECT_NAME, RUN_ID, VERSION, lstm=lstm)
    predictor = Predictor(TARGET_LENGTH, model_handler, cross_section_samples)

    init_input = torch.tensor(dataset[0][0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    UNCERT_THRESH = 10.0
    predictions, labels, ground_truth = predictor.predict(init_input, uncertainty_threshold=UNCERT_THRESH, mode="e2e")
    Plotter.plot_e2e(predictions, labels, ground_truth, title=f"E2E uncert:{UNCERT_THRESH}")

    print(f"fallback rate to alternative prediction: {np.sum(labels)/len(labels)*100:.2f}%")

    # predictions, labels, ground_truth = predictor.predict(init_input, uncertainty_threshold=4.5, mode="hybrid")
    # Plotter.plot_e2e(predictions, labels, ground_truth, title="Hybrid")


if __name__ == "__main__":
    main()
