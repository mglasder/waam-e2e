from pathlib import Path

from e2e.data.dataset import ShapeDataset
from e2e.helpers import timing

timing.ENABLE_TIMING = True
from e2e.data.loader import EXPERIMENT as EXP
from e2e.models.recurrent import LSTM
from e2e.prediction.end_to_end import DataLoader, ModelHandler, Plotter, Predictor

PROJECT_NAME = "waam-e2e-pre"
RUN_ID = "xtuq679p"
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
    predictor = Predictor(TARGET_LENGTH, model_handler)

    predictions, labels, ground_truth = predictor.predict(cross_section_samples, dataset)
    Plotter.plot_e2e(predictions, labels, ground_truth)


if __name__ == "__main__":
    main()
