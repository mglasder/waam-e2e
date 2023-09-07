import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from e2e.data.loader import SampleLoader
from e2e.helpers.wandb import download_model
from e2e.models.modelV2 import ModelV2

# Assuming the necessary imports are present


class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_samples(self, which):
        loader = SampleLoader(self.data_dir)
        samples = loader.load(which=which)
        return sorted(samples, key=lambda x: x.bead_id)


class ModelHandler:
    def __init__(self, project_name, run_id, version, lstm: nn.Module):
        self.model_path = download_model(project=project_name, run_id=run_id, version=version)
        self.model = ModelV2.load_from_checkpoint(model=lstm, checkpoint_path=self.model_path)
        self.model.eval()

    def predict_with_uncertainty(self, input_data):
        return self.model.predict_with_uncertainty(input_data.to(self.model.device), num_samples=150)


class Plotter:
    @staticmethod
    def debug_input_plot(x, title: str):
        fig, ax = plt.subplots(figsize=(10, 10))
        x_ = np.arange(0, len(x)) / 10
        ax.plot(x_, x)
        ax.set_aspect("equal")
        ax.set_title(f"{title}")
        plt.show()

    @staticmethod
    def plot_e2e(predictions, labels, ground_truth):
        fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
        for i in range(len(predictions)):
            pred = predictions[i]
            z_offset = np.mean(pred[130:150])
            y = pred - z_offset
            x = np.arange(0, len(y)) / 10
            color = "black" if labels[i] == 1 else "blue"
            ax.plot(x, y, color=color, alpha=1, linewidth=0.5)

            y_true = ground_truth[i] - z_offset
            ax.plot(x, y_true, color="green", alpha=0.5, linewidth=0.5)

            ax.set_aspect("equal")
        plt.show()


class Predictor:
    def __init__(self, output_length, model_handler):
        self.model_handler = model_handler
        self.output_length = output_length
        self.hl = output_length // 2

        self.STEP = 0

        # init during setup
        self.torchpositions = None
        self.curr_input = None
        self.LEN = None

        # prediction history
        self.ground_truth = []
        self.predictions = []
        self.labels = []

    def setup(self, cross_sections, dataset):
        self.torchpositions = [s.torchposition.global_y_idx for s in cross_sections]
        self.curr_input = torch.tensor(dataset[0][0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.LEN = len(dataset)

    def _get_next_input(self, cross_section_predicted):
        next_torch_idx = self.torchpositions[self.STEP + 1]

        next_input = cross_section_predicted[next_torch_idx - self.hl : next_torch_idx + self.hl]
        h0 = next_input[self.hl]
        next_input -= h0
        return torch.tensor(next_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def _update_curr_cross_section_with_pred(self, curr_cross_section_sample, pred):
        ys_predicted = curr_cross_section_sample.slice_based_before.ys

        curr_torch_idx = self.torchpositions[self.STEP]
        ys_predicted[curr_torch_idx - self.hl : curr_torch_idx + self.hl] = pred
        return ys_predicted

    def _handle_prediction(self, curr_input, curr_cross_section_sample, next_cross_section_sample=None):
        pred, uncertainty = self.model_handler.predict_with_uncertainty(curr_input)
        uncertainty_score = np.sum(uncertainty.detach().cpu().numpy())
        pred = pred.detach().cpu().numpy()

        cross_section_predicted = self._update_curr_cross_section_with_pred(curr_cross_section_sample, pred)

        if uncertainty_score > 4.5:
            cross_section_predicted = curr_cross_section_sample.slice_based_after.ys
            label = 1
        else:
            label = 0

        if next_cross_section_sample:
            next_input = self._get_next_input(cross_section_predicted.copy())
            curr_input = next_input

        return cross_section_predicted, label, curr_input

    def predict(self, cross_section_samples):
        for i in range(self.LEN):
            self.STEP = i

            curr_cross_section_sample = cross_section_samples[i]
            next_cross_section_sample = cross_section_samples[i + 1] if i != (self.LEN - 1) else None

            cross_section_predicted, label, curr_input = self._handle_prediction(
                self.curr_input, curr_cross_section_sample, next_cross_section_sample
            )

            self.ground_truth.append(curr_cross_section_sample.slice_based_before.ys)
            self.predictions.append(cross_section_predicted)
            self.labels.append(label)
            self.curr_input = curr_input

        return self.predictions, self.labels, self.ground_truth
