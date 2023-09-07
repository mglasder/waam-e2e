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
    def plot_e2e(predictions, labels, ground_truth, title: str):
        fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
        for i in range(len(predictions)):
            pred = predictions[i]
            y = pred - np.mean(pred[130:150])
            x = np.arange(0, len(y)) / 10
            color = "black" if labels[i] == 1 else "blue"
            ax.plot(x, y, color=color, alpha=1, linewidth=0.5)

            if ground_truth:
                t = ground_truth[i]
                y_true = t - np.mean(t[130:150])
                ax.plot(x, y_true, color="green", alpha=0.5, linewidth=0.5)

        ax.set_title(title)
        ax.set_aspect("equal")
        plt.show()


class Predictor:
    def __init__(self, output_length, model_handler, cross_section_samples):
        self.model_handler = model_handler
        self.output_length = output_length
        self.hl = output_length // 2

        self.x_sections = cross_section_samples

        self.STEP = 0

        self.torchpositions = [s.torchposition.global_y_idx for s in self.x_sections]
        self.ground_truth = [s.slice_based_before.ys for s in self.x_sections]
        self.LEN = len(self.x_sections)

        # init during setup
        self.curr_input = None

        # prediction history
        self.predictions = []
        self.labels = []

    def reset(self, init_input):
        self.curr_input = init_input
        self.STEP = 0
        self.predictions = []
        self.labels = []

    def predict(self, init_input, uncertainty_threshold=4.5, mode="e2e"):
        self.reset(init_input)

        for i in range(self.LEN):
            self.STEP = i

            pred, uncertainty_score = self._handle_prediction()
            print(f"uncertainty_score@{self.STEP}: {uncertainty_score}")

            if uncertainty_score < uncertainty_threshold:
                x_section_predicted = self._update_curr_cross_section_with_pred(pred, mode=mode)
                self.labels.append(0)
            else:
                x_section_predicted = self._handle_alternative_prediction()
                self.labels.append(1)

            self.predictions.append(x_section_predicted)

            if i != (self.LEN - 1):
                self.curr_input = self._get_next_input(self.predictions[-1].copy())

        return self.predictions, self.labels, self.ground_truth

    def _handle_prediction(self):
        pred, uncertainty = self.model_handler.predict_with_uncertainty(self.curr_input)
        uncertainty_score = np.sum(uncertainty.detach().cpu().numpy())
        pred = pred.detach().cpu().numpy()
        return pred, uncertainty_score

    def _update_curr_cross_section_with_pred(self, this_pred, mode: str):
        if mode == "e2e":
            if self.STEP == 0:
                last_prediction = self.x_sections[self.STEP].slice_based_before.ys.copy()
            else:
                last_prediction = self.predictions[-1].copy()

            next_input = last_prediction

        elif mode == "hybrid":
            next_input = self.x_sections[self.STEP].slice_based_before.ys.copy()

        else:
            raise ValueError(f"mode {mode} not supported")

        curr_torch_idx = self.torchpositions[self.STEP]
        next_input[curr_torch_idx - self.hl : curr_torch_idx + self.hl] = this_pred
        return next_input

    def _handle_alternative_prediction(self):
        cross_section_predicted = self.x_sections[self.STEP].slice_based_after.ys.copy()
        return cross_section_predicted

    def _get_next_input(self, cross_section_predicted):
        next_torch_idx = self.torchpositions[self.STEP + 1]
        next_input = cross_section_predicted[next_torch_idx - self.hl : next_torch_idx + self.hl]
        return torch.tensor(next_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
