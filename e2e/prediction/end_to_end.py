import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from torch import nn

from e2e.data.loader import SampleLoader
from e2e.helpers.wandb import download_model


class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_samples(self, which):
        loader = SampleLoader(self.data_dir)
        samples = loader.load(which=which)
        return sorted(samples, key=lambda x: x.bead_id)


class ModelHandler:
    def __init__(self, project_name, run_id, version, model, module: nn.Module, model_path=None):
        self.model_path = download_model(project=project_name, run_id=run_id, version=version)
        if model_path:
            self.model_path = model_path
        self.model = model.load_from_checkpoint(
            model=module, checkpoint_path=self.model_path, map_location=torch.device("cpu")
        )
        self.model.eval()

    def predict_with_uncertainty(self, input_data):
        return self.model.predict_with_uncertainty(input_data.to(self.model.device), num_samples=30)


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
        fig, ax = plt.subplots(figsize=(15, 5), dpi=600)
        for i in range(len(predictions)):
            pred = predictions[i]
            y = pred  # - np.mean(pred[130:150])
            xa = np.arange(0, len(y)) / 10
            xa = xa - xa.max() / 2
            color = "red" if labels[i] == 1 else "blue"
            ax.plot(xa, y, color=color, alpha=1, ls="-", linewidth=1)

            if ground_truth:
                t = ground_truth[i]
                y_true = t - np.mean(t[:40])
                xb = np.arange(0, len(y_true)) / 10
                ax.plot(xb, y_true, color="black", alpha=1, ls="-", linewidth=0.8)

        # ax.set_ylim(0, 25)
        # ax.set_xlim(-2, 50)
        ax.set_title(title)
        ax.set_aspect("equal")

        # add legend below graph
        ax.plot([], [], color="blue", alpha=1, linewidth=1, label="prediction")
        ax.plot([], [], color="black", alpha=1, linewidth=1, ls="-", label="measurement")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        plt.show()

        return fig

    @staticmethod
    def plot_outline(outline_pred, outline_true):
        fig, ax = plt.subplots(figsize=(15, 5), dpi=600)

        xa = np.arange(0, len(outline_pred)) / 10
        ax.plot(xa, outline_pred, color="blue", alpha=1, ls="-", linewidth=1)

        xb = np.arange(0, len(outline_true)) / 10
        ax.plot(xb, outline_true, color="black", alpha=1, ls="--", linewidth=0.8)

        ax.set_ylim(0, 22)
        ax.set_xlim(0, xb.max())
        ax.set_aspect("equal")

        # shade the area between the true outline and the predicted outline
        # in green if the difference is positive, and in red if it is negative

        diff = outline_pred - outline_true
        ax.fill_between(
            xa,
            outline_pred,
            outline_true,
            where=diff > 0,
            color="red",
            alpha=0.3,
            label="over predicted",
        )
        ax.fill_between(
            xa, outline_pred, outline_true, where=diff < 0, color="green", alpha=0.3, label="under predicted"
        )

        # fill inside of min of the two outlines with dashed lines
        ax.fill_between(
            xa,
            np.min((outline_pred, outline_true), axis=0),
            color="lightgrey",
            hatch="\\",
            edgecolor="black",
            alpha=0.3,
        )

        # add legend below graph
        ax.plot([], [], color="blue", alpha=1, linewidth=1, label="prediction")
        ax.plot([], [], color="black", alpha=1, linewidth=1, ls="--", label="measurement")

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        plt.show()
        return fig


class Predictor:
    def __init__(self, output_length, footprint_handler, shape_handler, cross_section_samples):
        self.footprint = footprint_handler
        self.shape = shape_handler

        self.output_length = output_length
        self.hl = output_length // 2

        self.x_sections = cross_section_samples

        self.STEP = 0

        self.torchpositions = [s.torchposition.global_y_idx for s in self.x_sections]
        self.ground_truth = [s.slice_based_before.ys for s in self.x_sections]
        self.true_footprints = [s.footprint_based for s in self.x_sections]
        self.LEN = len(self.x_sections)

        # init during setup
        self.curr_input = None

        # prediction history
        self.predictions = []
        self.labels = []
        self.footprint_errors = []

    def reset(self, init_input):
        self.curr_input = init_input
        self.STEP = 0
        self.predictions = []
        self.labels = []

    def predict(self, init_input, uncertainty_threshold=4.5, mode="e2e", predict_footprint=True):
        self.reset(init_input)
        self.predict_footprint = predict_footprint

        for i in range(self.LEN):
            self.STEP = i

            tic = time.time()
            pred, uncertainty_score, footprint = self._handle_prediction()
            toc = time.time()
            print(f"uncertainty_score@{self.STEP}: {uncertainty_score}")
            print(f"prediction took {toc-tic:.2f}s")

            if uncertainty_score < uncertainty_threshold:
                x_section_predicted = self._update_curr_cross_section_with_pred(pred, footprint, mode=mode)
                self.labels.append(0)
            else:
                x_section_predicted = self._handle_alternative_prediction()
                self.labels.append(1)

            self.predictions.append(x_section_predicted)

            if i != (self.LEN - 1):
                self.curr_input = self._get_next_input(self.predictions[-1].copy())

        return self.predictions, self.labels, self.ground_truth

    def _handle_prediction(self):
        # predict footprint
        if self.predict_footprint:
            footprint, fp_uncertainty = self.footprint.predict_with_uncertainty(self.curr_input)

        else:
            # use real footprint
            true_fp = self.true_footprints[self.STEP]
            tp = self.torchpositions[self.STEP]
            left_fp = 112 - np.abs(tp - true_fp.left_idx)
            right_fp = 112 + np.abs(tp - true_fp.right_idx)
            footprint = torch.tensor([[left_fp, right_fp]], dtype=torch.float32)

        # extract segment for shape prediction
        segment = self.curr_input.squeeze(0)[:, footprint[:, 0].int() : footprint[:, 1].int()].cpu().numpy().squeeze()

        # upsampling
        len_segment = len(segment)
        xs_original = np.arange(0, len_segment) / 10

        f = interp1d(xs_original, segment, kind="linear")
        xs_90 = np.linspace(0, (len_segment - 1) / 10, 90)
        curr_input_segment = f(xs_90)

        # predict shape
        pred, uncertainty = self.shape.predict_with_uncertainty(
            torch.tensor(curr_input_segment, dtype=torch.float32).view(-1, 90)
        )
        uncertainty_score = np.sum(uncertainty.detach().cpu().numpy())
        pred = pred.detach().cpu().numpy()

        # if (self.STEP % 5 == 0) or (self.STEP == 0):
        #     xs = np.arange(0, 90) / 10
        #     plt.plot(xs, pred.reshape(-1))
        #     plt.plot(xs, curr_input_segment.reshape(-1))
        #     plt.title(f"step {self.STEP}")
        #     plt.show()

        # downsampling
        f = interp1d(xs_90, pred, kind="linear")
        pred_downsampled = f(xs_original)

        return pred_downsampled, uncertainty_score, footprint

    def _update_curr_cross_section_with_pred(self, this_pred, footprint, mode: str):
        this_pred = this_pred.squeeze()

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

        mid_idx = 224 // 2
        curr_torch_idx = self.torchpositions[self.STEP]
        # base = next_input[curr_torch_idx - self.hl : curr_torch_idx + self.hl]
        # diff = np.abs(base - this_pred).squeeze()

        # find footprint
        left_fp = footprint[:, 0].int().item()
        # max(0, self._find_edge(diff, mid_idx, threshold=0.1, which="left"))
        right_fp = footprint[:, 1].int().item()
        # min(90, self._find_edge(diff, mid_idx, threshold=0.1, which="right"))

        # true_left_fp = self.true_footprints[self.STEP].left_idx
        # true_right_fp = self.true_footprints[self.STEP].right_idx

        # left_err = np.abs(np.abs(left_fp - mid_idx) - np.abs(true_left_fp - curr_torch_idx))
        # right_err = np.abs(np.abs(right_fp - mid_idx) - np.abs(true_right_fp - curr_torch_idx))

        # self.footprint_errors.append([left_err, right_err])

        next_input[curr_torch_idx - (mid_idx - left_fp) : curr_torch_idx + (right_fp - mid_idx)] = this_pred
        return next_input

    def _handle_alternative_prediction(self):
        cross_section_predicted = self.x_sections[self.STEP].slice_based_after.ys.copy()
        return cross_section_predicted

    def _get_next_input(self, cross_section_predicted):
        next_torch_idx = self.torchpositions[self.STEP + 1]
        next_input = cross_section_predicted[next_torch_idx - self.hl : next_torch_idx + self.hl]
        return torch.tensor(next_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # def _find_edge(self, y_diff, idx_peak, threshold=0.07, which="left"):
    #     # TODO: use different algorithm: e.g. island finding
    #     # https://stackoverflow.com/questions/52189433/find-the-first-index-for-which-an-array-goes-below-a-certain-
    #     threshold-and-stay
    #     if which == "left":
    #         sign = -1
    #     elif which == "right":
    #         sign = 1
    #     else:
    #         raise ValueError(f"which={which} is undefined.")
    #
    #     k = 0
    #     while y_diff[idx_peak + sign * k] >= threshold:
    #         k += 1
    #
    #         if (idx_peak + sign * k) < 0 or (idx_peak + sign * k) >= len(y_diff):
    #             # TODO: find out in which cases this happens and whether it can be avoided
    #             logger.debug(f"FootprintDetection: {self.STEP} - index out of bounds for k={k}")
    #             # Correcting k by {sign} and skipping.
    #             k -= sign
    #             break
    #
    #     return idx_peak + sign * k
