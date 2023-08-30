from typing import Optional

import numpy as np
from lightning import LightningModule
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader


class McUncertainty:
    def __init__(
        self,
        model: LightningModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        n_samples=150,
        logger=None,
    ):
        self._model = model
        self._calibrator = None

        self._train_dl = train_dataloader
        self._val_dl = val_dataloader
        self._test_dl = test_dataloader
        self._n_samples = n_samples

        self._uncertainty_preds = {"train": {}, "val": {}, "test": {}}

        self._logger = logger

    def predict(self):
        self._mc_predictions(self._train_dl, "train")
        self._mc_predictions(self._val_dl, "val")
        if self._test_dl is not None:
            self._mc_predictions(self._test_dl, "test")

    def _mc_predictions(self, data_loader: DataLoader, stage: str):
        mean_predictions = []
        uncertainties = []
        errors = []
        xs = []
        ys = []
        ids = []

        for i, batch in enumerate(data_loader):
            x, y, id_, _ = batch
            mean_prediction, uncertainty = self._model.predict_with_uncertainty(x, num_samples=150)
            error = np.abs(mean_prediction.detach().cpu().numpy() - y.detach().cpu().numpy()).tolist()
            mean_predictions.append(mean_prediction.detach().cpu().numpy().tolist())
            uncertainties.append(uncertainty.detach().cpu().numpy().tolist())
            errors.append(error)
            xs.append(x.detach().cpu())
            ys.append(y.detach().cpu())
            ids.append(id_)

        self._uncertainty_preds[stage]["mean_predictions"] = self._postprocess(mean_predictions)
        self._uncertainty_preds[stage]["uncertainties"] = self._postprocess(uncertainties)
        self._uncertainty_preds[stage]["errors"] = self._postprocess(errors)
        self._uncertainty_preds[stage]["xs"] = self._postprocess(xs)
        self._uncertainty_preds[stage]["ys"] = self._postprocess(ys)
        self._uncertainty_preds[stage]["ids"] = self._postprocess(ids)

    @staticmethod
    def _postprocess(outputs):
        return np.array([np.array(item) for sublist in outputs for item in sublist])

    def calibrate(self, strategy="mlp"):
        if strategy == "isotonic":
            # TODO: implement with one isotonic model per feature
            raise NotImplementedError
            # self._calibrator = IsotonicRegression(out_of_bounds="clip")
            #
            # uncertainties = self._uncertainty_preds["train"]["uncertainties"]
            # errors = self._uncertainty_preds["train"]["errors"]
            # _, n_features = errors.shape    # n_samples x n_features
            # calibrated_uncertainties = np.zeros_like(uncertainties)
            #
            # for i_feature in range(n_features):
            #     iso_reg = IsotonicRegression(out_of_bounds='clip')
            #     iso_reg.fit(uncertainties[:, i_feature], errors[:, i_feature])
            #
            # calibrated_uncertainties[:, i_feature] = iso_reg.transform(uncertainties[:, i_feature])

        elif strategy == "mlp":
            mlp = MLPRegressor(
                hidden_layer_sizes=(2, 120),
                max_iter=100,
                alpha=0.001,
                solver="adam",
                verbose=10,
                tol=1e-5,
                random_state=1,
                learning_rate_init=0.01,
            )

            uncertainties = self._uncertainty_preds["train"]["uncertainties"]
            errors = self._uncertainty_preds["train"]["errors"]

            mlp.fit(uncertainties, errors)
            calibrated_uncertainties = mlp.predict(uncertainties)
            self._uncertainty_preds["train"]["uncertainties_calib"] = calibrated_uncertainties
            self._uncertainty_preds["val"]["uncertainties_calib"] = mlp.predict(
                self._uncertainty_preds["val"]["uncertainties"]
            )
            if self._test_dl is not None:
                self._uncertainty_preds["test"]["uncertainties_calib"] = mlp.predict(
                    self._uncertainty_preds["test"]["uncertainties"]
                )

    def plot_predictions(self, stage, log=False, take=0):
        n_samples = self._uncertainty_preds[stage]["errors"].shape[0]

        if take:
            indices = np.random.choice(range(n_samples), take, replace=False)
        else:
            indices = range(n_samples)

        figs = []
        captions = []

        for i in indices:
            pred_mean = self._uncertainty_preds[stage]["mean_predictions"][i]
            pred_std = self._uncertainty_preds[stage]["uncertainties_calib"][i]
            uncalib_std = self._uncertainty_preds[stage]["uncertainties"][i]
            error = self._uncertainty_preds[stage]["errors"][i]
            trgt = self._uncertainty_preds[stage]["ys"][i]
            inpt = self._uncertainty_preds[stage]["xs"][i]
            id_ = self._uncertainty_preds[stage]["ids"][i]

            fig, caption = self._plot_example(pred_mean, pred_std, uncalib_std, error, trgt, inpt, id_, stage)
            figs.append(fig)
            captions.append(caption)
            plt.close()

        if log:
            self._logger.log_image(
                key=f"{stage}/preds_plots_uncertainties",
                images=figs,
                caption=captions,
            )

    @staticmethod
    def _plot_example(pred_mean, pred_std, uncalib_std, error, trgt, inpt, id_, stage):
        fig, (ax, ax2, ax3) = plt.subplots(3, sharex=True, gridspec_kw={"height_ratios": [6, 2, 2]})
        ax.plot(pred_mean, color="blue", label="pred_mean")
        # plot std around mean
        ax.fill_between(
            np.arange(len(pred_mean)),
            pred_mean - pred_std,
            pred_mean + pred_std,
            color="red",
            alpha=0.2,
            label="std",
        )
        ax.plot(trgt, color="green", label="target (after)")
        if inpt is not None:
            ax.plot(inpt, color="black", label="input (before)")
        # ax.plot(error, color="red", label="error", ls="--")
        caption = f"{stage} sample: {id_}"

        ax2.plot(pred_std, color="orange", label="calib uncertainty (std)", ls="--")
        ax2.plot(uncalib_std, color="black", label="uncalib uncertainty (std)", ls="--")

        ax3.plot(error, color="red", label="abs. error", ls="--")

        ax.set_title(caption)
        # ax.legend()
        # ax2.legend(loc="lower left", bbox_to_anchor=(0.5, -0.15))
        # ax3.legend()

        # join all three legends and put below ax3 in two columns
        fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)

        # extend plot at bottom to make space for legend
        fig.subplots_adjust(bottom=0.2)

        return fig, caption
