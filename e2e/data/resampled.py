from typing import Optional

import numpy as np
import torch
from scipy.interpolate import interp1d

from e2e.data.dataset import IDs, LineSegmentZ, Samples, WaamDataset
from e2e.data.sample import Mesh2D
from e2e.helpers.resample import interp_equidistant


class ResampledShapeDataset(WaamDataset):
    def __init__(self, mirror=False, segment_length=90):
        self.inputs: Optional[list[LineSegmentZ]] = None
        self.targets: Optional[list[LineSegmentZ]] = None
        self.ids: Optional[IDs] = None
        self.fp_idx: Optional[list[torch.tensor]] = None

        self.mirror = mirror
        self._seg_len = segment_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        inpt = self.inputs[idx]
        target = self.targets[idx]
        sample_id = self.ids[idx]
        footprint = self.fp_idx[idx]

        return inpt, target, sample_id, footprint

    def create(self, samples: Samples) -> WaamDataset:
        fp_indices = self._extract_footprint_idx(samples)
        inputs = self._extract_inputs(samples, fp_indices)
        targets = self._extract_targets(samples, fp_indices)
        ids = self._get_ids(samples)

        if self.mirror:
            self.inputs, self.targets, self.ids, self.fp_idx = self._mirror_dataset(inputs, targets, ids, fp_indices)

        else:
            self.inputs, self.targets, self.ids, self.fp_idx = inputs, targets, ids, fp_indices

        return self

    def _extract_inputs(self, samples: Samples, fp_indices: list[torch.tensor]) -> list[LineSegmentZ]:
        inputs = []
        for s, fp_idx in zip(samples, fp_indices):
            zs = self._get_resampled_segment_heights(s.slice_based_before.points, fp_idx)
            inputs.append(zs)

        return inputs

    def _extract_targets(self, samples: Samples, fp_indices: list[torch.tensor]) -> list[LineSegmentZ]:
        targets = []
        for s, fp_idx in zip(samples, fp_indices):
            zs = self._get_resampled_segment_heights(s.slice_based_after.points, fp_idx)
            targets.append(zs)

        return targets

    @staticmethod
    def _extract_footprint_idx(samples: Samples) -> list[torch.tensor]:
        footprint_idx = []
        for s in samples:
            footprint = s.footprint_based
            left = footprint.left_idx
            right = footprint.right_idx
            footprint_idx.append(torch.tensor([left, right]))
        return footprint_idx

    def _mirror_dataset(self, inputs, targets, sample_ids, fp_indices):
        inputs_h = []
        targets_h = []
        sample_ids_h = []
        footprint_idx_h = []

        for inpt, target, id_, fp_idx in zip(inputs, targets, sample_ids, fp_indices):
            inputs_h.append(inpt.flipud())
            targets_h.append(target.flipud())
            footprint_idx_h.append(fp_idx.flipud())
            sample_ids_h.append(id_ + "_hflip")

        inputs_h.extend(inputs)
        targets_h.extend(targets)
        sample_ids_h.extend(sample_ids)
        footprint_idx_h.extend(fp_indices)

        return inputs_h, targets_h, sample_ids_h, footprint_idx_h

    @staticmethod
    def _get_ids(samples: Samples) -> list[str]:
        ids = []
        for s in samples:
            id_ = s.experiment + "_" + str(s.bead_id)
            ids.append(id_)
        return ids

    def _get_resampled_segment_heights(self, points, fp_idx: torch.tensor) -> torch.Tensor:
        """resamples the segment between footprint edges to be of length self._seg_len"""
        left = fp_idx[0]
        right = fp_idx[1]

        ys = np.array([p.y[0] for p in points])[left:right]
        xs = np.arange(0, len(ys)) / 10

        # resample
        f = interp1d(xs, ys, kind="linear")
        xs_new = np.linspace(0, (len(ys) - 1) / 10, self._seg_len)
        ys_new = f(xs_new)

        return torch.tensor(ys_new, dtype=torch.float32)


class ResampledShapePointsDataset(WaamDataset):
    def __init__(self, mirror=False, segment_length=90):
        self.inputs: Optional[list[LineSegmentZ]] = None
        self.targets: Optional[list[LineSegmentZ]] = None
        self.ids: Optional[IDs] = None
        self.fp_idx: Optional[list[torch.tensor]] = None

        self.mirror = mirror
        self._seg_len = segment_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        inpt = self.inputs[idx]
        target = self.targets[idx]
        sample_id = self.ids[idx]
        footprint = self.fp_idx[idx]

        return inpt, target, sample_id, footprint

    def create(self, samples: Samples) -> WaamDataset:
        fp_indices = self._extract_footprint_idx(samples)
        # TODO: Get rid of hack: using different fp index for after in case of simulated data
        fp_indices_after = self._extract_footprint_idx_after(samples)
        inputs = self._extract_inputs(samples, fp_indices)
        targets = self._extract_targets(samples, fp_indices, fp_indices_after)
        ids = self._get_ids(samples)

        if self.mirror:
            self.inputs, self.targets, self.ids, self.fp_idx = self._mirror_dataset(inputs, targets, ids, fp_indices)

        else:
            self.inputs, self.targets, self.ids, self.fp_idx = inputs, targets, ids, fp_indices

        return self

    # TODO: unify the two below functions
    def _extract_inputs(self, samples: Samples, fp_indices: list[torch.tensor]) -> list[LineSegmentZ]:
        inputs = []
        for s, fp_idx in zip(samples, fp_indices):
            tp = s.torchposition.global_y_idx
            z_shift = s.slice_based_before.ys[tp]

            if "Simulation" in s.experiment:
                x_shift = s.slice_based_before.xs[tp]
                ps = self._get_resampled_segment_points_sim_data(
                    s.slice_based_before,
                    fp_idx,
                    x_shift,
                    z_shift,
                )
            else:
                ps = self._get_resampled_segment_points(
                    s.slice_based_before,
                    fp_idx,
                    tp,
                    z_shift,
                )
            inputs.append(ps)

        return inputs

    def _extract_targets(
        self, samples: Samples, fp_indices_before: list[torch.tensor], fp_indices_after: list[torch.tensor]
    ) -> list[LineSegmentZ]:
        targets = []
        for s, fp_idx_b, fp_idx_a in zip(samples, fp_indices_before, fp_indices_after):
            tp = s.torchposition.global_y_idx
            z_shift = s.slice_based_before.ys[tp]

            if "Simulation" in s.experiment:
                # using footprint index of after because they are not the same index anymore
                x_shift = s.slice_based_before.xs[tp]
                ps = self._get_resampled_segment_points_sim_data(
                    s.slice_based_after,
                    fp_idx_a,
                    x_shift,
                    z_shift,
                )
            else:
                ps = self._get_resampled_segment_points(
                    s.slice_based_after,
                    fp_idx_b,
                    tp,
                    z_shift,
                )
            targets.append(ps)

        return targets

    @staticmethod
    def _extract_footprint_idx(samples: Samples) -> list[torch.tensor]:
        footprint_idx = []
        for s in samples:
            footprint = s.footprint_based
            left = footprint.left_idx
            right = footprint.right_idx
            footprint_idx.append(torch.tensor([left, right]))
        return footprint_idx

    @staticmethod
    def _extract_footprint_idx_after(samples: Samples) -> list[torch.tensor]:
        footprint_idx = []
        for s in samples:
            if "Simulation" in s.experiment:
                footprint = s.footprint_after
            else:
                footprint = s.footprint_based

            left = footprint.left_idx
            right = footprint.right_idx
            footprint_idx.append(torch.tensor([left, right]))
        return footprint_idx

    def _mirror_dataset(self, inputs, targets, sample_ids, fp_indices):
        inputs_h, targets_h, sample_ids_h, footprint_idx_h = [], [], [], []

        for inpt, trgt, id_, fp_idx in zip(inputs, targets, sample_ids, fp_indices):
            inpt_h, trgt_h = map(self._flip_points_tensor, [inpt, trgt])
            inputs_h.append(inpt_h)
            targets_h.append(trgt_h)
            footprint_idx_h.append(fp_idx.flipud())
            sample_ids_h.append(id_ + "_hflip")

        inputs_h.extend(inputs)
        targets_h.extend(targets)
        sample_ids_h.extend(sample_ids)
        footprint_idx_h.extend(fp_indices)

        return inputs_h, targets_h, sample_ids_h, footprint_idx_h

    @staticmethod
    def _flip_points_tensor(data: torch.tensor) -> torch.tensor:
        return torch.flip(data, dims=(1,))

    @staticmethod
    def _get_ids(samples: Samples) -> list[str]:
        ids = []
        for s in samples:
            id_ = s.experiment + "_" + str(s.bead_id)
            ids.append(id_)
        return ids

    def _get_resampled_segment_points(
        self,
        points: Mesh2D,
        fp_idx: torch.tensor,
        tp: int,
        z_shift: float,
    ) -> torch.Tensor:
        """
        Get and resamples a segment of points on the Mesh2D object.

        :param points: The Mesh2D object containing the points.
        :type points: Mesh2D
        :param fp_idx: The indices of the first and last point of the segment to resample as a torch.Tensor of size (2,).
        :type fp_idx: torch.Tensor
        :return: The resampled segment points as a torch.Tensor of size (2, num_points).
        :rtype: torch.Tensor
        """
        left, right = fp_idx[0], fp_idx[1]

        zs = points.ys[left:right]
        xs = np.arange(0, len(zs)) / 10

        xs = xs - xs[tp - left]
        zs = zs - z_shift

        xs_new, zs_new = interp_equidistant(xs, zs, num_points=self._seg_len)

        f32 = torch.float32
        return torch.stack([torch.tensor(zs_new, dtype=f32), torch.tensor(xs_new, dtype=f32)])

    def _get_resampled_segment_points_sim_data(
        self,
        points: Mesh2D,
        fp_idx: torch.tensor,
        x_shift: float,
        z_shift: float,
    ) -> torch.Tensor:
        # TODO: unify this function and the one above
        """
        Get and resamples a segment of points on the Mesh2D object for simulation data.

        :param points: The Mesh2D object containing the points.
        :type points: Mesh2D
        :param fp_idx: The indices of the first and last point of the segment to resample as a torch.Tensor of size (2,).
        :type fp_idx: torch.Tensor
        :return: The resampled segment points as a torch.Tensor of size (2, num_points).
        :rtype: torch.Tensor
        """

        left, right = fp_idx[0], fp_idx[1]

        zs = points.ys[left:right]
        xs = points.xs[left:right]

        xs = xs - x_shift
        zs = zs - z_shift

        xs_new, zs_new = interp_equidistant(xs, zs, num_points=self._seg_len)

        f32 = torch.float32
        return torch.stack([torch.tensor(zs_new, dtype=f32), torch.tensor(xs_new, dtype=f32)])


class ResampledE2EDataset(ResampledShapeDataset):
    def __init__(self, mirror=False, segment_length=90):
        super().__init__(mirror, segment_length)

        self.x_sections = []

        self.torchpositions = []
        self.ground_truth = []
        self.true_footprints = []

        self.predictions = []
        self.fp_predictions = []
        self.labels = []
        self.fp_uncertainty_scores = []
        self.shape_uncertainty_scores = []
        self.prediction_times = []

    def create(self, samples: Samples) -> WaamDataset:
        self.x_sections = samples

        fp_indices = self._extract_footprint_idx(samples)
        inputs = self._extract_inputs(samples, fp_indices)
        targets = self._extract_targets(samples, fp_indices)
        ids = self._get_ids(samples)

        self.torchpositions = [s.torchposition.global_y_idx for s in samples]
        self.ground_truth = [s.slice_based_before.ys for s in samples]
        self.ground_truth.append(samples[-1].slice_based_after.ys)

        self.true_footprints = [s.footprint_based for s in samples]

        self.inputs, self.targets, self.ids, self.fp_idx = inputs, targets, ids, fp_indices

        return self
