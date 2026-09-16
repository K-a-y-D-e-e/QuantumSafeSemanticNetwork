from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset


class JIGSAWSKinematicsDataset(Dataset):
    """
    Loads numeric kinematic trajectories from JIGSAWS.

    The loader searches recursively inside the supplied JIGSAWS
    directory and extracts rows containing only numeric values.
    """

    def __init__(
        self,
        root_dir,
        task=None,
        sequence_length=32,
        stride=16,
        normalize=True
    ):
        self.root_dir = Path(root_dir)
        self.task = task
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize

        self.samples = []
        self.feature_mean = None
        self.feature_std = None

        self._load_dataset()

    def _find_files(self):
        search_root = self.root_dir

        if self.task:
            search_root = search_root / self.task

        return list(
            search_root.rglob(
                "*kinematics/AllGestures/*.txt"
            )
        )

    @staticmethod
    def _read_numeric_rows(file_path):
        """
        Extract rows containing only numeric values.
        This makes the loader tolerant of JIGSAWS metadata/header lines.
        """

        rows = []

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                values = line.replace(",", " ").split()

                try:
                    row = [float(value) for value in values]
                except ValueError:
                    continue

                if len(row) > 1:
                    rows.append(row)

        if not rows:
            return np.empty((0, 0), dtype=np.float32)

        # JIGSAWS files should have a consistent number of columns.
        # Keep the most common row length.
        lengths = Counter(len(row) for row in rows)
        feature_count = lengths.most_common(1)[0][0]

        rows = [
            row for row in rows
            if len(row) == feature_count
        ]

        return np.asarray(rows, dtype=np.float32)

    def _load_dataset(self):

        files = self._find_files()

        if not files:
            raise FileNotFoundError(
                f"No JIGSAWS kinematic files found in "
                f"{self.root_dir}"
            )

        trajectories = []

        for file_path in files:

            data = self._read_numeric_rows(file_path)

            if len(data) < self.sequence_length:
                continue

            trajectories.append(data)

        if not trajectories:
            raise RuntimeError(
                "No usable kinematic trajectories were found."
            )

        all_data = np.concatenate(
            trajectories,
            axis=0
        )

        if self.normalize:

            self.feature_mean = all_data.mean(
                axis=0,
                keepdims=True
            )

            self.feature_std = all_data.std(
                axis=0,
                keepdims=True
            )

            self.feature_std[
                self.feature_std < 1e-8
            ] = 1.0

        for trajectory in trajectories:

            if self.normalize:

                trajectory = (
                    trajectory - self.feature_mean
                ) / self.feature_std

            for start in range(
                0,
                len(trajectory) - self.sequence_length + 1,
                self.stride
            ):

                sequence = trajectory[
                    start:
                    start + self.sequence_length
                ]

                self.samples.append(
                    torch.tensor(
                        sequence,
                        dtype=torch.float32
                    )
                )

        if not self.samples:
            raise RuntimeError(
                "No training sequences could be created."
            )

        self.feature_dim = self.samples[0].shape[-1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
