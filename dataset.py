"""DEAP/DREAMER window datasets used by CRISP-Emo."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from constants import EEG_MONTAGES, TARGET_RATES


@dataclass(frozen=True)
class WindowRecord:
    dataset: str
    subject: str
    trial: str
    stimulus: str
    h5_key: str
    start_seconds: float
    valence: float
    arousal: float


class PhysioWindowDataset(Dataset):
    """Lazy subject-disjoint windows backed by the preprocessed HDF5 files."""

    DOMAIN_IDS = {"deap": 0, "dreamer": 1}

    def __init__(
        self,
        root: Path,
        fold: int = 0,
        partition: str = "train",
        datasets: tuple[str, ...] = ("deap", "dreamer"),
        window_seconds: float = 5.0,
        stride_seconds: float = 2.5,
        pps_offset_seconds: float = 0.0,
        trial_start_fraction: float = 0.0,
        eeg_channel_indices: tuple[int, ...] = tuple(range(32)),
        threshold: float = 5.0,
        neutral_policy: str = "low",
        eeg_normalization: str = "physical",
    ) -> None:
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"Unknown partition: {partition}")
        if any(dataset not in self.DOMAIN_IDS for dataset in datasets):
            raise ValueError(f"Only DEAP and DREAMER are supported: {datasets}")
        if neutral_policy not in {"low", "drop"}:
            raise ValueError("neutral_policy must be low or drop")
        if eeg_normalization != "physical":
            raise ValueError("The paper protocol requires physical normalization")
        if window_seconds <= 0 or stride_seconds <= 0:
            raise ValueError("Window and stride must be positive")
        if pps_offset_seconds < 0:
            raise ValueError("PPS offset must be non-negative")
        if not 0.0 <= trial_start_fraction < 1.0:
            raise ValueError("trial_start_fraction must be in [0,1)")
        if (
            not eeg_channel_indices
            or len(set(eeg_channel_indices)) != len(eeg_channel_indices)
            or min(eeg_channel_indices) < 0
            or max(eeg_channel_indices) >= 32
        ):
            raise ValueError("EEG channel indices must be unique values in [0,31]")

        self.root = Path(root)
        self.partition = partition
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.pps_offset_seconds = pps_offset_seconds
        self.trial_start_fraction = trial_start_fraction
        self.eeg_channel_indices = tuple(eeg_channel_indices)
        self.threshold = threshold
        self._handles: dict[str, h5py.File] = {}

        split_spec = json.loads(
            (self.root / "splits.json").read_text(encoding="utf-8")
        )
        selected_fold = next(
            item for item in split_spec["folds"] if int(item["fold"]) == fold
        )
        allowed = {
            dataset: set(selected_fold[partition].get(dataset, []))
            for dataset in datasets
        }
        with (self.root / "manifest.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            manifest = list(csv.DictReader(stream))

        subject_keys = sorted(
            {f"{row['dataset']}:{row['subject']}" for row in manifest}
        )
        self.subject_id_by_key = {
            key: index for index, key in enumerate(subject_keys)
        }
        self.windows: list[WindowRecord] = []
        for row in manifest:
            dataset = row["dataset"]
            if dataset not in allowed or row["subject"] not in allowed[dataset]:
                continue
            valence = float(row["valence"])
            arousal = float(row["arousal"])
            if neutral_policy == "drop" and (
                valence == threshold or arousal == threshold
            ):
                continue
            duration = float(row["duration_seconds"])
            required = window_seconds + pps_offset_seconds
            if duration < required:
                continue
            starts = np.arange(
                0.0,
                duration - required + 1e-6,
                stride_seconds,
            )
            starts = starts[starts >= duration * trial_start_fraction]
            for start in starts:
                self.windows.append(
                    WindowRecord(
                        dataset=dataset,
                        subject=row["subject"],
                        trial=row["trial"],
                        stimulus=f"{dataset}:{row['trial']}",
                        h5_key=row["h5_key"],
                        start_seconds=float(start),
                        valence=valence,
                        arousal=arousal,
                    )
                )
        if not self.windows:
            raise RuntimeError(
                f"No windows for fold={fold}, partition={partition}, datasets={datasets}"
            )
        self._sample_weights = (
            self._compute_sample_weights()
            if partition == "train"
            else torch.ones(len(self.windows), dtype=torch.double)
        )

    def __len__(self) -> int:
        return len(self.windows)

    def _handle(self, dataset: str) -> h5py.File:
        if dataset not in self._handles:
            self._handles[dataset] = h5py.File(
                self.root / f"{dataset}.h5", "r"
            )
        return self._handles[dataset]

    @staticmethod
    def _slice(
        group: h5py.Group,
        name: str,
        start_seconds: float,
        window_seconds: float,
    ) -> np.ndarray:
        rate = TARGET_RATES[name]
        start = int(round(start_seconds * rate))
        length = int(round(window_seconds * rate))
        output = np.asarray(
            group[name][..., start : start + length], dtype=np.float32
        )
        if output.shape[-1] != length:
            raise RuntimeError(
                f"Short {name} window at {group.name}: "
                f"{output.shape[-1]} != {length}"
            )
        return output

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.windows[index]
        group = self._handle(record.dataset)[record.h5_key]
        eeg = self._slice(
            group, "eeg", record.start_seconds, self.window_seconds
        )[list(self.eeg_channel_indices)]
        pps_start = record.start_seconds + self.pps_offset_seconds
        gsr = self._slice(group, "gsr", pps_start, self.window_seconds)
        skt = self._slice(group, "skt", pps_start, self.window_seconds)

        eeg = np.clip(eeg, -500.0, 500.0) / 100.0
        gsr = np.sign(gsr) * np.log1p(np.clip(np.abs(gsr), 0.0, 100.0))
        skt = np.clip(skt, -5.0, 5.0)
        labels = np.asarray(
            [
                record.valence > self.threshold,
                record.arousal > self.threshold,
            ],
            dtype=np.int64,
        )
        trial_id = f"{record.dataset}:{record.subject}:{record.h5_key}"
        return {
            "eeg": torch.from_numpy(eeg.copy()),
            "gsr": torch.from_numpy(gsr.copy()),
            "skt": torch.from_numpy(skt.copy()),
            "labels": torch.from_numpy(labels),
            "domain": torch.tensor(
                self.DOMAIN_IDS[record.dataset], dtype=torch.long
            ),
            "subject_id": torch.tensor(
                self.subject_id_by_key[f"{record.dataset}:{record.subject}"],
                dtype=torch.long,
            ),
            "dataset": record.dataset,
            "subject": record.subject,
            "trial_id": trial_id,
            "stimulus_id": record.stimulus,
            "start_seconds": torch.tensor(
                record.start_seconds, dtype=torch.float32
            ),
            "sample_weight": self._sample_weights[index].float(),
        }

    def _compute_sample_weights(self) -> torch.DoubleTensor:
        strata = [
            (
                record.dataset,
                int(record.valence > self.threshold),
                int(record.arousal > self.threshold),
            )
            for record in self.windows
        ]
        trial_ids = [
            f"{record.dataset}:{record.subject}:{record.h5_key}"
            for record in self.windows
        ]
        trials_per_stratum: dict[tuple[str, int, int], set[str]] = {}
        windows_per_trial: dict[str, int] = {}
        for stratum, trial_id in zip(strata, trial_ids, strict=True):
            trials_per_stratum.setdefault(stratum, set()).add(trial_id)
            windows_per_trial[trial_id] = windows_per_trial.get(trial_id, 0) + 1
        return torch.tensor(
            [
                1.0
                / (
                    len(trials_per_stratum[stratum])
                    * windows_per_trial[trial_id]
                )
                for stratum, trial_id in zip(strata, trial_ids, strict=True)
            ],
            dtype=torch.double,
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, TypeError):
            pass


class SensorAwareWindowDataset(Dataset):
    """Periphery10 EEG with corpus-native DEAP PPS or DREAMER ECG."""

    def __init__(
        self,
        root: Path,
        partition: str,
        fold: int = 0,
        eeg_montage: str = "periphery10",
    ) -> None:
        self.root = Path(root)
        self.fold = fold
        if eeg_montage != "periphery10":
            raise ValueError("CRISP-Emo uses the Periphery10 montage")
        self.eeg_montage = eeg_montage
        self.base = PhysioWindowDataset(
            root=self.root,
            datasets=("deap", "dreamer"),
            fold=fold,
            partition=partition,
            window_seconds=5.0,
            stride_seconds=2.5,
            threshold=5.0,
            neutral_policy="low",
            eeg_normalization="physical",
            eeg_channel_indices=EEG_MONTAGES[eeg_montage],
        )
        self._ecg_handle: h5py.File | None = None
        names = sorted({record.stimulus for record in self.base.windows})
        self.stimulus_ids = {name: index for index, name in enumerate(names)}

    @property
    def windows(self) -> list:
        return self.base.windows

    def __len__(self) -> int:
        return len(self.base)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_ecg_handle"] = None
        return state

    def close(self) -> None:
        self.base.close()
        if self._ecg_handle is not None:
            self._ecg_handle.close()
            self._ecg_handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, TypeError):
            pass

    def _ecg(self, key: str, start_seconds: float) -> np.ndarray:
        if self._ecg_handle is None:
            self._ecg_handle = h5py.File(
                self.root / "dreamer_ecg_causal.h5", "r"
            )
        start = int(round(start_seconds * 256))
        length = 5 * 256
        ecg = np.asarray(
            self._ecg_handle[key]["ecg"][..., start : start + length],
            dtype=np.float32,
        )
        if ecg.shape != (2, length):
            raise RuntimeError(f"Short ECG window at {key}: {ecg.shape}")
        return ecg

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        record = self.base.windows[index]
        if record.dataset == "deap":
            eda = item["gsr"]
            temperature = item["skt"]
            ecg = torch.zeros(2, 5 * 256, dtype=torch.float32)
            sensor_type = 0
        else:
            eda = torch.zeros(1, 5 * 32, dtype=torch.float32)
            temperature = torch.zeros(1, 5 * 8, dtype=torch.float32)
            ecg = torch.from_numpy(
                self._ecg(record.h5_key, record.start_seconds).copy()
            )
            sensor_type = 1
        return {
            **item,
            "eda": eda,
            "temperature": temperature,
            "ecg": ecg,
            "sensor_type": torch.tensor(sensor_type, dtype=torch.long),
            "stimulus_index": torch.tensor(
                self.stimulus_ids[record.stimulus], dtype=torch.long
            ),
        }


def weighted_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.flatten(0, 1),
        labels.flatten(0, 1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(labels.shape)
    weights = sample_weights.float()[:, None]
    denominator = (weights.sum() * labels.shape[1]).clamp_min(1e-6)
    return (losses * weights).sum() / denominator
