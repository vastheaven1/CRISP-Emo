"""Build the strictly causal DEAP+DREAMER data used by CRISP-Emo.

Raw datasets are not redistributed. Obtain DEAP and DREAMER from their
official providers, then pass their local paths to this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import mne
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfilt, sosfilt_zi

from constants import (
    BASELINE_SECONDS,
    DREAMER_ECG_RATE,
    DREAMER_EEG_RATE,
    EEG_CHANNELS,
    EEG_MONTAGES,
    TARGET_RATES,
)


FIXED_SPLIT = {
    "seed": 2026,
    "strategy": "fixed subject-disjoint development split",
    "folds": [
        {
            "fold": 0,
            "train": {
                "deap": [
                    "s07", "s08", "s09", "s13", "s23", "s31", "s01",
                    "s02", "s06", "s10", "s12", "s25", "s26", "s04",
                    "s16", "s17", "s24", "s28", "s32",
                ],
                "dreamer": [
                    "s16", "s17", "s18", "s21", "s05", "s08", "s09",
                    "s14", "s03", "s06", "s07", "s10", "s11",
                ],
            },
            "val": {
                "deap": ["s03", "s11", "s18", "s21", "s22", "s29"],
                "dreamer": ["s01", "s12", "s19", "s20", "s22"],
            },
            "test": {
                "deap": ["s05", "s14", "s15", "s19", "s20", "s27", "s30"],
                "dreamer": ["s02", "s04", "s13", "s15", "s23"],
            },
        }
    ],
}


@dataclass(frozen=True)
class TrialRecord:
    dataset: str
    subject: str
    trial: str
    source_file: str
    h5_key: str
    duration_seconds: float
    valence: float
    arousal: float
    eeg_samples: int
    gsr_samples: int
    skt_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deap-root", type=Path, required=True)
    parser.add_argument("--dreamer-mat", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-deap-subjects", type=int, default=None)
    parser.add_argument("--max-dreamer-subjects", type=int, default=None)
    return parser.parse_args()


def _status_events(raw: mne.io.BaseRaw) -> np.ndarray:
    candidates = ("Status", "", "-0")
    channel = next((name for name in candidates if name in raw.ch_names), None)
    if channel is None:
        raise ValueError(f"No DEAP status channel found in {raw.filenames[0]}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.find_events(
            raw,
            stim_channel=channel,
            shortest_event=1,
            mask=255,
            mask_type="and",
            verbose="ERROR",
        )


def _trial_intervals(raw: mne.io.BaseRaw) -> list[tuple[int, int]]:
    events = _status_events(raw)
    starts = events[events[:, 2] == 4, 0]
    stops = events[events[:, 2] == 5, 0]
    intervals = []
    for index, start in enumerate(starts):
        next_start = starts[index + 1] if index + 1 < len(starts) else np.iinfo(np.int64).max
        candidates = stops[(stops > start) & (stops < next_start)]
        if not len(candidates):
            raise RuntimeError(f"No DEAP stop marker follows sample {start}")
        intervals.append((int(start), int(candidates[0])))
    if len(intervals) != 40:
        raise RuntimeError(f"Expected 40 DEAP trials, found {len(intervals)}")
    return intervals


def _causal_filter(
    baseline: np.ndarray,
    stimulus: np.ndarray,
    sample_rate: int,
    low: float | None,
    high: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    nyquist = sample_rate / 2.0
    if low is not None and high is not None:
        sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    elif low is not None:
        sos = butter(4, low / nyquist, btype="highpass", output="sos")
    elif high is not None:
        sos = butter(4, high / nyquist, btype="lowpass", output="sos")
    else:
        return baseline.astype(np.float32), stimulus.astype(np.float32)
    initial = sosfilt_zi(sos)[:, None, :] * baseline[:, 0][None, :, None]
    filtered_baseline, state = sosfilt(sos, baseline, axis=-1, zi=initial)
    filtered_stimulus, _ = sosfilt(sos, stimulus, axis=-1, zi=state)
    return filtered_baseline.astype(np.float32), filtered_stimulus.astype(np.float32)


def _causal_downsample(
    signal: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate:
        return signal.astype(np.float32, copy=False)
    if source_rate % target_rate:
        raise ValueError(f"Non-integer causal downsampling {source_rate}->{target_rate}")
    return signal[..., :: source_rate // target_rate].astype(np.float32, copy=False)


def _baseline_center(
    baseline: np.ndarray,
    stimulus: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    baseline = _causal_downsample(baseline, source_rate, target_rate)
    stimulus = _causal_downsample(stimulus, source_rate, target_rate)
    return (stimulus - baseline.mean(axis=-1, keepdims=True)).astype(np.float32)


def _set_common_attrs(handle: h5py.File, dataset: str) -> None:
    handle.attrs.update(
        {
            "schema_version": "crisp-emo-1",
            "dataset": dataset,
            "baseline_normalization": "mean",
            "eeg_channels": json.dumps(EEG_CHANNELS),
            "target_rates": json.dumps(TARGET_RATES),
            "strictly_causal": True,
            "future_stimulus_used": False,
            "filtering": (
                "fourth-order Butterworth sosfilt with state warmed on "
                "pre-stimulus baseline"
            ),
            "resampling": "causal filtering followed by integer-stride selection",
        }
    )


def _replace_file(temporary: Path, target: Path, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"{target} exists; pass --overwrite to rebuild it")
    temporary.replace(target)


def _build_deap(
    deap_root: Path,
    output_root: Path,
    overwrite: bool,
    max_subjects: int | None,
) -> list[TrialRecord]:
    bdf_files = sorted((deap_root / "data_original").glob("s*.bdf"))
    if max_subjects is not None:
        bdf_files = bdf_files[:max_subjects]
    if not bdf_files:
        raise FileNotFoundError(f"No DEAP BDF files under {deap_root / 'data_original'}")

    output = output_root / "deap.h5"
    temporary = output.with_suffix(".h5.tmp")
    temporary.unlink(missing_ok=True)
    records = []
    with h5py.File(temporary, "w") as handle:
        _set_common_attrs(handle, "deap")
        handle.attrs["baseline_seconds"] = BASELINE_SECONDS
        for bdf_path in bdf_files:
            subject = bdf_path.stem
            labels_path = deap_root / "data_preprocessed_python" / f"{subject}.dat"
            with labels_path.open("rb") as stream:
                labels = np.asarray(
                    pickle.load(stream, encoding="latin1")["labels"], dtype=float
                )
            raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="ERROR")
            sample_rate = int(round(raw.info["sfreq"]))
            if sample_rate != 512:
                raise ValueError(f"Unexpected DEAP sample rate: {sample_rate}")
            intervals = _trial_intervals(raw)
            for trial_index, (start, marker_stop) in enumerate(intervals):
                # The audited manifest quantized trial duration on the 128-Hz
                # EEG grid before deriving every modality length.
                eeg_samples = int(
                    round((marker_stop - start) * TARGET_RATES["eeg"] / sample_rate)
                )
                duration_seconds = eeg_samples / TARGET_RATES["eeg"]
                stop = min(marker_stop, start + int(round(duration_seconds * sample_rate)))
                baseline_start = start - int(round(BASELINE_SECONDS * sample_rate))
                if baseline_start < 0:
                    raise RuntimeError(f"Insufficient baseline for {subject} trial {trial_index + 1}")
                eeg_full = raw.get_data(
                    picks=list(EEG_CHANNELS), start=baseline_start, stop=stop
                ).astype(np.float64) * 1e6
                gsr_full = raw.get_data(
                    picks=["GSR1"], start=baseline_start, stop=stop
                ).astype(np.float64) / 1000.0
                skt_full = raw.get_data(
                    picks=["Temp"], start=baseline_start, stop=stop
                ).astype(np.float64)
                split = start - baseline_start
                eeg_b, eeg_s = _causal_filter(
                    eeg_full[:, :split], eeg_full[:, split:], sample_rate, 0.5, 45.0
                )
                eeg_b -= eeg_b.mean(axis=0, keepdims=True)
                eeg_s -= eeg_s.mean(axis=0, keepdims=True)
                gsr_b, gsr_s = _causal_filter(
                    gsr_full[:, :split], gsr_full[:, split:], sample_rate, None, 5.0
                )
                skt_b, skt_s = _causal_filter(
                    skt_full[:, :split], skt_full[:, split:], sample_rate, None, 1.0
                )
                eeg = _baseline_center(eeg_b, eeg_s, sample_rate, TARGET_RATES["eeg"])
                gsr = _baseline_center(gsr_b, gsr_s, sample_rate, TARGET_RATES["gsr"])
                skt = _baseline_center(skt_b, skt_s, sample_rate, TARGET_RATES["skt"])
                eeg = eeg[..., : round(duration_seconds * TARGET_RATES["eeg"])]
                gsr = gsr[..., : round(duration_seconds * TARGET_RATES["gsr"])]
                skt = skt[..., : round(duration_seconds * TARGET_RATES["skt"])]
                trial = f"trial_{trial_index + 1:02d}"
                key = f"trials/{subject}/{trial}"
                group = handle.create_group(key)
                for name, array in (("eeg", eeg), ("gsr", gsr), ("skt", skt)):
                    if not np.isfinite(array).all():
                        raise ValueError(f"Non-finite {name} in {subject}/{trial}")
                    group.create_dataset(name, data=array, compression="lzf", chunks=True)
                valence = float(labels[trial_index, 0])
                arousal = float(labels[trial_index, 1])
                group.attrs.update(
                    subject=subject,
                    trial=trial,
                    valence=valence,
                    arousal=arousal,
                    source_file=str(bdf_path),
                )
                records.append(
                    TrialRecord(
                        "deap", subject, trial, str(bdf_path), key,
                        duration_seconds, valence, arousal,
                        eeg.shape[-1], gsr.shape[-1], skt.shape[-1],
                    )
                )
            print(f"[DEAP] {subject}: 40 trials", flush=True)
    _replace_file(temporary, output, overwrite)
    return records


def _place_shared14(eeg: np.ndarray) -> np.ndarray:
    full = np.zeros((len(EEG_CHANNELS), eeg.shape[-1]), dtype=np.float32)
    full[list(EEG_MONTAGES["shared14"])] = eeg
    return full


def _robust_ecg(baseline: np.ndarray, stimulus: np.ndarray) -> np.ndarray:
    baseline_f, stimulus_f = _causal_filter(
        baseline.T, stimulus.T, DREAMER_ECG_RATE, 0.5, 40.0
    )
    outputs = []
    for lead in range(2):
        center = float(np.median(baseline_f[lead]))
        mad = float(1.4826 * np.median(np.abs(baseline_f[lead] - center)))
        scale = max(mad, float(baseline_f[lead].std()) * 0.1, 1e-6)
        outputs.append(np.clip((stimulus_f[lead] - center) / scale, -20.0, 20.0))
    return np.asarray(outputs, dtype=np.float32)


def _build_dreamer(
    dreamer_mat: Path,
    output_root: Path,
    overwrite: bool,
    max_subjects: int | None,
) -> list[TrialRecord]:
    mat = sio.loadmat(str(dreamer_mat), struct_as_record=False, squeeze_me=True)
    root = mat["DREAMER"]
    subject_count = int(root.noOfSubjects)
    trial_count = int(root.noOfVideoSequences)
    if (subject_count, trial_count) != (23, 18):
        raise ValueError(f"Expected DREAMER 23x18, found {subject_count}x{trial_count}")
    if max_subjects is not None:
        subject_count = min(subject_count, max_subjects)

    signal_output = output_root / "dreamer.h5"
    ecg_output = output_root / "dreamer_ecg_causal.h5"
    signal_temporary = signal_output.with_suffix(".h5.tmp")
    ecg_temporary = ecg_output.with_suffix(".h5.tmp")
    signal_temporary.unlink(missing_ok=True)
    ecg_temporary.unlink(missing_ok=True)
    records = []
    with h5py.File(signal_temporary, "w") as signal_handle, h5py.File(
        ecg_temporary, "w"
    ) as ecg_handle:
        _set_common_attrs(signal_handle, "dreamer")
        signal_handle.attrs["baseline_seconds"] = 61.0
        ecg_handle.attrs.update(
            schema_version="crisp-emo-1",
            dataset="dreamer",
            sample_rate=DREAMER_ECG_RATE,
            strictly_causal=True,
            future_stimulus_used=False,
        )
        for subject_index in range(subject_count):
            subject_data = root.Data[subject_index]
            subject = f"s{subject_index + 1:02d}"
            valence_scores = np.atleast_1d(subject_data.ScoreValence)
            arousal_scores = np.atleast_1d(subject_data.ScoreArousal)
            for trial_index in range(trial_count):
                eeg_baseline = np.asarray(
                    subject_data.EEG.baseline[trial_index], dtype=np.float64
                )
                eeg_stimulus = np.asarray(
                    subject_data.EEG.stimuli[trial_index], dtype=np.float64
                )
                eeg_b, eeg_s = _causal_filter(
                    eeg_baseline.T,
                    eeg_stimulus.T,
                    DREAMER_EEG_RATE,
                    0.5,
                    45.0,
                )
                eeg_b -= eeg_b.mean(axis=0, keepdims=True)
                eeg_s -= eeg_s.mean(axis=0, keepdims=True)
                eeg = _place_shared14(
                    _baseline_center(
                        eeg_b, eeg_s, DREAMER_EEG_RATE, TARGET_RATES["eeg"]
                    )
                )
                seconds = eeg.shape[-1] / TARGET_RATES["eeg"]
                gsr = np.zeros(
                    (1, int(round(seconds * TARGET_RATES["gsr"]))), dtype=np.float32
                )
                skt = np.zeros(
                    (1, int(round(seconds * TARGET_RATES["skt"]))), dtype=np.float32
                )
                trial = f"trial_{trial_index + 1:02d}"
                key = f"trials/{subject}/{trial}"
                group = signal_handle.create_group(key)
                group.create_dataset("eeg", data=eeg, compression="lzf", chunks=True)
                group.create_dataset("gsr", data=gsr, compression="lzf", chunks=True)
                group.create_dataset("skt", data=skt, compression="lzf", chunks=True)
                valence = 2.0 * float(valence_scores[trial_index]) - 1.0
                arousal = 2.0 * float(arousal_scores[trial_index]) - 1.0
                group.attrs.update(
                    subject=subject,
                    trial=trial,
                    valence=valence,
                    arousal=arousal,
                    source_file=str(dreamer_mat),
                )

                ecg_baseline = np.asarray(
                    subject_data.ECG.baseline[trial_index], dtype=np.float64
                )
                ecg_stimulus = np.asarray(
                    subject_data.ECG.stimuli[trial_index], dtype=np.float64
                )
                ecg = _robust_ecg(ecg_baseline, ecg_stimulus)
                ecg_group = ecg_handle.create_group(key)
                ecg_group.create_dataset("ecg", data=ecg, compression="lzf", chunks=True)
                records.append(
                    TrialRecord(
                        "dreamer", subject, trial, str(dreamer_mat), key,
                        seconds, valence, arousal,
                        eeg.shape[-1], gsr.shape[-1], skt.shape[-1],
                    )
                )
            print(f"[DREAMER] {subject}: 18 trials", flush=True)

    _replace_file(signal_temporary, signal_output, overwrite)
    _replace_file(ecg_temporary, ecg_output, overwrite)
    return records


def _write_manifest(records: list[TrialRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "manifest.csv"
    splits = args.output_root / "splits.json"
    if not args.overwrite and (manifest.exists() or splits.exists()):
        raise FileExistsError(
            f"{args.output_root} already contains metadata; pass --overwrite to rebuild it"
        )

    deap_records = _build_deap(
        args.deap_root, args.output_root, args.overwrite, args.max_deap_subjects
    )
    dreamer_records = _build_dreamer(
        args.dreamer_mat,
        args.output_root,
        args.overwrite,
        args.max_dreamer_subjects,
    )
    records = deap_records + dreamer_records
    _write_manifest(records, manifest)
    splits.write_text(json.dumps(FIXED_SPLIT, indent=2) + "\n", encoding="utf-8")
    audit = {
        "strictly_causal": True,
        "future_stimulus_used": False,
        "normalization_fit": "pre-stimulus baseline only",
        "deap_trials": len(deap_records),
        "dreamer_trials": len(dreamer_records),
        "paper_split_seed": 2026,
    }
    (args.output_root / "preprocessing_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
