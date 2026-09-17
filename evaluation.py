"""Metrics and evaluation for CRISP-Emo."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import weighted_classification_loss


TASK_NAMES = ("valence", "arousal")
DOMAIN_NAMES = {0: "deap", 1: "dreamer"}


def _binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "decision_threshold": float(threshold),
    }
    result["auroc"] = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else float("nan")
    )
    return result


def _fit_binary_threshold(
    labels: np.ndarray, probabilities: np.ndarray, objective: str
) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    sorted_probabilities = probabilities[order]
    group_ends = np.flatnonzero(
        np.r_[sorted_probabilities[1:] != sorted_probabilities[:-1], True]
    )
    included = group_ends + 1
    true_positive = np.cumsum(sorted_labels)[group_ends].astype(float)
    false_positive = included - true_positive
    positive = float(sorted_labels.sum())
    negative = float(len(sorted_labels) - positive)
    false_negative = positive - true_positive
    true_negative = negative - false_positive
    thresholds = sorted_probabilities[group_ends]

    if objective == "youden":
        scores = true_positive / positive + true_negative / negative - 1.0
    elif objective == "balanced_accuracy":
        scores = 0.5 * (true_positive / positive + true_negative / negative)
    elif objective == "accuracy":
        scores = (true_positive + true_negative) / (positive + negative)
    elif objective == "macro_f1":
        positive_f1 = 2.0 * true_positive / np.maximum(
            2.0 * true_positive + false_positive + false_negative, 1.0
        )
        negative_f1 = 2.0 * true_negative / np.maximum(
            2.0 * true_negative + false_positive + false_negative, 1.0
        )
        scores = 0.5 * (positive_f1 + negative_f1)
    else:
        raise ValueError(f"Unknown threshold objective: {objective}")
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12))
    return float(thresholds[best[np.argmin(np.abs(thresholds[best] - 0.5))]])


def _fit_thresholds(
    labels: np.ndarray, probabilities: np.ndarray, objective: str
) -> list[float]:
    fitted = []
    for task_id in range(len(TASK_NAMES)):
        fitted.append(
            _fit_binary_threshold(
                labels[:, task_id], probabilities[:, task_id], objective
            )
        )
    return fitted


def _summarize_flat_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    decision_thresholds: tuple[float, float] | list[float] | dict[str, list[float]] | None,
    fit_thresholds: bool,
    count_name: str,
    threshold_objective: str,
    threshold_scope: str,
    selection_tasks: tuple[str, ...],
) -> dict:
    if fit_thresholds:
        if threshold_scope == "global":
            decision_thresholds = _fit_thresholds(
                labels, probabilities, threshold_objective
            )
        elif threshold_scope == "dataset":
            decision_thresholds = {
                domain_name: _fit_thresholds(
                    labels[domains == domain_id],
                    probabilities[domains == domain_id],
                    threshold_objective,
                )
                for domain_id, domain_name in DOMAIN_NAMES.items()
                if (domains == domain_id).any()
            }
        else:
            raise ValueError(f"Unknown threshold scope: {threshold_scope}")
    elif decision_thresholds is None:
        decision_thresholds = (0.5, 0.5)
    if not isinstance(decision_thresholds, dict) and len(decision_thresholds) != len(TASK_NAMES):
        raise ValueError("Expected one decision threshold per task.")

    report: dict[str, dict] = {
        count_name: int(len(labels)),
        "evaluation_unit": "window" if count_name == "n_windows" else "trial",
        "decision_thresholds": (
            decision_thresholds
            if isinstance(decision_thresholds, dict)
            else list(decision_thresholds)
        ),
        "threshold_objective": threshold_objective,
        "threshold_scope": threshold_scope,
        "datasets": {},
    }
    selection_scores: dict[str, list[float]] = {
        "auroc": [],
        "macro_f1": [],
        "balanced_accuracy": [],
    }
    for domain_id, domain_name in DOMAIN_NAMES.items():
        mask = domains == domain_id
        if not mask.any():
            continue
        domain_report = {count_name: int(mask.sum()), "tasks": {}}
        domain_thresholds = (
            decision_thresholds[domain_name]
            if isinstance(decision_thresholds, dict)
            else decision_thresholds
        )
        for task_id, task_name in enumerate(TASK_NAMES):
            metrics = _binary_metrics(
                labels[mask, task_id],
                probabilities[mask, task_id],
                float(domain_thresholds[task_id]),
            )
            domain_report["tasks"][task_name] = metrics
            if task_name in selection_tasks:
                for metric_name in selection_scores:
                    if np.isfinite(metrics[metric_name]):
                        selection_scores[metric_name].append(metrics[metric_name])
        report["datasets"][domain_name] = domain_report
    report["selection_scores"] = {
        metric_name: float(np.mean(values)) if values else float("nan")
        for metric_name, values in selection_scores.items()
    }
    report["selection_score"] = report["selection_scores"]["auroc"]
    return report


def summarize_window_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    decision_thresholds: tuple[float, float] | list[float] | dict[str, list[float]] | None = None,
    fit_thresholds: bool = False,
    threshold_objective: str = "youden",
    threshold_scope: str = "global",
    selection_tasks: tuple[str, ...] = TASK_NAMES,
) -> dict:
    return _summarize_flat_predictions(
        probabilities,
        labels,
        domains,
        decision_thresholds,
        fit_thresholds,
        count_name="n_windows",
        threshold_objective=threshold_objective,
        threshold_scope=threshold_scope,
        selection_tasks=selection_tasks,
    )


def summarize_trial_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    trial_ids: list[str],
    decision_thresholds: tuple[float, float] | list[float] | dict[str, list[float]] | None = None,
    fit_thresholds: bool = False,
    threshold_objective: str = "youden",
    threshold_scope: str = "global",
    selection_tasks: tuple[str, ...] = TASK_NAMES,
) -> dict:
    grouped: dict[str, dict] = defaultdict(
        lambda: {"probabilities": [], "labels": None, "domain": None}
    )
    for probability, label, domain, trial_id in zip(
        probabilities, labels, domains, trial_ids, strict=True
    ):
        grouped[trial_id]["probabilities"].append(probability)
        grouped[trial_id]["labels"] = label
        grouped[trial_id]["domain"] = int(domain)

    trial_probabilities = np.asarray(
        [np.mean(item["probabilities"], axis=0) for item in grouped.values()]
    )
    trial_labels = np.asarray([item["labels"] for item in grouped.values()])
    trial_domains = np.asarray([item["domain"] for item in grouped.values()])

    return _summarize_flat_predictions(
        trial_probabilities,
        trial_labels,
        trial_domains,
        decision_thresholds,
        fit_thresholds,
        count_name="n_trials",
        threshold_objective=threshold_objective,
        threshold_scope=threshold_scope,
        selection_tasks=selection_tasks,
    )


def summarize_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    trial_ids: list[str],
    evaluation_unit: str,
    decision_thresholds: tuple[float, float] | list[float] | dict[str, list[float]] | None = None,
    fit_thresholds: bool = False,
    threshold_objective: str = "youden",
    threshold_scope: str = "global",
    selection_tasks: tuple[str, ...] = TASK_NAMES,
) -> dict:
    if evaluation_unit == "window":
        return summarize_window_predictions(
            probabilities,
            labels,
            domains,
            decision_thresholds=decision_thresholds,
            fit_thresholds=fit_thresholds,
            threshold_objective=threshold_objective,
            threshold_scope=threshold_scope,
            selection_tasks=selection_tasks,
        )
    if evaluation_unit == "trial":
        return summarize_trial_predictions(
            probabilities,
            labels,
            domains,
            trial_ids,
            decision_thresholds=decision_thresholds,
            fit_thresholds=fit_thresholds,
            threshold_objective=threshold_objective,
            threshold_scope=threshold_scope,
            selection_tasks=selection_tasks,
        )
    raise ValueError(f"Unknown evaluation unit: {evaluation_unit}")


def move_batch(raw: dict, device: torch.device, include_teacher: bool = False) -> dict:
    del include_teacher  # Teacher views are moved explicitly by the training loop.
    tensor_keys = (
        "eeg",
        "eda",
        "temperature",
        "ecg",
        "sensor_type",
        "labels",
        "domain",
        "subject_id",
        "sample_weight",
    )
    batch = {key: raw[key].to(device, non_blocking=True) for key in tensor_keys}
    batch["trial_id"] = raw["trial_id"]
    return batch


def forward_model(model, batch: dict) -> dict:
    return model(
        batch["eeg"],
        batch["eda"],
        batch["temperature"],
        batch["ecg"],
        batch["sensor_type"],
    )


@torch.no_grad()
def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    decision_thresholds=None,
    fit_thresholds: bool = False,
) -> dict:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    domains: list[np.ndarray] = []
    trial_ids: list[str] = []
    loss_sum = 0.0
    sample_count = 0
    for batch_index, raw in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw, device)
        output = forward_model(model, batch)
        batch_size = batch["labels"].shape[0]
        loss_sum += float(
            weighted_classification_loss(
                output["logits"],
                batch["labels"],
                torch.ones_like(batch["sample_weight"]),
                label_smoothing=0.0,
            )
        ) * batch_size
        sample_count += batch_size
        full_probability = output["logits"].softmax(dim=-1)[..., 1]
        probabilities.append(full_probability.cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        domains.append(batch["domain"].cpu().numpy())
        trial_ids.extend(batch["trial_id"])

    report = summarize_predictions(
        np.concatenate(probabilities),
        np.concatenate(labels),
        np.concatenate(domains),
        trial_ids,
        evaluation_unit="window",
        decision_thresholds=decision_thresholds,
        fit_thresholds=fit_thresholds,
        threshold_objective="macro_f1",
        threshold_scope="dataset",
        selection_tasks=("valence", "arousal"),
    )
    report["loss"] = loss_sum / max(sample_count, 1)
    return report


def primary_scores(report: dict) -> dict[str, float]:
    accuracies = [
        task_report["accuracy"]
        for dataset_report in report["datasets"].values()
        for task_report in dataset_report["tasks"].values()
    ]
    accuracy = float(np.mean(accuracies))
    macro_f1 = float(report["selection_scores"]["macro_f1"])
    balanced_accuracy = float(report["selection_scores"]["balanced_accuracy"])
    auroc = float(report["selection_scores"]["auroc"])
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "auroc": auroc,
        "accuracy_f1_mean": 0.5 * (accuracy + macro_f1),
    }
