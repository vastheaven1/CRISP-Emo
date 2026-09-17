"""Training losses used by CRISP-Emo."""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from dataset import SensorAwareWindowDataset


class TeacherViewDataset(Dataset):
    def __init__(self, base: SensorAwareWindowDataset, views: torch.Tensor) -> None:
        if views.shape != (len(base), 2, 200):
            raise ValueError((tuple(views.shape), len(base)))
        self.base = base
        self.views = views.float().contiguous()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        item["rsd_teacher_views"] = self.views[index]
        return item


class ArchitectureAgnosticDecoupler(nn.Module):
    """Training-only AAD: 64 -> 128 -> 200, following RSD gamma=2."""

    def __init__(self, student_dim: int = 64, teacher_dim: int = 200) -> None:
        super().__init__()
        hidden = student_dim * 2
        self.projector = nn.Sequential(
            nn.Linear(student_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, teacher_dim, bias=False),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projector(value)


def weighted_normalize(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float().clamp_min(0)
    weights = weights / weights.sum().clamp_min(1e-8)
    mean = (features.float() * weights[:, None]).sum(0)
    centered = features.float() - mean
    variance = (centered.square() * weights[:, None]).sum(0)
    return centered / (variance.sqrt() + 1e-5)


def weighted_rsd_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    weights: torch.Tensor,
    kappa: float = 0.01,
) -> torch.Tensor:
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError((tuple(student.shape), tuple(teacher.shape)))
    normalized_weights = weights.float().clamp_min(0)
    normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(1e-8)
    student = weighted_normalize(student, normalized_weights)
    teacher = weighted_normalize(teacher.detach(), normalized_weights)
    correlation = teacher.T @ (student * normalized_weights[:, None])
    dimension = correlation.shape[0]
    identity = torch.eye(dimension, device=correlation.device, dtype=correlation.dtype)
    squared = (correlation - identity).square()
    squared = torch.where(identity.bool(), squared, squared * kappa)
    return squared.sum() / dimension


def corpus_separated_rsd_loss(
    projected_student: torch.Tensor,
    teacher_views: torch.Tensor,
    domains: torch.Tensor,
    sample_weights: torch.Tensor,
    kappa: float = 0.01,
) -> torch.Tensor:
    fused_teacher = teacher_views.float().mean(dim=1)
    losses = []
    for domain in (0, 1):
        selected = domains == domain
        if int(selected.sum()) < 2:
            continue
        losses.append(
            weighted_rsd_loss(
                projected_student[selected],
                fused_teacher[selected],
                sample_weights[selected],
                kappa=kappa,
            )
        )
    if not losses:
        return projected_student.sum() * 0.0
    return torch.stack(losses).mean()


class SelfPacedProjector(nn.Module):
    """Training-only projection head used by the contrastive objective."""

    def __init__(self, input_dim: int = 64, output_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(value.float()), dim=-1)


class TrialReliabilityMemory:
    """Online train-only EMA of label agreement for every trial and task."""

    def __init__(self, decay: float = 0.8) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1).")
        self.decay = decay
        self.values: dict[tuple[str, int], float] = {}

    @torch.no_grad()
    def reliability_and_update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        trial_ids: list[str],
    ) -> torch.Tensor:
        probabilities = logits.float().softmax(dim=-1)
        assigned = probabilities.gather(-1, labels[..., None]).squeeze(-1)
        output = assigned.clone()
        grouped: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
        for sample, trial_id in enumerate(trial_ids):
            for task in range(labels.shape[1]):
                key = (trial_id, task)
                current = float(assigned[sample, task])
                output[sample, task] = self.values.get(key, current)
                grouped[key].append((sample, current))
        for key, observations in grouped.items():
            batch_mean = sum(value for _, value in observations) / len(observations)
            previous = self.values.get(key)
            self.values[key] = (
                batch_mean if previous is None
                else self.decay * previous + (1.0 - self.decay) * batch_mean
            )
        # Geometric blending keeps a locally implausible window from being
        # promoted only because the rest of its trial is easy.
        return (assigned * output).clamp_min(1e-8).sqrt()


def self_paced_ratio(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    start_ratio: float,
    end_ratio: float,
) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    progress = min(max((epoch - warmup_epochs - 1) / max(ramp_epochs - 1, 1), 0.0), 1.0)
    return start_ratio + progress * (end_ratio - start_ratio)


def _stratified_selection(
    reliability: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    selected = torch.zeros_like(labels, dtype=torch.bool)
    for task in range(labels.shape[1]):
        for domain in domains.unique():
            for label in (0, 1):
                group = (domains == domain) & (labels[:, task] == label)
                indices = group.nonzero(as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                keep = max(1, math.ceil(indices.numel() * ratio))
                ranked = reliability[indices, task].topk(keep, largest=True).indices
                selected[indices[ranked], task] = True
    return selected


def trial_aware_self_paced_contrastive_loss(
    projected: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    subject_ids: torch.Tensor,
    trial_ids: list[str],
    memory: TrialReliabilityMemory,
    ratio: float,
    temperature: float = 0.12,
) -> tuple[torch.Tensor, dict[str, float]]:
    """ItS2CLR-inspired reliable-instance contrast across source subjects.

    The ordinary window classification loss remains active outside this
    function. Reliable anchors are selected separately per corpus, task and
    class; positives must come from another subject in the same corpus.
    """

    reliability = memory.reliability_and_update(logits.detach(), labels, trial_ids)
    if ratio <= 0.0:
        return projected.sum() * 0.0, {
            "spcl_ratio": 0.0,
            "spcl_coverage": 0.0,
            "spcl_valid_anchors": 0.0,
            "spcl_reliability": float(reliability.mean()),
        }
    selected = _stratified_selection(reliability, labels, domains, ratio)
    similarity = projected @ projected.T / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(projected.shape[0], dtype=torch.bool, device=projected.device)
    different_subject = subject_ids[:, None] != subject_ids[None, :]
    same_domain = domains[:, None] == domains[None, :]
    losses = []
    weights = []
    valid_anchor_count = 0
    for task in range(labels.shape[1]):
        candidate = selected[:, task]
        candidate_pair = candidate[:, None] & candidate[None, :]
        eligible = candidate_pair & same_domain & different_subject & ~eye
        positive = eligible & (labels[:, task, None] == labels[None, :, task])
        valid = candidate & positive.any(dim=1) & (eligible & ~positive).any(dim=1)
        if not valid.any():
            continue
        exponentiated = similarity.exp()
        numerator = (exponentiated * positive.float()).sum(dim=1)
        denominator = (exponentiated * eligible.float()).sum(dim=1)
        task_loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8))
        losses.append(task_loss[valid])
        weights.append(reliability[valid, task])
        valid_anchor_count += int(valid.sum())
    if not losses:
        loss = projected.sum() * 0.0
    else:
        all_losses = torch.cat(losses)
        all_weights = torch.cat(weights).detach()
        loss = (all_losses * all_weights).sum() / all_weights.sum().clamp_min(1e-6)
    return loss, {
        "spcl_ratio": float(ratio),
        "spcl_coverage": float(selected.float().mean()),
        "spcl_valid_anchors": float(valid_anchor_count),
        "spcl_reliability": float(reliability.mean()),
    }


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def corpus_task_competitive_loss(
    outputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    labels: torch.Tensor,
    domains: torch.Tensor,
    sample_weight: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Choose logit teachers per corpus/task and feature teachers per corpus."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    cell_kd: list[torch.Tensor] = []
    corpus_feature_kd: list[torch.Tensor] = []
    winners: list[int] = []
    for domain in torch.unique(domains, sorted=True):
        mask = domains == domain
        weights = sample_weight[mask]
        task_losses: list[list[torch.Tensor]] = [[], []]
        for task in range(labels.shape[1]):
            for learner in range(2):
                per_sample_ce = F.cross_entropy(
                    outputs[learner]["logits"][mask, task], labels[mask, task],
                    reduction="none", label_smoothing=0.1,
                )
                task_losses[learner].append(_weighted_mean(per_sample_ce, weights))
            winner = int(
                task_losses[1][task].detach().item()
                < task_losses[0][task].detach().item()
            )
            winners.append(winner)
            loser = 1 - winner
            per_sample_kl = F.kl_div(
                F.log_softmax(
                    outputs[loser]["logits"][mask, task] / temperature, dim=-1
                ),
                F.softmax(
                    outputs[winner]["logits"][mask, task].detach() / temperature,
                    dim=-1,
                ),
                reduction="none",
            ).sum(dim=-1) * (temperature * temperature)
            cell_kd.append(_weighted_mean(per_sample_kl, weights))

        corpus_losses = [torch.stack(values).mean() for values in task_losses]
        feature_winner = int(
            corpus_losses[1].detach().item() < corpus_losses[0].detach().item()
        )
        feature_loser = 1 - feature_winner
        loser_feature = F.normalize(
            outputs[feature_loser]["fused_embedding"][mask], dim=-1
        )
        winner_feature = F.normalize(
            outputs[feature_winner]["fused_embedding"][mask].detach(), dim=-1
        )
        corpus_feature_kd.append(
            _weighted_mean((loser_feature - winner_feature).square().sum(-1), weights)
        )

    zero = outputs[0]["logits"].sum() * 0.0
    logit_kd = torch.stack(cell_kd).mean() if cell_kd else zero
    feature_kd = torch.stack(corpus_feature_kd).mean() if corpus_feature_kd else zero
    winner0 = sum(winner == 0 for winner in winners) / max(len(winners), 1)
    return logit_kd, feature_kd, {"winner0": winner0, "winner1": 1.0 - winner0}


def _true_class_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits.detach(), dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)


def sample_reliable_competitive_loss(
    outputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    labels: torch.Tensor,
    domains: torch.Tensor,
    sample_weight: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Transfer only where the selected teacher has lower true-label NLL."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logit_terms: list[torch.Tensor] = []
    feature_terms: list[torch.Tensor] = []
    coverages: list[torch.Tensor] = []
    teacher_errors: list[torch.Tensor] = []
    winners: list[int] = []

    for domain_tensor in torch.unique(domains, sorted=True):
        mask = domains == domain_tensor
        weights = sample_weight[mask]
        task_losses: list[list[torch.Tensor]] = [[], []]
        task_nll: list[list[torch.Tensor]] = [[], []]
        for task in range(labels.shape[1]):
            for learner in range(2):
                per_sample_ce = F.cross_entropy(
                    outputs[learner]["logits"][mask, task], labels[mask, task],
                    reduction="none", label_smoothing=0.1,
                )
                task_losses[learner].append(_weighted_mean(per_sample_ce, weights))
                task_nll[learner].append(
                    _true_class_nll(
                        outputs[learner]["logits"][mask, task], labels[mask, task]
                    )
                )
            winner = int(
                task_losses[1][task].detach().item()
                < task_losses[0][task].detach().item()
            )
            loser = 1 - winner
            winners.append(winner)
            reliable = task_nll[winner][task] <= task_nll[loser][task]
            coverages.append(reliable.float().mean())
            if bool(reliable.any()):
                selected_weights = weights[reliable]
                per_sample_kl = F.kl_div(
                    F.log_softmax(
                        outputs[loser]["logits"][mask, task][reliable]
                        / temperature,
                        dim=-1,
                    ),
                    F.softmax(
                        outputs[winner]["logits"][mask, task][reliable].detach()
                        / temperature,
                        dim=-1,
                    ),
                    reduction="none",
                ).sum(-1) * (temperature * temperature)
                logit_terms.append(_weighted_mean(per_sample_kl, selected_weights))
                teacher_errors.append(
                    _weighted_mean(
                        outputs[winner]["logits"][mask, task][reliable]
                        .detach().argmax(-1).ne(labels[mask, task][reliable]).float(),
                        selected_weights,
                    )
                )

        corpus_losses = [torch.stack(values).mean() for values in task_losses]
        feature_winner = int(
            corpus_losses[1].detach().item() < corpus_losses[0].detach().item()
        )
        feature_loser = 1 - feature_winner
        reliable_feature = (
            torch.stack(task_nll[feature_winner], dim=0).mean(0)
            <= torch.stack(task_nll[feature_loser], dim=0).mean(0)
        )
        coverages.append(reliable_feature.float().mean())
        if bool(reliable_feature.any()):
            loser_feature = F.normalize(
                outputs[feature_loser]["fused_embedding"][mask][reliable_feature],
                dim=-1,
            )
            winner_feature = F.normalize(
                outputs[feature_winner]["fused_embedding"][mask][reliable_feature]
                .detach(),
                dim=-1,
            )
            feature_terms.append(
                _weighted_mean(
                    (loser_feature - winner_feature).square().sum(-1),
                    weights[reliable_feature],
                )
            )

    zero = outputs[0]["logits"].sum() * 0.0
    logit_kd = torch.stack(logit_terms).mean() if logit_terms else zero
    feature_kd = torch.stack(feature_terms).mean() if feature_terms else zero
    winner0 = sum(winner == 0 for winner in winners) / max(len(winners), 1)
    return logit_kd, feature_kd, {
        "winner0": winner0,
        "winner1": 1.0 - winner0,
        "reliable_coverage": float(torch.stack(coverages).mean().detach()),
        "peer_teacher_error": (
            float(torch.stack(teacher_errors).mean().detach())
            if teacher_errors else 0.0
        ),
    }


def reliable_logit_full_geometry_loss(
    outputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    labels: torch.Tensor,
    domains: torch.Tensor,
    sample_weight: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Combine sample-gated logit KL with corpus-selected feature geometry.

    Logit teachers are selected independently for each corpus/task cell. The
    KL term is accepted only where the selected teacher has no larger
    true-class NLL. Feature teachers are selected per corpus and supervise all
    samples in that corpus, matching the method reported in the paper.
    """

    reliable_logit, _, diagnostics = sample_reliable_competitive_loss(
        outputs, labels, domains, sample_weight, temperature=temperature
    )
    _, full_geometry, selector = corpus_task_competitive_loss(
        outputs, labels, domains, sample_weight, temperature=temperature
    )
    diagnostics.update(winner0=selector["winner0"], winner1=selector["winner1"])
    return reliable_logit, full_geometry, diagnostics
