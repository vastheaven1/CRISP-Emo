from __future__ import annotations

"""Train CRISP-Emo."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from constants import EEG_CHANNELS, EEG_MONTAGES
from evaluation import evaluate_model, forward_model, move_batch, primary_scores
from model import CRISPEmo
from losses import (
    ArchitectureAgnosticDecoupler,
    SelfPacedProjector,
    TeacherViewDataset,
    TrialReliabilityMemory,
    corpus_separated_rsd_loss,
    reliable_logit_full_geometry_loss,
    self_paced_ratio,
    trial_aware_self_paced_contrastive_loss,
)
from dataset import SensorAwareWindowDataset, weighted_classification_loss


METHOD_NAME = "crisp_emo_five_objective_golu_fusion"
TEACHER_RULE = "corpus-task winner with reliable sample logit gate"
LOSER_OBJECTIVE = (
    "reliable peer-logit KL plus all-sample normalized fused-feature geometry"
)
VALIDATION_SELECTION = (
    "predeclared branch0; validation selects epoch and thresholds only"
)

def select_validation_branch(primary: list[dict[str, float]]) -> int:
    """The deployment anchor is declared before optimization."""
    del primary
    return 0


def competitive_step(
    outputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    parts: tuple[dict[str, torch.Tensor | float], dict[str, torch.Tensor | float]],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Apply the paper's corpus/task-aware peer selector."""
    del parts
    return reliable_logit_full_geometry_loss(
        outputs, batch["labels"], batch["domain"], batch["sample_weight"],
        temperature=args.peer_temperature,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--rsd-weight", type=float, default=0.10)
    parser.add_argument("--rsd-kappa", type=float, default=0.01)
    parser.add_argument("--spcl-weight", type=float, default=0.02)
    parser.add_argument("--spcl-temperature", type=float, default=0.12)
    parser.add_argument("--spcl-warmup-epochs", type=int, default=0)
    parser.add_argument("--spcl-ramp-epochs", type=int, default=6)
    parser.add_argument("--spcl-start-ratio", type=float, default=0.30)
    parser.add_argument("--spcl-end-ratio", type=float, default=0.80)
    parser.add_argument("--trial-ema-decay", type=float, default=0.80)
    parser.add_argument("--peer-logit-weight", type=float, default=1.0)
    parser.add_argument("--peer-feature-weight", type=float, default=0.10)
    parser.add_argument("--peer-temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_learner(
    device: torch.device, eeg_channels: int
) -> tuple[nn.Module, nn.Module, nn.Module]:
    model = CRISPEmo(
        eeg_channels=eeg_channels, embed_dim=64, fusion_layers=1, modality_dropout=0.10
    ).to(device)
    return (
        model,
        ArchitectureAgnosticDecoupler().to(device),
        SelfPacedProjector().to(device),
    )


def learner_base_loss(
    model: nn.Module,
    aad: nn.Module,
    projector: nn.Module,
    reliability_memory: TrialReliabilityMemory,
    batch: dict[str, torch.Tensor],
    teacher_views: torch.Tensor,
    ratio: float,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | float]]:
    output = forward_model(model, batch)
    classification = weighted_classification_loss(
        output["logits"], batch["labels"], batch["sample_weight"],
        label_smoothing=0.1,
    )
    rsd = corpus_separated_rsd_loss(
        aad(output["fused_embedding"]), teacher_views,
        batch["domain"], batch["sample_weight"], args.rsd_kappa,
    )
    spcl, diagnostics = trial_aware_self_paced_contrastive_loss(
        projector(output["fused_embedding"]), output["logits"],
        batch["labels"], batch["domain"], batch["subject_id"], batch["trial_id"],
        reliability_memory, ratio, args.spcl_temperature,
    )
    total = classification + args.rsd_weight * rsd + args.spcl_weight * spcl
    return output, {
        "base": total,
        "classification": classification,
        "rsd": rsd,
        "spcl": spcl,
        **diagnostics,
    }


def validate_teacher_cache(
    payload: dict, dataset: SensorAwareWindowDataset, fold: int
) -> None:
    if payload.get("strictly_causal_signal_preprocessing") is not True:
        raise RuntimeError("CRISP-Emo requires the strict-causal main-train GRAM cache")
    if payload.get("partition") != "train" or payload.get("test_or_validation_embeddings_built"):
        raise RuntimeError("Teacher cache must contain main-train embeddings only")
    expected = sorted({f"{record.dataset}:{record.subject}" for record in dataset.windows})
    if payload.get("train_subjects") != expected or payload.get("fold") != fold:
        raise RuntimeError("Teacher cache does not match this training fold")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    base_train = SensorAwareWindowDataset(
        args.data_root, "train", fold=args.fold, eeg_montage="periphery10"
    )
    payload = torch.load(args.teacher_cache, map_location="cpu", weights_only=False)
    validate_teacher_cache(payload, base_train, args.fold)
    train_data = TeacherViewDataset(base_train, payload["embeddings"])
    validation_data = SensorAwareWindowDataset(
        args.data_root, "val", fold=args.fold, eeg_montage="periphery10"
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    eeg_channels = len(EEG_MONTAGES["periphery10"])
    set_seed(args.seed)
    learner0 = build_learner(device, eeg_channels)
    set_seed(args.seed + 10_000)
    learner1 = build_learner(device, eeg_channels)
    learners = (learner0, learner1)
    parameters = [parameter for learner in learners for module in learner for parameter in module.parameters()]
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    memories = (
        TrialReliabilityMemory(decay=args.trial_ema_decay),
        TrialReliabilityMemory(decay=args.trial_ema_decay),
    )
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.results_root / f"{timestamp}__{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    model0 = learners[0][0]
    config = {
        **vars(args),
        "data_root": str(args.data_root.resolve()),
        "teacher_cache": str(args.teacher_cache.resolve()),
        "results_root": str(args.results_root.resolve()),
        "model": METHOD_NAME,
        "base_model": model0.MODEL_VERSION,
        "fold": args.fold,
        "eeg_montage": "periphery10",
        "eeg_channel_names": [
            EEG_CHANNELS[index] for index in EEG_MONTAGES["periphery10"]
        ],
        "eeg_channel_count": eeg_channels,
        "peer_initialization_seeds": [args.seed, args.seed + 10_000],
        "dynamic_teacher_rule": TEACHER_RULE,
        "winner_objective": "RSD + SPCL base objective",
        "loser_objective": LOSER_OBJECTIVE,
        "validation_selection": VALIDATION_SELECTION,
        "test_windows_during_training": 0,
        "deployment_models": 1,
        "deployment_parameters": model0.parameter_count(),
        "training_models": 2,
        "inference_ensemble": False,
        "inference_postprocessing": False,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    keys = (
        "total", "base0", "base1", "classification0", "classification1",
        "rsd0", "rsd1", "spcl0", "spcl1", "peer_logit", "peer_feature",
        "winner0", "winner1", "reliable_coverage",
        "peer_teacher_error",
        "spcl_coverage0", "spcl_coverage1",
    )
    history: list[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    best_branch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        for learner in learners:
            for module in learner:
                module.train()
        ratio = self_paced_ratio(
            epoch, args.spcl_warmup_epochs, args.spcl_ramp_epochs,
            args.spcl_start_ratio, args.spcl_end_ratio,
        )
        sums = {key: 0.0 for key in keys}
        batches = 0
        for batch_index, raw in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            batch = move_batch(raw, device, include_teacher=False)
            teacher_views = raw["rsd_teacher_views"].to(device, non_blocking=True)
            output0, parts0 = learner_base_loss(
                *learner0, memories[0], batch, teacher_views, ratio, args
            )
            output1, parts1 = learner_base_loss(
                *learner1, memories[1], batch, teacher_views, ratio, args
            )
            outputs = (output0, output1)
            peer_logit, peer_feature, competition = competitive_step(
                outputs, (parts0, parts1), batch, args
            )
            total = (
                parts0["base"] + parts1["base"]
                + args.peer_logit_weight * peer_logit
                + args.peer_feature_weight * peer_feature
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            values = {
                "total": total,
                "base0": parts0["base"],
                "base1": parts1["base"],
                "classification0": parts0["classification"],
                "classification1": parts1["classification"],
                "rsd0": parts0["rsd"],
                "rsd1": parts1["rsd"],
                "spcl0": parts0["spcl"],
                "spcl1": parts1["spcl"],
                "peer_logit": peer_logit,
                "peer_feature": peer_feature,
                "winner0": competition["winner0"],
                "winner1": competition["winner1"],
                "reliable_coverage": competition.get("reliable_coverage", 0.0),
                "peer_teacher_error": competition.get("peer_teacher_error", 0.0),
                "spcl_coverage0": parts0["spcl_coverage"],
                "spcl_coverage1": parts1["spcl_coverage"],
            }
            for key, value in values.items():
                sums[key] += float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
            batches += 1

        validations = [
            evaluate_model(
                learner[0], validation_loader, device, args.max_eval_batches,
                fit_thresholds=True,
            )
            for learner in learners
        ]
        primary = [primary_scores(report) for report in validations]
        selected_branch = select_validation_branch(primary)
        selected_score = primary[selected_branch]["accuracy_f1_mean"]
        record = {
            "epoch": epoch,
            "training": {key: value / max(batches, 1) for key, value in sums.items()},
            "validation_by_branch": validations,
            "primary_validation_by_branch": primary,
            "epoch_selected_branch": selected_branch,
        }
        history.append(record)
        with (run_dir / "finetune_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(
            f"epoch={epoch} total={record['training']['total']:.5f} "
            f"peer_kl={record['training']['peer_logit']:.5f} "
            f"peer_feat={record['training']['peer_feature']:.5f} "
            f"winner0={record['training']['winner0']:.3f} "
            f"reliable={record['training']['reliable_coverage']:.3f} "
            f"b0={primary[0]['accuracy']:.6f}/{primary[0]['macro_f1']:.6f} "
            f"b1={primary[1]['accuracy']:.6f}/{primary[1]['macro_f1']:.6f}",
            flush=True,
        )
        if selected_score > best_score + 1e-8:
            best_score = selected_score
            best_epoch = epoch
            best_branch = selected_branch
            stale = 0
            torch.save(
                {
                    "model": learners[selected_branch][0].state_dict(),
                    "selected_branch": selected_branch,
                    "epoch": epoch,
                    "validation": validations[selected_branch],
                    "primary_validation": primary[selected_branch],
                    "validation_by_branch": validations,
                    "primary_validation_by_branch": primary,
                },
                run_dir / "checkpoints" / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(
        run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False
    )
    summary = {
        "config": config,
        "best_epoch": best_epoch,
        "selected_branch": best_branch,
        "validation": checkpoint["validation"],
        "primary_validation": checkpoint["primary_validation"],
        "validation_by_branch": checkpoint["validation_by_branch"],
        "primary_validation_by_branch": checkpoint["primary_validation_by_branch"],
        "test": None,
        "finetuning": history,
        "parameters": {
            "deployment": model0.parameter_count(),
            "training_models": 2 * model0.parameter_count(),
            "training_only_aads": 2 * sum(p.numel() for p in learner0[1].parameters()),
            "training_only_spcl_projectors": 2 * sum(p.numel() for p in learner0[2].parameters()),
        },
        "protocol": {
            "train_windows": len(train_data),
            "validation_windows": len(validation_data),
            "test_constructed_or_evaluated": False,
            "complete_pipeline_seed": args.seed,
            "selection_uses_only_validation": True,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"run_dir={run_dir} best_epoch={best_epoch} branch={best_branch} "
        f"primary={best_score:.6f}", flush=True,
    )


if __name__ == "__main__":
    main()
