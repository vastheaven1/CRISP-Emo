"""Evaluate a validation-selected CRISP-Emo checkpoint on the fixed holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))

from constants import EEG_MONTAGES  # noqa: E402
from model import CRISPEmo  # noqa: E402
from evaluation import evaluate_model  # noqa: E402
from evaluation import primary_scores  # noqa: E402
from dataset import SensorAwareWindowDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path,
        help="Override the data path recorded in the training summary.",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    output_path = args.run / "development_holdout_evaluation.json"
    if output_path.exists():
        raise FileExistsError(f"Evaluation already exists: {output_path}")
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    eligible = {"crisp_emo_five_objective_golu_fusion", "v201_golu_fusion_v188"}
    if config.get("model") not in eligible:
        raise RuntimeError("Run is not an eligible formal CRISP-Emo checkpoint")
    if summary.get("test") is not None or not summary["protocol"][
        "selection_uses_only_validation"
    ]:
        raise RuntimeError("Checkpoint must have been selected without test data")
    data_root = args.data_root or Path(config["data_root"])
    if not data_root.is_absolute():
        candidates = [
            (root / data_root).resolve()
            for root in (Path.cwd(), HERE, HERE.parent, HERE.parents[1])
        ]
        data_root = next((path for path in candidates if path.exists()), candidates[0])
    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root}. Pass --data-root explicitly."
        )
    device = torch.device(args.device)
    fold = int(config.get("fold", 0))
    eeg_montage = config.get("eeg_montage", "periphery10")
    data = SensorAwareWindowDataset(
        data_root, "test", fold=fold, eeg_montage=eeg_montage
    )
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = CRISPEmo(
        eeg_channels=len(EEG_MONTAGES[eeg_montage]),
        embed_dim=64,
        fusion_layers=1,
        modality_dropout=0.10,
    ).to(device).eval()
    checkpoint = torch.load(
        args.run / "checkpoints" / "best.pt",
        map_location=device,
        weights_only=False,
    )
    retained = model.state_dict()
    filtered = {key: value for key, value in checkpoint["model"].items() if key in retained}
    incompatible = model.load_state_dict(filtered)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(incompatible)
    report = evaluate_model(
        model,
        loader,
        device,
        max_batches=None,
        decision_thresholds=summary["validation"]["decision_thresholds"],
        fit_thresholds=False,
    )
    primary = primary_scores(report)
    payload = {
        "protocol": "validation-selected CRISP-Emo evaluated once on fixed development holdout",
        "run": str(args.run),
        "seed": config["seed"],
        "fold": fold,
        "eeg_montage": eeg_montage,
        "eeg_channel_count": len(EEG_MONTAGES[eeg_montage]),
        "best_epoch_selected_by_validation": summary["best_epoch"],
        "branch_selected_by_validation": summary["selected_branch"],
        "development_holdout": report,
        "primary_development_holdout": primary,
        "thresholds_frozen_from_validation": True,
        "holdout_selected_checkpoint_branch_or_threshold": False,
        "holdout_used_as_primary_method_result": True,
        "inference_ensemble": False,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seed": config["seed"], **primary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
