"""Build train-only GRAM targets for CRISP-Emo."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import types
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from constants import EEG_CHANNELS, EEG_MONTAGES
from dataset import PhysioWindowDataset


class _CheckpointDict(dict):
    """Minimal compatibility type required by the published GRAM checkpoint."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key: str, value) -> None:
        self[key] = value

    def __setstate__(self, state: dict) -> None:
        self.update(state)


def _install_addict_compatibility() -> None:
    if "addict.addict" in sys.modules:
        return
    package = types.ModuleType("addict")
    package.__path__ = []
    module = types.ModuleType("addict.addict")
    _CheckpointDict.__module__ = "addict.addict"
    module.Dict = _CheckpointDict
    package.Dict = _CheckpointDict
    sys.modules["addict"] = package
    sys.modules["addict.addict"] = module


class _TemporalConvolution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, 8)
        self.conv2 = nn.Conv2d(8, 8, (1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, 8)
        self.conv3 = nn.Conv2d(8, 8, (1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, 8)
        self.gelu3 = nn.GELU()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal = signal.unsqueeze(1)
        signal = self.gelu1(self.norm1(self.conv1(signal)))
        signal = self.gelu2(self.norm2(self.conv2(signal)))
        signal = self.gelu3(self.norm3(self.conv3(signal)))
        return signal.permute(0, 2, 3, 1).flatten(2)


class _Attention(nn.Module):
    def __init__(self, dim: int = 200, heads: int = 10, dropout: float = 0.05) -> None:
        super().__init__()
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, dim = tokens.shape
        head_dim = dim // self.heads
        key = self.key(tokens).view(batch, length, self.heads, head_dim).transpose(1, 2)
        query = self.query(tokens).view(batch, length, self.heads, head_dim).transpose(1, 2)
        value = self.value(tokens).view(batch, length, self.heads, head_dim).transpose(1, 2)
        attention = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
        attention = self.attn_drop(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).contiguous().view(batch, length, dim)
        return self.resid_drop(self.proj(output))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gamma_1 = nn.Parameter(torch.full((200,), 0.1))
        self.gamma_2 = nn.Parameter(torch.full((200,), 0.1))
        self.ln1 = nn.LayerNorm(200)
        self.ln2 = nn.LayerNorm(200)
        self.attn = _Attention()
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(200, 800)),
                    ("gelu", nn.GELU()),
                    ("fc2", nn.Linear(800, 200)),
                    ("dp", nn.Dropout(0.05)),
                ]
            )
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.gamma_1 * self.attn(self.ln1(tokens))
        return tokens + self.gamma_2 * self.mlp(self.ln2(tokens))


class _GramEncoder(nn.Module):
    def __init__(self, channel_ids: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("channel_ids", channel_ids, persistent=False)
        self.ch_emb = nn.Parameter(torch.zeros(1, 130, 200), requires_grad=False)
        self.time_emb = nn.Parameter(torch.zeros(1, 16, 200), requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 200))
        self.proj_weights = nn.Parameter(torch.ones(6, 1, 1, 1))
        self.tconv = _TemporalConvolution()
        self.blocks = nn.ModuleList([_Block() for _ in range(12)])
        self.norm = nn.LayerNorm(200)
        self.proj_layers = nn.ModuleList([nn.Linear(200, 200) for _ in range(5)])
        self.out_indices = (0, 2, 4, 6, 8, 11)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch, token_count, _ = patches.shape
        channels = self.channel_ids.numel()
        time_steps = token_count // channels
        if token_count % channels or time_steps > 16:
            raise ValueError("GRAM input must contain complete channel sets and <=16 patches")
        channel_embedding = self.ch_emb[:, self.channel_ids]
        channel_embedding = channel_embedding.unsqueeze(1).expand(
            batch, time_steps, channels, -1
        ).flatten(1, 2)
        time_embedding = self.time_emb[:, :time_steps].unsqueeze(2).expand(
            batch, time_steps, channels, -1
        ).flatten(1, 2)
        tokens = self.tconv(patches) + channel_embedding + time_embedding
        tokens = torch.cat([self.cls_token.expand(batch, -1, -1), tokens], dim=1)
        levels = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in self.out_indices:
                levels.append(self.proj_layers[len(levels)](tokens) if index != 11 else tokens)
        fused = torch.stack(levels).mul(self.proj_weights.softmax(dim=0)).sum(dim=0)
        return self.norm(fused)[:, 0]


class FrozenGramTeacher(nn.Module):
    """Dependency-free inference adapter for the official GRAM-B checkpoint."""

    output_dim = 200
    sample_rate = 200
    patch_samples = 200

    def __init__(self, weights_path: Path, eeg_channel_indices: tuple[int, ...]) -> None:
        super().__init__()
        if not weights_path.is_file():
            raise FileNotFoundError(f"GRAM weights not found: {weights_path}")
        _install_addict_compatibility()
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        all_channels = [name.upper() for name in checkpoint["cf"].all_ch_list]
        if not eeg_channel_indices or len(set(eeg_channel_indices)) != len(eeg_channel_indices):
            raise ValueError("EEG channel indices must be non-empty and unique")
        self.eeg_channel_indices = tuple(eeg_channel_indices)
        channel_ids = torch.tensor(
            [all_channels.index(EEG_CHANNELS[index].upper()) for index in eeg_channel_indices],
            dtype=torch.long,
        )
        model = _GramEncoder(channel_ids)
        source = checkpoint["model"]
        target = model.state_dict()
        compatible = {key: source[key] for key in target if key in source}
        missing = sorted(set(target) - set(compatible))
        if missing:
            raise RuntimeError(f"GRAM encoder weights are missing: {missing}")
        model.load_state_dict(compatible, strict=True)
        model.requires_grad_(False)
        model.eval()
        self.model = model

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def forward(self, eeg: torch.Tensor, source_rate: int = 128) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1] != len(self.eeg_channel_indices):
            raise ValueError(
                f"Expected EEG [batch,{len(self.eeg_channel_indices)},samples], "
                f"got {tuple(eeg.shape)}"
            )
        samples = int(round(eeg.shape[-1] * self.sample_rate / source_rate))
        if samples % self.patch_samples:
            raise ValueError("Resampled EEG duration must be a whole number of seconds")
        signal = F.interpolate(eeg, size=samples, mode="linear", align_corners=False)
        patches = signal.unfold(-1, self.patch_samples, self.patch_samples)
        patches = patches.permute(0, 2, 1, 3).flatten(1, 2).contiguous()
        return self.model(patches).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gram-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_fingerprint(dataset: PhysioWindowDataset) -> str:
    digest = hashlib.sha256()
    for record in dataset.windows:
        digest.update(
            f"{record.dataset}:{record.subject}:{record.h5_key}:"
            f"{record.start_seconds:.6f};".encode()
        )
    return digest.hexdigest()


def _embed(
    dataset: PhysioWindowDataset,
    teacher: FrozenGramTeacher,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    chunks = []
    for index, batch in enumerate(loader, start=1):
        chunks.append(teacher(batch["eeg"].to(device, non_blocking=True)).cpu().half())
        if index == 1 or index % 25 == 0 or index == len(loader):
            print(f"teacher batch {index}/{len(loader)}", flush=True)
    return torch.cat(chunks)


def _dataset(
    root: Path,
    fold: int,
    datasets: tuple[str, ...],
    montage: str,
) -> PhysioWindowDataset:
    return PhysioWindowDataset(
        root=root,
        fold=fold,
        partition="train",
        datasets=datasets,
        window_seconds=5.0,
        stride_seconds=2.5,
        threshold=5.0,
        neutral_policy="low",
        eeg_normalization="physical",
        eeg_channel_indices=EEG_MONTAGES[montage],
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if not (args.data_root / "preprocessing_audit.json").is_file():
        raise RuntimeError("Missing preprocessing_audit.json; run preprocess_data.py first")

    combined = _dataset(args.data_root, args.fold, ("deap", "dreamer"), "periphery10")
    deap = _dataset(args.data_root, args.fold, ("deap",), "full32")
    dreamer = _dataset(args.data_root, args.fold, ("dreamer",), "shared14")
    if len(deap) + len(dreamer) != len(combined):
        raise RuntimeError("Corpus-specific and combined training windows do not align")
    expected = [
        (record.dataset, record.subject, record.h5_key, record.start_seconds)
        for record in combined.windows
    ]
    actual = [
        (record.dataset, record.subject, record.h5_key, record.start_seconds)
        for record in [*deap.windows, *dreamer.windows]
    ]
    if actual != expected:
        raise RuntimeError("Teacher datasets do not preserve combined window order")

    device = torch.device(args.device)
    print("Embedding DEAP Full32 privileged view", flush=True)
    teacher = FrozenGramTeacher(args.gram_weights, EEG_MONTAGES["full32"]).to(device)
    deap_privileged = _embed(
        deap, teacher, device, args.batch_size, args.num_workers
    )
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Embedding DREAMER native Shared14 privileged view", flush=True)
    teacher = FrozenGramTeacher(args.gram_weights, EEG_MONTAGES["shared14"]).to(device)
    dreamer_privileged = _embed(
        dreamer, teacher, device, args.batch_size, args.num_workers
    )
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Embedding shared Periphery10 deployable view", flush=True)
    teacher = FrozenGramTeacher(args.gram_weights, EEG_MONTAGES["periphery10"]).to(device)
    deployable = _embed(
        combined, teacher, device, args.batch_size, args.num_workers
    )
    privileged = torch.cat([deap_privileged, dreamer_privileged])
    if privileged.shape != deployable.shape or privileged.shape != (len(combined), 200):
        raise RuntimeError((privileged.shape, deployable.shape, len(combined)))

    payload = {
        "teacher": "GRAM-B",
        "representation": (
            "DEAP Full32 or DREAMER native Shared14 privileged view, "
            "plus shared Periphery10 deployable view"
        ),
        "embeddings": torch.stack([privileged, deployable], dim=1),
        "strictly_causal_signal_preprocessing": True,
        "partition": "train",
        "fold": args.fold,
        "test_or_validation_embeddings_built": False,
        "train_subjects": sorted(
            {f"{record.dataset}:{record.subject}" for record in combined.windows}
        ),
        "window_fingerprint": _window_fingerprint(combined),
        "splits_sha256": _sha256(args.data_root / "splits.json"),
        "manifest_sha256": _sha256(args.data_root / "manifest.csv"),
        "teacher_weights_sha256": _sha256(args.gram_weights),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(f"Saved {tuple(payload['embeddings'].shape)} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
