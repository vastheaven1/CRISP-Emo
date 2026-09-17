"""CRISP-Emo model."""

from __future__ import annotations

import torch
from torch import nn


class PatchStem(nn.Module):
    """Encode non-overlapping one-second patches for one physical sensor."""

    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        patches = signal.unfold(-1, self.patch_size, self.patch_size)
        batch, channels, steps, width = patches.shape
        encoded = self.encoder(patches.reshape(batch * channels * steps, 1, width))
        return encoded.reshape(batch, channels, steps, -1)


class TemporalChannelPool(nn.Module):
    """Pool physical sensor leads independently at each one-second step."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, channels, steps, width = tokens.shape
        context = tokens.permute(0, 2, 1, 3).reshape(batch * steps, channels, width)
        query = self.query.expand(batch * steps, -1, -1)
        normalized_context = self.context_norm(context)
        attended, _ = self.attention(
            self.query_norm(query),
            normalized_context,
            normalized_context,
            need_weights=False,
        )
        pooled = self.output_norm(query + attended)
        return pooled.reshape(batch, steps, width)


class SeparableConv2d(nn.Module):
    def __init__(self, channels: int, kernel_length: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, (1, kernel_length), groups=channels,
            bias=False, padding="same",
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(value))


class EEGNetTemporalEncoder(nn.Module):
    """Official 128-Hz EEGNet core, exposed as five current-window tokens."""

    def __init__(
        self,
        channels: int = 10,
        embed_dim: int = 64,
        temporal_filters: int = 8,
        spatial_filters: int = 4,
        dropout: float = 0.50,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = channels
        hidden = temporal_filters * spatial_filters
        self.conv1 = nn.Conv2d(
            1, temporal_filters, (1, 64), padding="same", bias=False
        )
        # Per-window GroupNorm avoids pooling running statistics across source
        # subjects and corpora; it has no train/eval running state.
        self.batchnorm1 = nn.GroupNorm(temporal_filters, temporal_filters)
        self.spatial = nn.Conv2d(
            temporal_filters, hidden, (channels, 1),
            groups=temporal_filters, bias=False,
        )
        self.batchnorm2 = nn.GroupNorm(8, hidden)
        self.pooling2 = nn.AvgPool2d((1, 4))
        self.separable = SeparableConv2d(hidden, 16)
        self.batchnorm3 = nn.GroupNorm(8, hidden)
        self.pooling3 = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(dropout)
        self.token_projection = nn.Linear(hidden, embed_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1:] != (self.channels, 640):
            raise ValueError(
                f"EEGNet expects [batch,{self.channels},640], got {tuple(eeg.shape)}"
            )
        value = self.batchnorm1(self.conv1(eeg[:, None]))
        value = torch.nn.functional.elu(self.batchnorm2(self.spatial(value)))
        value = self.dropout(self.pooling2(value))
        value = torch.nn.functional.elu(self.batchnorm3(self.separable(value)))
        value = self.dropout(self.pooling3(value))
        # [B, 32, 1, 20] -> four adjacent positions per one-second token.
        value = value.squeeze(2).transpose(1, 2).contiguous()
        if value.shape[1] != 20:
            raise RuntimeError(f"Unexpected EEGNet temporal length: {value.shape[1]}")
        value = value.reshape(value.shape[0], 5, 4, value.shape[2]).mean(dim=2)
        return self.token_projection(value)


class EEGNetSensorAwareFusionTiny(nn.Module):
    """Periphery10 EEGNet + physical-sensor-aware genuine multimodal fusion."""

    MODEL_VERSION = "eegnet_sensor_aware_fusion"

    def __init__(
        self,
        eeg_channels: int = 10,
        embed_dim: int = 64,
        num_heads: int = 4,
        fusion_layers: int = 1,
        modality_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if eeg_channels <= 0:
            raise ValueError("eeg_channels must be positive")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.modality_dropout = modality_dropout
        self.eeg_encoder = EEGNetTemporalEncoder(eeg_channels, embed_dim)
        self.eda_stem = PatchStem(32, embed_dim)
        self.temperature_stem = PatchStem(8, embed_dim)
        self.ecg_stem = PatchStem(256, embed_dim)
        self.sensor_embedding = nn.Parameter(torch.randn(2, embed_dim) * 0.02)
        self.ecg_lead_pool = TemporalChannelPool(embed_dim, num_heads)
        self.deap_sensor_fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 2), nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )
        self.autonomic_adapter = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU()
        )
        self.eeg_from_pps = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.pps_from_eeg = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.cross_fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 4), nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(embed_dim * 2, embed_dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.fusion_type_embedding = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=fusion_layers)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.joint_heads = nn.ModuleList([nn.Linear(embed_dim, 2) for _ in range(2)])
        self.eeg_heads = nn.ModuleList([nn.Linear(embed_dim, 2) for _ in range(2)])
        self.pps_heads = nn.ModuleList([nn.Linear(embed_dim, 2) for _ in range(2)])
        self.rating_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2), nn.Sigmoid()
        )
        alignment_dim = embed_dim // 2
        self.eeg_alignment = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, alignment_dim)
        )
        self.pps_alignment = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, alignment_dim)
        )

    def _sensor_tokens(
        self, eda: torch.Tensor, temperature: torch.Tensor,
        ecg: torch.Tensor, sensor_type: torch.Tensor,
    ) -> torch.Tensor:
        eda_tokens = self.eda_stem(eda)[:, 0]
        temperature_tokens = self.temperature_stem(temperature)[:, 0]
        deap_tokens = self.deap_sensor_fusion(torch.cat([eda_tokens, temperature_tokens], dim=-1))
        ecg_tokens = self.ecg_lead_pool(self.ecg_stem(ecg))
        selector = sensor_type.float()[:, None, None]
        tokens = deap_tokens * (1.0 - selector) + ecg_tokens * selector
        return self.autonomic_adapter(tokens + self.sensor_embedding[sensor_type][:, None])

    def _modality_dropout(
        self, eeg: torch.Tensor, pps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return eeg, pps
        choice = torch.rand(eeg.shape[0], device=eeg.device)
        eeg_drop = choice < self.modality_dropout * 0.5
        pps_drop = (choice >= self.modality_dropout * 0.5) & (choice < self.modality_dropout)
        return (
            eeg.masked_fill(eeg_drop[:, None, None], 0.0),
            pps.masked_fill(pps_drop[:, None, None], 0.0),
        )

    def encode_modalities(
        self, eeg: torch.Tensor, eda: torch.Tensor, temperature: torch.Tensor,
        ecg: torch.Tensor, sensor_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both input modalities into five temporal tokens."""
        eeg_temporal = self.eeg_encoder(eeg)
        pps_temporal = self._sensor_tokens(eda, temperature, ecg, sensor_type)
        if eeg_temporal.shape[1] != pps_temporal.shape[1]:
            raise RuntimeError("EEGNet and PPS token counts must both equal five")
        return eeg_temporal, pps_temporal

    def fuse_modality_tokens(
        self, eeg_temporal: torch.Tensor, pps_temporal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the unchanged nonlinear joint fusion from modality tokens."""
        if eeg_temporal.ndim != 3 or eeg_temporal.shape != pps_temporal.shape:
            raise ValueError((tuple(eeg_temporal.shape), tuple(pps_temporal.shape)))
        eeg_context, _ = self.eeg_from_pps(
            eeg_temporal, pps_temporal, pps_temporal, need_weights=False
        )
        pps_context, _ = self.pps_from_eeg(
            pps_temporal, eeg_temporal, eeg_temporal, need_weights=False
        )
        cross_temporal = self.cross_fusion(
            torch.cat([
                eeg_temporal + eeg_context, pps_temporal + pps_context,
                eeg_temporal * pps_temporal, (eeg_temporal - pps_temporal).abs(),
            ], dim=-1)
        )
        eeg_embedding = eeg_temporal.mean(dim=1)
        pps_embedding = pps_temporal.mean(dim=1)
        cross_embedding = cross_temporal.mean(dim=1)
        cls = self.cls_token.expand(eeg_temporal.shape[0], -1, -1)
        fusion_tokens = torch.cat([
            cls, eeg_embedding[:, None], pps_embedding[:, None], cross_embedding[:, None]
        ], dim=1)
        fused_embedding = self.final_norm(
            self.fusion(fusion_tokens + self.fusion_type_embedding)[:, 0]
        )
        return {
            "logits": torch.stack([head(fused_embedding) for head in self.joint_heads], dim=1),
            "eeg_logits": torch.stack([head(eeg_embedding) for head in self.eeg_heads], dim=1),
            "pps_logits": torch.stack([head(pps_embedding) for head in self.pps_heads], dim=1),
            "rating_predictions": self.rating_head(fused_embedding),
            "eeg_embedding": eeg_embedding,
            "pps_embedding": pps_embedding,
            "cross_embedding": cross_embedding,
            "fused_embedding": fused_embedding,
            "eeg_alignment": self.eeg_alignment(eeg_embedding),
            "pps_alignment": self.pps_alignment(pps_embedding),
            "eeg_tokens": eeg_temporal,
            "pps_tokens": pps_temporal,
        }

    def forward(
        self, eeg: torch.Tensor, eda: torch.Tensor, temperature: torch.Tensor,
        ecg: torch.Tensor, sensor_type: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eeg_temporal, pps_temporal = self.encode_modalities(
            eeg, eda, temperature, ecg, sensor_type
        )
        eeg_temporal, pps_temporal = self._modality_dropout(
            eeg_temporal, pps_temporal
        )
        return self.fuse_modality_tokens(eeg_temporal, pps_temporal)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class GoLU(nn.Module):
    """Parameter-free standard Gompertz Linear Unit."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.exp(-torch.exp(-value))


def _replace_gelu(module: nn.Module) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GELU):
            setattr(module, name, GoLU())
            count += 1
        else:
            count += _replace_gelu(child)
    return count


class _CheckpointCompatibleGoLUFusion(EEGNetSensorAwareFusionTiny):
    """Preserve the initialization stream used by the reported runs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.explicit_gelu_replacements = _replace_gelu(self)
        for layer in self.fusion.layers:
            layer.activation = GoLU()
            layer.activation_relu_or_gelu = 0


class CRISPEmo(_CheckpointCompatibleGoLUFusion):
    """Reported CRISP-Emo deployment graph with inactive heads removed."""

    MODEL_VERSION = "crisp_emo_five_objective_golu_fusion"

    def __init__(self, *args, **kwargs) -> None:
        # Preserve the reported seeded initialization stream, then remove the
        # inactive auxiliary heads before optimization and deployment.
        super().__init__(*args, **kwargs)
        del self.eeg_heads
        del self.pps_heads
        del self.rating_head
        del self.eeg_alignment
        del self.pps_alignment

    def fuse_modality_tokens(
        self,
        eeg_temporal: torch.Tensor,
        pps_temporal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if eeg_temporal.ndim != 3 or eeg_temporal.shape != pps_temporal.shape:
            raise ValueError(
                (tuple(eeg_temporal.shape), tuple(pps_temporal.shape))
            )
        eeg_context, _ = self.eeg_from_pps(
            eeg_temporal,
            pps_temporal,
            pps_temporal,
            need_weights=False,
        )
        pps_context, _ = self.pps_from_eeg(
            pps_temporal,
            eeg_temporal,
            eeg_temporal,
            need_weights=False,
        )
        cross_temporal = self.cross_fusion(
            torch.cat(
                [
                    eeg_temporal + eeg_context,
                    pps_temporal + pps_context,
                    eeg_temporal * pps_temporal,
                    (eeg_temporal - pps_temporal).abs(),
                ],
                dim=-1,
            )
        )
        eeg_embedding = eeg_temporal.mean(dim=1)
        pps_embedding = pps_temporal.mean(dim=1)
        cross_embedding = cross_temporal.mean(dim=1)
        cls = self.cls_token.expand(eeg_temporal.shape[0], -1, -1)
        tokens = torch.cat(
            [
                cls,
                eeg_embedding[:, None],
                pps_embedding[:, None],
                cross_embedding[:, None],
            ],
            dim=1,
        )
        fused_embedding = self.final_norm(
            self.fusion(tokens + self.fusion_type_embedding)[:, 0]
        )
        return {
            "logits": torch.stack(
                [head(fused_embedding) for head in self.joint_heads], dim=1
            ),
            "fused_embedding": fused_embedding,
        }

# Third-party notice for the EEGNet-derived portion above (not the remaining code):
# MIT License
# Copyright (c) 2025 LIUYIN YANG
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
