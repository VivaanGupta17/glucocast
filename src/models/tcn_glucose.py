"""
Temporal Convolutional Network (TCN) for glucose prediction.

Implements dilated causal convolutions in the style of WaveNet
(van den Oord et al., 2016) adapted for multi-horizon CGM forecasting.

Key advantages for CGM:
  - Receptive field grows exponentially with depth, capturing both
    short-term CGM kinetics (5-30 min) and long-term insulin action (3-6 hrs)
  - Fully parallelisable training (unlike LSTM sequential dependency)
  - Residual connections prevent gradient vanishing over deep stacks

Receptive field calculation:
  RF = 1 + sum_i(2 * (kernel_size - 1) * dilation_i)
  With kernel=3, dilations=[1,2,4,8,16,32,64,128]: RF = 1 + 2*2*(255) = 1021 steps
  At 5-min CGM intervals: RF ≈ 85 hours — covers full IOB + meal curves

References:
  - Bai et al. (2018). An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling.
    https://arxiv.org/abs/1803.01271
  - van den Oord et al. (2016). WaveNet: A Generative Model for Raw Audio.
    https://arxiv.org/abs/1609.03499
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D causal convolution with zero-padding on the left.

    Causal: output at position t depends only on inputs at positions ≤ t.
    This is essential for inference — we cannot look into the future.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,   # Manual causal padding below
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad to maintain causal property
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class DilatedResidualBlock(nn.Module):
    """
    WaveNet-style dilated residual block.

    Structure:
        Input ──► CausalConv(dilation) ──► WeightNorm ──► Gated Tanh+Sigmoid ──►
               ──► Dropout ──► 1x1 Conv ──► + Residual ──► Output
                                              └──────────► Skip
    The gated activation (tanh * sigmoid) is borrowed from WaveNet and acts
    as a learned gate — analogous to the LSTM gate but operating on channels.
    """

    def __init__(
        self,
        n_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.dilation = dilation

        # Gated activation requires 2x channels before splitting
        self.dilated_conv = CausalConv1d(n_channels, n_channels * 2, kernel_size, dilation)
        if use_weight_norm:
            self.dilated_conv.conv = nn.utils.weight_norm(self.dilated_conv.conv)

        self.residual_conv = nn.Conv1d(n_channels, n_channels, 1)
        self.skip_conv = nn.Conv1d(n_channels, n_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, T] — channels-first format

        Returns:
            residual_out: [B, C, T] — for next block
            skip_out:     [B, C, T] — accumulated for output
        """
        residual = x

        # Dilated causal conv + gated activation
        conv_out = self.dilated_conv(x)   # [B, 2C, T]
        gate_in, gate_sigmoid = conv_out.chunk(2, dim=1)
        gated = torch.tanh(gate_in) * torch.sigmoid(gate_sigmoid)  # [B, C, T]
        gated = self.dropout(gated)

        # Skip connection
        skip = self.skip_conv(gated)      # [B, C, T]

        # Residual connection
        residual_out = self.residual_conv(gated) + residual  # [B, C, T]

        # Layer norm (applied over channel dim)
        residual_out = self.layer_norm(residual_out.transpose(1, 2)).transpose(1, 2)

        return residual_out, skip


class MultiScaleBlock(nn.Module):
    """
    Multi-scale temporal feature extraction.

    Runs dilated convolutions at multiple scales in parallel and concatenates
    their outputs. This is particularly useful for CGM because:
      - Scale 1 (dilation 1): captures rapid CGM excursions (5-15 min)
      - Scale 2 (dilation 4): captures postprandial peaks (20-40 min)
      - Scale 3 (dilation 12): captures insulin action curves (1 hr)
      - Scale 4 (dilation 36): captures overnight trends (3 hrs)
    """

    def __init__(self, n_channels: int, scales: List[int], kernel_size: int = 3):
        super().__init__()
        n_out = n_channels // len(scales)  # distribute channels across scales
        self.convs = nn.ModuleList([
            CausalConv1d(n_channels, n_out, kernel_size, dilation=d)
            for d in scales
        ])
        self.proj = nn.Conv1d(n_out * len(scales), n_channels, 1)
        self.norm = nn.LayerNorm(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_outs = [conv(x) for conv in self.convs]
        multi_scale = torch.cat(scale_outs, dim=1)   # [B, n_channels, T]
        out = self.proj(multi_scale)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out + x   # Residual


class TCNGlucose(nn.Module):
    """
    Temporal Convolutional Network for multi-horizon CGM glucose prediction.

    Architecture:
        Input Embedding
            └──► Stack of Dilated Residual Blocks
                  (exponentially increasing dilation: 1,2,4,8,16,32,64,128)
                └──► Skip connections summed
                     └──► Multi-Scale Block (multi-resolution fusion)
                          └──► Output projection for each horizon

    Args:
        input_size:          Number of input features per timestep.
        n_channels:          Internal channel width (default 64).
        kernel_size:         Convolution kernel size (default 3).
        num_blocks:          Number of dilated blocks (dilation doubles each block).
        dropout:             Dropout rate.
        encoder_steps:       Historical lookback window.
        prediction_horizons: Steps to forecast (e.g. [6, 12, 24]).
        multi_scale_dilations: Dilation rates for the multi-scale fusion block.
    """

    def __init__(
        self,
        input_size: int = 12,
        n_channels: int = 64,
        kernel_size: int = 3,
        num_blocks: int = 8,
        dropout: float = 0.2,
        encoder_steps: int = 72,
        prediction_horizons: List[int] = None,
        multi_scale_dilations: Optional[List[int]] = None,
    ):
        super().__init__()
        self.encoder_steps = encoder_steps
        self.prediction_horizons = prediction_horizons or [6, 12, 24]
        self.max_horizon = max(self.prediction_horizons)
        self.n_channels = n_channels

        # Receptive field logging (informational)
        dilations = [2 ** i for i in range(num_blocks)]
        self.receptive_field = 1 + sum(2 * (kernel_size - 1) * d for d in dilations)

        # Input embedding
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_size, n_channels, 1),
            nn.LayerNorm(n_channels),   # LayerNorm needs channel last, applied differently
        )
        self.input_norm = nn.LayerNorm(n_channels)

        # Stack of dilated residual blocks
        self.residual_blocks = nn.ModuleList([
            DilatedResidualBlock(n_channels, kernel_size, dilation=dilations[i], dropout=dropout)
            for i in range(num_blocks)
        ])

        # Multi-scale fusion
        ms_dilations = multi_scale_dilations or [1, 4, 12, 36]
        self.multi_scale = MultiScaleBlock(n_channels, ms_dilations, kernel_size)

        # Output MLP for each horizon
        self.output_head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_channels, n_channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(n_channels // 2, len(self.prediction_horizons), 1),
        )

        # Horizon-specific linear calibration layers
        self.horizon_calibrators = nn.ModuleList([
            nn.Linear(encoder_steps + self.max_horizon, 1)
            for _ in self.prediction_horizons
        ])

        # Future feature conditioning
        self.future_input_size = 6
        self.future_cond = nn.Linear(self.future_input_size, n_channels)

    def forward(
        self,
        historical: torch.Tensor,             # [B, T_enc, input_size]
        future: Optional[torch.Tensor] = None, # [B, T_fut, future_input_size]
    ) -> dict:
        """
        Args:
            historical: [B, encoder_steps, input_size]
            future:     [B, max_horizon, future_input_size] — known future features

        Returns:
            dict with "predictions" [B, n_horizons, 1]
        """
        B, T_enc, _ = historical.shape

        # Optionally concatenate future conditioning
        if future is not None:
            future_cond = self.future_cond(future)   # [B, T_fut, n_channels]
            # Convert to channels-first for conv
            x = historical.transpose(1, 2)   # [B, input_size, T_enc]
            x = self.input_conv(x)            # [B, n_channels, T_enc]
            x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
            # Append conditioned future (channels-first)
            fc = future_cond.transpose(1, 2)  # [B, n_channels, T_fut]
            x = torch.cat([x, fc], dim=-1)    # [B, n_channels, T_enc + T_fut]
        else:
            x = historical.transpose(1, 2)    # [B, input_size, T_enc]
            x = self.input_conv(x)
            x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)

        # Run through dilated blocks, accumulate skip connections
        skip_sum = torch.zeros_like(x)
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip

        # Multi-scale fusion on skip sum
        fused = self.multi_scale(skip_sum)

        # Output projection → [B, n_horizons, T_total]
        raw_output = self.output_head(fused)  # [B, n_horizons, T_total]

        # Per-horizon calibrator: pool over time to scalar prediction
        T_total = raw_output.shape[-1]
        horizon_preds = []
        for i, _ in enumerate(self.prediction_horizons):
            h_seq = raw_output[:, i, :]    # [B, T_total]
            # Pad or truncate to expected length for calibrator
            expected_len = self.encoder_steps + self.max_horizon
            if T_total < expected_len:
                h_seq = F.pad(h_seq, (0, expected_len - T_total))
            elif T_total > expected_len:
                h_seq = h_seq[:, :expected_len]
            pred = self.horizon_calibrators[i](h_seq)  # [B, 1]
            horizon_preds.append(pred)

        predictions = torch.stack(horizon_preds, dim=1)  # [B, n_horizons, 1]

        return {
            "predictions": predictions,
            "receptive_field_steps": self.receptive_field,
            "receptive_field_minutes": self.receptive_field * 5,
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"TCNGlucose("
            f"channels={self.n_channels}, "
            f"blocks={len(self.residual_blocks)}, "
            f"RF={self.receptive_field} steps / {self.receptive_field * 5} min, "
            f"params={self.num_parameters:,})"
        )
