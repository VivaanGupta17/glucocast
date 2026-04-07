"""
Temporal Fusion Transformer (TFT) for multi-horizon glucose prediction.

Implements the architecture from:
    Lim et al. (2021). Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting. IJF 37(4):1748-1764.
    https://arxiv.org/abs/1912.09363

Adapted for CGM glucose prediction with:
  - Static covariates: patient demographics (age, weight, TDD, HbA1c)
  - Historical dynamic inputs: CGM values, IOB, COB, exercise intensity
  - Known future inputs: planned meals (carb estimates), scheduled boluses,
    time-of-day features, day-of-week
  - Quantile regression: 10th, 50th, 90th percentiles for prediction intervals
  - Multi-horizon: simultaneous 30/60/120-min outputs
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class GatedLinearUnit(nn.Module):
    """GLU activation: splits input in half and gates second half through sigmoid."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class GatedResidualNetwork(nn.Module):
    """
    GRN as described in TFT paper (Fig. 2).

    Optionally accepts an external context vector (e.g. static covariate
    embedding) to condition the transformation.

    Args:
        input_size:   Input feature dimension.
        hidden_size:  Internal hidden dimension.
        output_size:  Output dimension (defaults to input_size).
        dropout:      Dropout rate.
        context_size: Dimension of optional context vector.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        output_size = output_size or input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.glu = GatedLinearUnit(hidden_size, output_size, dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        # Residual projection if dimensions differ
        self.residual_proj = (
            nn.Linear(input_size, output_size, bias=False) if input_size != output_size else None
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = self.residual_proj(x) if self.residual_proj else x

        hidden = self.fc1(x)
        if context is not None and self.context_fc is not None:
            # Broadcast context over sequence dimension if needed
            if context.dim() == 2 and hidden.dim() == 3:
                context = context.unsqueeze(1)
            hidden = hidden + self.context_fc(context)
        hidden = F.elu(hidden)
        hidden = self.fc2(hidden)

        output = self.glu(hidden)
        return self.layer_norm(output + residual)


class VariableSelectionNetwork(nn.Module):
    """
    VSN learns soft feature selection weights for a set of input variables.

    Each variable is first projected to a common hidden_size, then a GRN
    produces a softmax distribution over variables. This provides interpretable
    feature importance scores at inference time.

    Args:
        num_inputs:  Number of input variables (each of size input_size).
        input_size:  Dimension of each variable after initial embedding.
        hidden_size: GRN hidden dimension.
        dropout:     Dropout rate.
        context_size: Optional static context conditioning dimension.
    """

    def __init__(
        self,
        num_inputs: int,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size

        # Per-variable GRNs
        self.variable_grns = nn.ModuleList(
            [GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout) for _ in range(num_inputs)]
        )

        # Combined softmax GRN
        self.flattened_grn = GatedResidualNetwork(
            num_inputs * input_size,
            hidden_size,
            num_inputs,
            dropout,
            context_size=context_size,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:       List of [batch, seq_len, input_size] or [batch, input_size] tensors.
            context: Optional [batch, context_size] static embedding.

        Returns:
            processed: [batch, seq_len, hidden_size] or [batch, hidden_size]
            weights:   [batch, seq_len, num_inputs] — variable selection weights
        """
        # Stack and process each variable through its GRN
        var_outputs = [grn(xi) for grn, xi in zip(self.variable_grns, x)]
        stacked = torch.stack(var_outputs, dim=-2)  # [..., num_inputs, hidden_size]

        # Flatten for selection GRN
        flat_inputs = torch.cat(x, dim=-1)  # [..., num_inputs * input_size]
        weights = self.flattened_grn(flat_inputs, context)  # [..., num_inputs]
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)  # [..., num_inputs, 1]

        # Weighted sum over variables
        processed = (weights * stacked).sum(dim=-2)  # [..., hidden_size]
        return processed, weights.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with shared value projection across heads.

    The shared value projection forces the model to explain its behavior
    through head-specific attention patterns rather than value transformations,
    improving interpretability (TFT paper, Section 4.3.1).
    """

    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)  # Shared across heads
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = query.shape
        H = self.num_heads
        Dh = self.d_head

        q = self.q_proj(query).view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k_proj(key).view(B, -1, H, Dh).transpose(1, 2)
        v = self.v_proj(value).view(B, -1, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # [B, H, T, Dh]
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        output = self.out_proj(context)
        output = self.layer_norm(output + query)  # Residual

        # Average attention across heads for interpretability
        attn_avg = attn_weights.mean(dim=1)  # [B, T, T_k]
        return output, attn_avg


# ---------------------------------------------------------------------------
# Main TFT Model
# ---------------------------------------------------------------------------

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon CGM glucose prediction.

    Input variables (configure in ohio_config.yaml):

    Static (patient-level, constant across time):
        - patient_age        (years)
        - patient_weight     (kg)
        - total_daily_dose   (units/day, 90-day average)
        - hba1c              (%, most recent)
        - diabetes_duration  (years)

    Historical dynamic (observed up to current time t):
        - cgm                (mg/dL, 5-min intervals)
        - iob                (units, insulin on board)
        - cob                (g, carbs on board)
        - basal_rate         (units/hr)
        - exercise_intensity (0-3 scale)
        - cgm_roc            (mg/dL/min, rate of change)

    Known future (available at time t for t+1..t+H):
        - announced_meal     (g carbs, 0 if none)
        - planned_bolus      (units, 0 if none)
        - time_sin / time_cos (circadian encoding)
        - day_of_week_sin/cos (weekly rhythm encoding)

    Args:
        num_static_vars:     Number of static covariate variables.
        num_historical_vars: Number of historical dynamic variables.
        num_future_vars:     Number of known future variables.
        hidden_size:         Core model dimension.
        num_heads:           Multi-head attention heads.
        num_lstm_layers:     LSTM encoder/decoder depth.
        dropout:             Dropout rate.
        encoder_steps:       Lookback window length (e.g. 72 = 6 hours at 5-min).
        prediction_horizons: List of prediction steps (e.g. [6, 12, 24] for 30/60/120 min).
        quantiles:           Quantile levels for output (e.g. [0.1, 0.5, 0.9]).
    """

    def __init__(
        self,
        num_static_vars: int = 5,
        num_historical_vars: int = 6,
        num_future_vars: int = 6,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        encoder_steps: int = 72,
        prediction_horizons: List[int] = None,
        quantiles: List[float] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_steps = encoder_steps
        self.prediction_horizons = prediction_horizons or [6, 12, 24]
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.num_quantiles = len(self.quantiles)

        # ------------------------------------------------------------------
        # Input Embeddings
        # Each continuous variable is projected to hidden_size via a linear.
        # Categorical variables would use nn.Embedding (not shown here).
        # ------------------------------------------------------------------
        self.static_var_embeddings = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(num_static_vars)]
        )
        self.historical_var_embeddings = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(num_historical_vars)]
        )
        self.future_var_embeddings = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(num_future_vars)]
        )

        # ------------------------------------------------------------------
        # Static Covariate Encoders
        # ------------------------------------------------------------------
        self.static_vsn = VariableSelectionNetwork(
            num_inputs=num_static_vars,
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        # Static covariate context vectors (4 distinct uses in TFT)
        self.static_context_enrichment = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.static_context_state_h = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.static_context_state_c = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.static_context_selection = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)

        # ------------------------------------------------------------------
        # Variable Selection Networks for temporal inputs
        # ------------------------------------------------------------------
        self.historical_vsn = VariableSelectionNetwork(
            num_inputs=num_historical_vars,
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,
        )
        self.future_vsn = VariableSelectionNetwork(
            num_inputs=num_future_vars,
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,
        )

        # ------------------------------------------------------------------
        # Sequence-to-Sequence LSTM
        # ------------------------------------------------------------------
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Gate on LSTM outputs
        self.post_seq_glu = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.post_seq_norm = nn.LayerNorm(hidden_size)

        # ------------------------------------------------------------------
        # Static Enrichment Layer
        # ------------------------------------------------------------------
        self.static_enrichment_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,
        )

        # ------------------------------------------------------------------
        # Temporal Self-Attention
        # ------------------------------------------------------------------
        self.positional_encoding = PositionalEncoding(hidden_size, dropout=dropout)
        self.self_attention = InterpretableMultiHeadAttention(num_heads, hidden_size, dropout)
        self.post_attn_glu = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        # ------------------------------------------------------------------
        # Position-wise Feed-Forward (GRN)
        # ------------------------------------------------------------------
        self.ff_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.pre_output_glu = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.pre_output_norm = nn.LayerNorm(hidden_size)

        # ------------------------------------------------------------------
        # Output projection: one head per quantile per horizon
        # ------------------------------------------------------------------
        max_horizon = max(self.prediction_horizons)
        self.output_projection = nn.Linear(
            hidden_size, self.num_quantiles * len(self.prediction_horizons)
        )

    def _embed_variables(
        self, x: torch.Tensor, embeddings: nn.ModuleList
    ) -> List[torch.Tensor]:
        """Project each variable (last dim) through its embedding layer."""
        n_vars = x.shape[-1]
        assert n_vars == len(embeddings), f"Expected {len(embeddings)} vars, got {n_vars}"
        return [embeddings[i](x[..., i : i + 1]) for i in range(n_vars)]

    def forward(
        self,
        static: torch.Tensor,           # [B, num_static_vars]
        historical: torch.Tensor,        # [B, encoder_steps, num_historical_vars]
        future: torch.Tensor,            # [B, max_horizon, num_future_vars]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full TFT.

        Returns:
            Dict with keys:
              - "predictions": [B, len(horizons), num_quantiles]  glucose forecasts
              - "historical_weights": [B, encoder_steps, num_historical_vars]
              - "future_weights":     [B, max_horizon, num_future_vars]
              - "static_weights":     [B, num_static_vars]
              - "attention_weights":  [B, T_total, T_total]
        """
        B = static.shape[0]

        # ------------------------------------------------------------------
        # 1. Static covariate processing
        # ------------------------------------------------------------------
        static_embs = self._embed_variables(static, self.static_var_embeddings)
        static_ctx, static_weights = self.static_vsn(static_embs)

        cs = self.static_context_selection(static_ctx)      # context for VSN
        ce = self.static_context_enrichment(static_ctx)     # context for enrichment
        ch = self.static_context_state_h(static_ctx)        # LSTM h init
        cc = self.static_context_state_c(static_ctx)        # LSTM c init

        # ------------------------------------------------------------------
        # 2. Historical variable selection
        # ------------------------------------------------------------------
        hist_embs = self._embed_variables(historical, self.historical_var_embeddings)
        # hist_embs: list of [B, enc_steps, hidden]
        hist_selected, hist_weights = self.historical_vsn(hist_embs, context=cs)

        # ------------------------------------------------------------------
        # 3. Future variable selection
        # ------------------------------------------------------------------
        fut_embs = self._embed_variables(future, self.future_var_embeddings)
        fut_selected, fut_weights = self.future_vsn(fut_embs, context=cs)

        # ------------------------------------------------------------------
        # 4. Sequence-to-sequence LSTM
        # ------------------------------------------------------------------
        # Initialise LSTM state from static context
        num_layers = self.encoder_lstm.num_layers
        h0 = ch.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
        c0 = cc.unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        enc_out, (hn, cn) = self.encoder_lstm(hist_selected, (h0, c0))
        dec_out, _ = self.decoder_lstm(fut_selected, (hn, cn))

        # Concatenate encoder + decoder outputs over time
        lstm_out = torch.cat([enc_out, dec_out], dim=1)  # [B, enc+dec, hidden]
        enc_len = enc_out.shape[1]

        # Gate LSTM outputs (skip-connection from VSN outputs)
        vsn_out = torch.cat([hist_selected, fut_selected], dim=1)
        lstm_gated = self.post_seq_glu(lstm_out)
        lstm_out = self.post_seq_norm(lstm_gated + vsn_out)

        # ------------------------------------------------------------------
        # 5. Static enrichment
        # ------------------------------------------------------------------
        enriched = self.static_enrichment_grn(lstm_out, context=ce)

        # ------------------------------------------------------------------
        # 6. Temporal self-attention (decoder-only mask for causal prediction)
        # ------------------------------------------------------------------
        enriched_pe = self.positional_encoding(enriched)
        T_total = enriched_pe.shape[1]

        # Causal mask: future positions cannot attend to later future positions,
        # but all positions can attend to encoder outputs.
        causal_mask = torch.ones(T_total, T_total, device=static.device)
        future_start = enc_len
        causal_mask[future_start:, future_start:] = torch.tril(
            torch.ones(T_total - future_start, T_total - future_start, device=static.device)
        )

        attn_out, attn_weights = self.self_attention(
            enriched_pe, enriched_pe, enriched_pe, mask=causal_mask.unsqueeze(0).unsqueeze(0)
        )
        attn_gated = self.post_attn_glu(attn_out)
        attn_out = self.post_attn_norm(attn_gated + enriched)

        # ------------------------------------------------------------------
        # 7. Position-wise feed-forward (GRN)
        # ------------------------------------------------------------------
        ff_out = self.ff_grn(attn_out)
        ff_gated = self.pre_output_glu(ff_out)
        ff_out = self.pre_output_norm(ff_gated + attn_out)

        # ------------------------------------------------------------------
        # 8. Output projection (decoder steps only → prediction horizons)
        # ------------------------------------------------------------------
        decoder_out = ff_out[:, enc_len:, :]  # [B, max_horizon, hidden]

        # Select timesteps corresponding to requested horizons
        horizon_outputs = []
        for h_step in self.prediction_horizons:
            step_out = decoder_out[:, h_step - 1, :]   # [B, hidden]
            horizon_outputs.append(step_out)
        horizon_tensor = torch.stack(horizon_outputs, dim=1)  # [B, n_horizons, hidden]

        # Project to quantiles
        raw = self.output_projection(horizon_tensor)  # [B, n_horizons, n_quantiles]
        # Reshape
        B, H, _ = raw.shape
        predictions = raw.view(B, H, self.num_quantiles)

        # Enforce quantile monotonicity (lower ≤ median ≤ upper)
        predictions = torch.cummax(predictions, dim=-1).values

        return {
            "predictions": predictions,          # [B, n_horizons, n_quantiles]
            "historical_weights": hist_weights,  # [B, enc_steps, n_hist_vars]
            "future_weights": fut_weights,       # [B, max_horizon, n_fut_vars]
            "static_weights": static_weights,    # [B, n_static_vars]
            "attention_weights": attn_weights,   # [B, T_total, T_total]
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quantile Loss
# ---------------------------------------------------------------------------

class QuantileLoss(nn.Module):
    """
    Pinball (quantile) loss summed over all quantile levels and horizons.

    For quantile q and residual r = y_true - y_pred:
        L_q(r) = q * r       if r >= 0
                 (q-1) * r   if r < 0
    """

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(
        self,
        predictions: torch.Tensor,  # [B, n_horizons, n_quantiles]
        targets: torch.Tensor,       # [B, n_horizons]
    ) -> torch.Tensor:
        targets = targets.unsqueeze(-1)  # [B, n_horizons, 1]
        errors = targets - predictions   # [B, n_horizons, n_quantiles]
        q = self.quantiles.view(1, 1, -1)
        loss = torch.max((q - 1) * errors, q * errors)
        return loss.mean()
