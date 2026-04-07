"""
Encoder-decoder LSTM with attention for multi-horizon glucose prediction.

Architecture:
  - Bidirectional encoder LSTM processes the input window
  - Attention mechanism produces context vectors for each decoder step
  - Decoder LSTM with teacher forcing during training
  - Separate output heads per prediction horizon

References:
  - Luong et al. (2015). Effective Approaches to Attention-based Neural
    Machine Translation. https://arxiv.org/abs/1508.04025
  - Cho et al. (2014). Learning Phrase Representations using RNN
    Encoder-Decoder. https://arxiv.org/abs/1406.1078
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention for sequence-to-sequence LSTM.

    Computes alignment scores between decoder hidden state and all
    encoder outputs, then produces a weighted context vector.

    CGM motivation: The encoder window spans 6-12 hours of CGM readings.
    Attention allows the decoder to focus on physiologically relevant
    periods — e.g., recent meal absorption, insulin activity peaks.
    """

    def __init__(self, encoder_hidden: int, decoder_hidden: int, attention_dim: int = 64):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_hidden, attention_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_hidden, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,    # [B, T, encoder_hidden]
        decoder_hidden: torch.Tensor,     # [B, decoder_hidden]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context:  [B, encoder_hidden] weighted sum of encoder states
            weights:  [B, T] attention distribution (interpretable!)
        """
        # Project encoder outputs and decoder hidden state
        enc_proj = self.encoder_proj(encoder_outputs)                    # [B, T, att_dim]
        dec_proj = self.decoder_proj(decoder_hidden).unsqueeze(1)       # [B, 1, att_dim]
        energy = torch.tanh(enc_proj + dec_proj)                         # [B, T, att_dim]
        scores = self.v(energy).squeeze(-1)                              # [B, T]

        weights = F.softmax(scores, dim=-1)                              # [B, T]
        context = (weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)  # [B, encoder_hidden]
        return context, weights


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder.

    Processes the historical CGM window and outputs:
      - encoder_outputs: All hidden states for attention
      - (h_n, c_n): Final state to initialise decoder

    Using bidirectional LSTM captures both the recent trend (forward pass)
    and longer-range context such as meal absorption patterns (backward pass).
    The backward pass is truncated for causal deployment — it sees the full
    window but never future data.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Project bidirectional state back to decoder hidden size
        if bidirectional:
            self.h_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.c_proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, input_size] historical features

        Returns:
            encoder_outputs: [B, T, hidden_size * num_directions]
            (h, c):          [num_layers, B, hidden_size] decoder init state
        """
        enc_out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            # Merge forward/backward final states for each layer
            h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
            c_n = c_n.view(self.num_layers, 2, -1, self.hidden_size)
            h_merged = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)   # [n_layers, B, 2*H]
            c_merged = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)
            h_n = torch.tanh(self.h_proj(h_merged))   # [n_layers, B, H]
            c_n = torch.tanh(self.c_proj(c_merged))

        return enc_out, (h_n, c_n)


class LSTMDecoder(nn.Module):
    """
    Autoregressive LSTM decoder with Bahdanau attention.

    At each decoding step:
        1. Concatenate previous output (or target via teacher forcing) with context
        2. LSTM step
        3. Attend over encoder outputs → new context
        4. Project to glucose prediction

    Teacher forcing: During training, feed ground-truth previous CGM value
    as the decoder input with probability `teacher_forcing_ratio`. This
    stabilises training but requires scheduled decay during fine-tuning.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        encoder_hidden_size: int,
        output_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2,
        attention_dim: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input: [prev_glucose, context_vector]
        encoder_output_size = encoder_hidden_size * 2   # bidirectional encoder
        self.lstm_input_size = input_size + encoder_output_size

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = BahdanauAttention(
            encoder_hidden=encoder_output_size,
            decoder_hidden=hidden_size,
            attention_dim=attention_dim,
        )

        # Output projection: concatenate LSTM output + context → glucose
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size + encoder_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        x: torch.Tensor,                  # [B, input_size]
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,     # [B, T, encoder_out]
    ) -> Tuple[torch.Tensor, Tuple, torch.Tensor]:
        """Single decoder step. Returns (output, new_hidden, attention_weights)."""
        h_n = hidden[0][-1]                              # Last layer hidden state [B, H]
        context, attn_w = self.attention(encoder_outputs, h_n)

        lstm_in = torch.cat([x, context], dim=-1).unsqueeze(1)  # [B, 1, lstm_input]
        lstm_out, new_hidden = self.lstm(lstm_in, hidden)
        lstm_out = lstm_out.squeeze(1)                           # [B, hidden]

        combined = torch.cat([lstm_out, context], dim=-1)
        output = self.output_proj(combined)   # [B, output_size]
        return output, new_hidden, attn_w

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_state: Tuple[torch.Tensor, torch.Tensor],
        target_sequence: Optional[torch.Tensor] = None,   # [B, max_steps, 1] for teacher forcing
        prediction_steps: int = 24,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode for `prediction_steps` steps.

        Args:
            encoder_outputs:      [B, T, encoder_out]
            encoder_state:        Initial (h, c) from encoder
            target_sequence:      Ground-truth CGM for teacher forcing (training only)
            prediction_steps:     Max horizon steps to decode
            teacher_forcing_ratio: Fraction of steps using ground-truth input

        Returns:
            outputs:          [B, prediction_steps, 1]
            attention_matrix: [B, prediction_steps, T]
        """
        B = encoder_outputs.shape[0]
        hidden = encoder_state

        # Start with last known CGM value (zero-initialised, set by caller)
        x = torch.zeros(B, encoder_outputs.shape[-1] - encoder_outputs.shape[-1] + 1,
                        device=encoder_outputs.device)
        # Actually use a learned start token
        x = encoder_outputs[:, -1, :1]  # Last CGM value proxy from encoder

        outputs = []
        attn_weights = []
        use_teacher_forcing = (target_sequence is not None) and (teacher_forcing_ratio > 0)

        for step in range(prediction_steps):
            output, hidden, attn_w = self.forward_step(x, hidden, encoder_outputs)
            outputs.append(output)
            attn_weights.append(attn_w)

            if use_teacher_forcing and torch.rand(1).item() < teacher_forcing_ratio:
                x = target_sequence[:, step, :]   # Teacher: use ground truth
            else:
                x = output                         # Student: use own prediction

        outputs = torch.stack(outputs, dim=1)           # [B, steps, 1]
        attn_weights = torch.stack(attn_weights, dim=1)  # [B, steps, T]
        return outputs, attn_weights


class GlucoseLSTM(nn.Module):
    """
    Complete encoder-decoder LSTM for multi-horizon CGM glucose prediction.

    Feature input layout (historical_features):
        [cgm, iob, cob, basal_rate, exercise_intensity, cgm_roc,
         meal_flag, bolus_flag, time_sin, time_cos, dow_sin, dow_cos]

    Args:
        input_size:           Number of input features per timestep.
        hidden_size:          LSTM hidden dimension.
        encoder_steps:        Historical lookback window length (timesteps).
        prediction_horizons:  List of horizon steps [6, 12, 24] for 30/60/120 min.
        num_encoder_layers:   Encoder LSTM depth.
        num_decoder_layers:   Decoder LSTM depth.
        dropout:              Dropout rate.
        teacher_forcing_ratio: Teacher forcing probability (training).
    """

    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        encoder_steps: int = 72,
        prediction_horizons: List[int] = None,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.2,
        teacher_forcing_ratio: float = 0.5,
    ):
        super().__init__()
        self.encoder_steps = encoder_steps
        self.prediction_horizons = prediction_horizons or [6, 12, 24]
        self.max_horizon = max(self.prediction_horizons)
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Input projection: normalize & project features
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.encoder = LSTMEncoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_encoder_layers,
            dropout=dropout,
            bidirectional=True,
        )

        # Future feature embedding for decoder input
        self.future_input_size = 6   # time_sin, time_cos, dow_sin, dow_cos, meal, bolus
        self.future_proj = nn.Linear(self.future_input_size, 1)

        self.decoder = LSTMDecoder(
            input_size=1,
            hidden_size=hidden_size,
            encoder_hidden_size=hidden_size,
            num_layers=num_decoder_layers,
            dropout=dropout,
        )

        # Per-horizon output heads (finer-grained learning)
        self.horizon_heads = nn.ModuleList([
            nn.Linear(1, 1) for _ in self.prediction_horizons
        ])

    def forward(
        self,
        historical: torch.Tensor,             # [B, encoder_steps, input_size]
        future: Optional[torch.Tensor] = None, # [B, max_horizon, future_input_size]
        target_sequence: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            historical:      Historical CGM + covariates
            future:          Known future features (meals, planned boluses, time encoding)
            target_sequence: Ground-truth future CGM (for teacher forcing, training only)

        Returns:
            dict with "predictions" [B, n_horizons, 1] and "attention_weights"
        """
        B = historical.shape[0]

        # Encode historical sequence
        hist_proj = self.input_proj(historical)
        encoder_outputs, encoder_state = self.encoder(hist_proj)

        # Decode
        outputs, attn_weights = self.decoder(
            encoder_outputs=encoder_outputs,
            encoder_state=encoder_state,
            target_sequence=target_sequence,
            prediction_steps=self.max_horizon,
            teacher_forcing_ratio=self.teacher_forcing_ratio if self.training else 0.0,
        )

        # Select outputs at requested horizons and apply fine-tuning head
        horizon_preds = []
        for i, h_step in enumerate(self.prediction_horizons):
            pred = outputs[:, h_step - 1, :]     # [B, 1]
            pred = self.horizon_heads[i](pred)
            horizon_preds.append(pred)

        predictions = torch.stack(horizon_preds, dim=1)  # [B, n_horizons, 1]

        return {
            "predictions": predictions,
            "attention_weights": attn_weights,   # [B, max_horizon, encoder_steps]
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
