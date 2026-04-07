"""
N-BEATS: Neural Basis Expansion Analysis for interpretable glucose forecasting.

Implements the N-BEATS architecture from:
    Oreshkin et al. (2019). N-BEATS: Neural basis expansion analysis for
    interpretable time series forecasting. ICLR 2020.
    https://arxiv.org/abs/1905.10437

N-BEATS advantages for CGM:
  - Interpretable: explicit trend + seasonality decomposition mirrors the
    physiological components of the glucose signal
  - No inductive bias from recurrence or convolution — learns directly from data
  - Doubly residual: both backcast and forecast residuals propagate through stacks
  - Trend block: captures slow insulin action / basal drift
  - Seasonality block: captures circadian rhythm (~24h) and postprandial pattern

Glucose signal decomposition:
  - Trend component: basal glucose drift, HbA1c level, dawn phenomenon
  - Seasonality component: diurnal variation, dawn effect (cortisol ~4-8am),
    post-lunch variability
  - Residual component: meal spikes, bolus responses, exercise effects

Modifications for CGM:
  - Generic stacks added alongside trend/seasonality for meal/insulin residuals
  - Exogenous covariates (IOB, COB) injected as additional block inputs
  - Multi-horizon output: all three horizons from a single forward pass
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basis Functions
# ---------------------------------------------------------------------------

def trend_basis(degree: int, steps: int, device: torch.device) -> torch.Tensor:
    """
    Polynomial trend basis matrix.

    Generates columns [t^0, t^1, t^2, ..., t^degree] for t in [0, 1].
    A degree-3 polynomial can represent constant offset, linear drift,
    quadratic curvature, and cubic inflection.

    Args:
        degree: Polynomial degree (inclusive). Degree=3 recommended for CGM.
        steps:  Number of time steps (lookback + horizon).
        device: Target device.

    Returns:
        [steps, degree+1] basis matrix
    """
    t = torch.linspace(0, 1, steps, device=device).unsqueeze(-1)  # [steps, 1]
    powers = torch.arange(degree + 1, device=device).float()      # [degree+1]
    return t ** powers   # [steps, degree+1]


def seasonality_basis(n_harmonics: int, steps: int, device: torch.device) -> torch.Tensor:
    """
    Fourier seasonality basis.

    Generates sin/cos pairs for harmonics 1..n_harmonics.
    For CGM with 5-min resolution:
      - Harmonic 1: 24h circadian (288 steps/cycle)
      - Harmonic 2: 12h sub-circadian (cortisol peaks)
      - Harmonic 3: 8h meal pattern
      - Higher harmonics: postprandial oscillations

    Args:
        n_harmonics: Number of Fourier harmonics (each produces sin + cos).
        steps:       Number of time steps.
        device:      Target device.

    Returns:
        [steps, 2 * n_harmonics] basis matrix
    """
    t = torch.linspace(0, 1, steps, device=device)
    freqs = torch.arange(1, n_harmonics + 1, device=device).float()
    phases = 2 * math.pi * freqs.unsqueeze(0) * t.unsqueeze(-1)  # [steps, n_harmonics]
    basis = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # [steps, 2*H]
    return basis


# ---------------------------------------------------------------------------
# N-BEATS Block
# ---------------------------------------------------------------------------

class NBeatsBlock(nn.Module):
    """
    Single N-BEATS block with shared FC stack + basis expansion.

    Each block:
        1. Fully connected stack: input → hidden representations
        2. Basis coefficient projection: θ_backcast, θ_forecast
        3. Basis expansion: multiply coefficients by basis vectors

    For generic blocks, the basis is a learned linear projection.
    For trend/seasonality blocks, the basis is fixed (polynomial/Fourier).

    Args:
        input_size:      Lookback window length (number of input timesteps).
        theta_size:      Dimension of basis coefficient vector.
        horizon:         Number of forecast timesteps.
        hidden_size:     FC hidden width.
        n_layers:        Number of FC layers in the stack.
        block_type:      'generic' | 'trend' | 'seasonality'
        trend_degree:    Polynomial degree for trend block.
        n_harmonics:     Fourier harmonics for seasonality block.
        covariate_size:  Optional extra covariates injected at block input.
        dropout:         Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        horizon: int,
        hidden_size: int = 256,
        n_layers: int = 4,
        block_type: str = "generic",
        trend_degree: int = 3,
        n_harmonics: int = 6,
        covariate_size: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_type = block_type
        self.input_size = input_size
        self.horizon = horizon
        self.trend_degree = trend_degree
        self.n_harmonics = n_harmonics

        # Fully connected stack
        fc_input = input_size + covariate_size
        layers = []
        for i in range(n_layers):
            in_d = fc_input if i == 0 else hidden_size
            layers.extend([nn.Linear(in_d, hidden_size), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.fc_stack = nn.Sequential(*layers)

        # Theta (basis coefficient) projection
        if block_type == "trend":
            theta_size = trend_degree + 1
        elif block_type == "seasonality":
            theta_size = 2 * n_harmonics

        self.theta_backcast = nn.Linear(hidden_size, theta_size, bias=False)
        self.theta_forecast = nn.Linear(hidden_size, theta_size, bias=False)

        # Generic blocks: basis is a learned linear projection
        if block_type == "generic":
            self.backcast_basis = nn.Linear(theta_size, input_size, bias=False)
            self.forecast_basis = nn.Linear(theta_size, horizon, bias=False)

    def forward(
        self,
        x: torch.Tensor,                          # [B, input_size]
        covariates: Optional[torch.Tensor] = None, # [B, covariate_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            backcast: [B, input_size] — what this block explains of the input
            forecast: [B, horizon]    — this block's forecast contribution
        """
        if covariates is not None:
            inp = torch.cat([x, covariates], dim=-1)
        else:
            inp = x

        hidden = self.fc_stack(inp)
        theta_b = self.theta_backcast(hidden)
        theta_f = self.theta_forecast(hidden)

        if self.block_type == "generic":
            backcast = self.backcast_basis(theta_b)
            forecast = self.forecast_basis(theta_f)
        elif self.block_type == "trend":
            B_mat = trend_basis(self.trend_degree, self.input_size, x.device)   # [input_size, deg+1]
            F_mat = trend_basis(self.trend_degree, self.horizon, x.device)       # [horizon, deg+1]
            backcast = torch.matmul(theta_b, B_mat.T)  # [B, input_size]
            forecast = torch.matmul(theta_f, F_mat.T)  # [B, horizon]
        elif self.block_type == "seasonality":
            B_mat = seasonality_basis(self.n_harmonics, self.input_size, x.device)  # [input_size, 2H]
            F_mat = seasonality_basis(self.n_harmonics, self.horizon, x.device)      # [horizon, 2H]
            backcast = torch.matmul(theta_b, B_mat.T)
            forecast = torch.matmul(theta_f, F_mat.T)
        else:
            raise ValueError(f"Unknown block type: {self.block_type}")

        return backcast, forecast


# ---------------------------------------------------------------------------
# N-BEATS Stack
# ---------------------------------------------------------------------------

class NBeatsStack(nn.Module):
    """
    Stack of N-BEATS blocks of the same type.

    Within a stack, blocks share the same basis type and are connected by
    doubly residual links:
      - Input to each block = previous block's backcast residual
      - Stack forecast = sum of all block forecasts
    """

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(
        self,
        x: torch.Tensor,                          # [B, input_size]
        covariates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            residual_backcast: [B, input_size] — remaining unexplained signal
            stack_forecast:    [B, horizon]    — stack's total forecast contribution
        """
        stack_forecast = 0.0
        backcast = x
        for block in self.blocks:
            block_backcast, block_forecast = block(backcast, covariates)
            backcast = backcast - block_backcast   # Residual: pass unexplained part
            stack_forecast = stack_forecast + block_forecast
        return backcast, stack_forecast


# ---------------------------------------------------------------------------
# Main N-BEATS Model
# ---------------------------------------------------------------------------

class NBeatsGlucose(nn.Module):
    """
    N-BEATS for multi-horizon CGM glucose prediction.

    Three-stack architecture:
        Stack 1 (Trend):       Captures slow glucose drift, dawn phenomenon
        Stack 2 (Seasonality): Captures circadian patterns, meal timing rhythms
        Stack 3 (Generic):     Captures meal spikes, bolus responses, residuals

    For multi-horizon output, separate models are trained for each horizon,
    OR a shared backbone with horizon-specific output heads is used (here).

    Args:
        input_size:          Lookback window size (timesteps).
        prediction_horizons: Forecast steps [6, 12, 24] for 30/60/120 min.
        hidden_size:         FC hidden width per block.
        n_blocks_per_stack:  Number of blocks in each stack.
        n_fc_layers:         FC layers within each block.
        trend_degree:        Polynomial degree for trend blocks.
        n_harmonics:         Fourier harmonics for seasonality blocks.
        n_generic_stacks:    Number of generic stacks (for complex residuals).
        theta_size:          Generic block theta dimension.
        covariate_size:      Optional exogenous covariate size (IOB, COB, etc.).
        dropout:             Dropout rate.
    """

    def __init__(
        self,
        input_size: int = 72,
        prediction_horizons: List[int] = None,
        hidden_size: int = 256,
        n_blocks_per_stack: int = 3,
        n_fc_layers: int = 4,
        trend_degree: int = 3,
        n_harmonics: int = 6,
        n_generic_stacks: int = 2,
        theta_size: int = 32,
        covariate_size: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.prediction_horizons = prediction_horizons or [6, 12, 24]
        self.max_horizon = max(self.prediction_horizons)

        # We model each horizon independently with its own N-BEATS
        # to avoid horizon interference (different uncertainties at 30/60/120 min)
        self.horizon_models = nn.ModuleList()

        for h_step in self.prediction_horizons:
            horizon = h_step  # number of forecast steps for this horizon

            # --- Trend Stack ---
            trend_blocks = nn.ModuleList([
                NBeatsBlock(
                    input_size=input_size,
                    theta_size=trend_degree + 1,
                    horizon=horizon,
                    hidden_size=hidden_size,
                    n_layers=n_fc_layers,
                    block_type="trend",
                    trend_degree=trend_degree,
                    covariate_size=covariate_size,
                    dropout=dropout,
                )
                for _ in range(n_blocks_per_stack)
            ])

            # --- Seasonality Stack ---
            season_blocks = nn.ModuleList([
                NBeatsBlock(
                    input_size=input_size,
                    theta_size=2 * n_harmonics,
                    horizon=horizon,
                    hidden_size=hidden_size,
                    n_layers=n_fc_layers,
                    block_type="seasonality",
                    n_harmonics=n_harmonics,
                    covariate_size=covariate_size,
                    dropout=dropout,
                )
                for _ in range(n_blocks_per_stack)
            ])

            # --- Generic Stacks ---
            generic_stacks_list = []
            for _ in range(n_generic_stacks):
                gen_blocks = nn.ModuleList([
                    NBeatsBlock(
                        input_size=input_size,
                        theta_size=theta_size,
                        horizon=horizon,
                        hidden_size=hidden_size,
                        n_layers=n_fc_layers,
                        block_type="generic",
                        covariate_size=covariate_size,
                        dropout=dropout,
                    )
                    for _ in range(n_blocks_per_stack)
                ])
                generic_stacks_list.append(NBeatsStack(gen_blocks))

            stacks = nn.ModuleList([
                NBeatsStack(trend_blocks),
                NBeatsStack(season_blocks),
                *generic_stacks_list,
            ])
            self.horizon_models.append(stacks)

    def forward(
        self,
        historical: torch.Tensor,             # [B, input_size] or [B, T, features]
        covariates: Optional[torch.Tensor] = None,  # [B, covariate_size]
    ) -> dict:
        """
        Args:
            historical: [B, input_size] — CGM lookback window (single variable)
                        OR [B, T, features] — if multi-feature, the first feature
                        (CGM) is extracted for the main N-BEATS backbone and
                        remaining features become covariates.
            covariates: [B, covariate_size] — optional IOB, COB, etc.

        Returns:
            dict with:
              "predictions": [B, n_horizons, 1]
              "trend_forecast": [B, n_horizons, max_horizon]
              "seasonality_forecast": [B, n_horizons, max_horizon]
        """
        if historical.dim() == 3:
            # Multi-feature input: extract CGM + build covariates
            cgm_signal = historical[:, :, 0]      # [B, T] — CGM only
            if covariates is None and historical.shape[-1] > 1:
                extra = historical[:, -1, 1:]     # Last timestep covariate values [B, n_extra]
                covariates = extra
        else:
            cgm_signal = historical  # [B, T]

        # Truncate or pad CGM signal to input_size
        T = cgm_signal.shape[1]
        if T > self.input_size:
            cgm_signal = cgm_signal[:, -self.input_size:]
        elif T < self.input_size:
            cgm_signal = F.pad(cgm_signal, (self.input_size - T, 0))

        all_predictions = []
        trend_forecasts = []
        season_forecasts = []

        for h_idx, stacks in enumerate(self.horizon_models):
            x = cgm_signal
            total_forecast = 0.0
            trend_f = None
            season_f = None

            for s_idx, stack in enumerate(stacks):
                residual, stack_forecast = stack(x, covariates)
                x = residual
                total_forecast = total_forecast + stack_forecast
                if s_idx == 0:
                    trend_f = stack_forecast      # [B, horizon_steps]
                elif s_idx == 1:
                    season_f = stack_forecast

            # Extract the final horizon step prediction
            # total_forecast shape: [B, h_step]
            pred = total_forecast[:, -1:]   # [B, 1] — last predicted step = target horizon
            all_predictions.append(pred)
            trend_forecasts.append(trend_f[:, -1:] if trend_f is not None else pred)
            season_forecasts.append(season_f[:, -1:] if season_f is not None else pred)

        predictions = torch.stack(all_predictions, dim=1)    # [B, n_horizons, 1]
        trend_out = torch.stack(trend_forecasts, dim=1)
        season_out = torch.stack(season_forecasts, dim=1)

        return {
            "predictions": predictions,
            "trend_forecast": trend_out,
            "seasonality_forecast": season_out,
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
