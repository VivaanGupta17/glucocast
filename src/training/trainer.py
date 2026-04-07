"""
GlucoCast Training Pipeline.

Handles:
  - Multi-horizon loss with horizon-specific weighting
    (near-term predictions penalised more heavily — clinical priority)
  - Quantile loss for prediction intervals
  - Population model training (all patients) + patient-specific fine-tuning
  - Mixed precision training (torch.cuda.amp)
  - Early stopping on validation MAE with configurable patience
  - Learning rate scheduling (cosine annealing with warm restarts)
  - Gradient clipping for stable LSTM training
  - Checkpoint saving with best-model tracking

Loss Design Rationale:
  The 30-minute prediction horizon has highest clinical priority (hypoglycemia
  prevention). Accordingly, loss weights default to [0.5, 0.3, 0.2] for
  [30-min, 60-min, 120-min]. This ensures the model is optimised for the
  near-term safety-critical window, while 60/120-min accuracy supports
  longer-term AID system planning.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class MultiHorizonQuantileLoss(nn.Module):
    """
    Combined quantile + horizon-weighted loss for multi-horizon prediction.

    For each prediction horizon h and quantile q:
        L_q(r) = q * max(r, 0) + (1-q) * max(-r, 0)   (pinball loss)
        where r = y_true - y_pred

    Total loss:
        L = sum_h(w_h * mean_q(L_q(y_true_h, y_pred_h)))

    For point prediction models (no quantile output), falls back to
    horizon-weighted Huber loss (robust to outliers from sensor noise).

    Args:
        quantiles:        List of quantile levels (e.g. [0.1, 0.5, 0.9]).
                          Pass None for point-prediction models.
        horizon_weights:  Per-horizon loss weights. Must sum to 1.
                          Default: [0.5, 0.3, 0.2] favours 30-min horizon.
        huber_delta:      Delta parameter for Huber loss fallback.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        horizon_weights: Optional[List[float]] = None,
        huber_delta: float = 15.0,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.huber_delta = huber_delta

        if quantiles is not None:
            self.register_buffer(
                "q_tensor", torch.tensor(quantiles, dtype=torch.float32)
            )

        if horizon_weights is not None:
            self.register_buffer(
                "h_weights", torch.tensor(horizon_weights, dtype=torch.float32)
            )
        else:
            self.h_weights = None

    def forward(
        self,
        predictions: torch.Tensor,   # [B, n_horizons, n_quantiles] or [B, n_horizons, 1]
        targets: torch.Tensor,        # [B, n_horizons]
    ) -> torch.Tensor:
        B, n_horizons = targets.shape
        n_quantiles = predictions.shape[-1]

        if self.h_weights is not None:
            assert len(self.h_weights) == n_horizons, (
                f"horizon_weights length {len(self.h_weights)} != n_horizons {n_horizons}"
            )
            h_w = self.h_weights
        else:
            h_w = torch.ones(n_horizons, device=targets.device) / n_horizons

        total_loss = torch.tensor(0.0, device=targets.device)

        for h_idx in range(n_horizons):
            y_pred_h = predictions[:, h_idx, :]  # [B, n_quantiles]
            y_true_h = targets[:, h_idx]          # [B]

            if self.quantiles is not None:
                # Quantile (pinball) loss
                y_true_expanded = y_true_h.unsqueeze(-1)  # [B, 1]
                errors = y_true_expanded - y_pred_h        # [B, n_quantiles]
                q = self.q_tensor.view(1, -1)
                h_loss = torch.max((q - 1) * errors, q * errors).mean()
            else:
                # Huber loss for point prediction
                y_pred_point = y_pred_h[:, 0]   # Take first (and only) quantile
                h_loss = nn.functional.huber_loss(
                    y_pred_point, y_true_h, delta=self.huber_delta
                )

            total_loss = total_loss + h_w[h_idx] * h_loss

        return total_loss


# ---------------------------------------------------------------------------
# Training State
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyperparameters for a training run."""
    # Optimiser
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_t0: int = 10        # CosineAnnealingWarmRestarts T_0
    scheduler_t_mult: int = 2     # T_mult for warm restarts
    scheduler_eta_min: float = 1e-6

    # Training loop
    max_epochs: int = 100
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_mae_30min"
    batch_size: int = 64

    # Loss
    quantiles: Optional[List[float]] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    horizon_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    huber_delta: float = 15.0   # ~1 SD of typical glucose variability

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5

    # Fine-tuning
    finetune_epochs: int = 20
    finetune_lr: float = 1e-5
    freeze_base: bool = False   # If True, freeze all but output layers during fine-tune


@dataclass
class TrainingState:
    """Mutable training state tracked across epochs."""
    epoch: int = 0
    best_val_metric: float = float("inf")
    best_epoch: int = 0
    patience_counter: int = 0
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_mae_by_horizon: List[Dict[str, float]] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GlucocCastTrainer:
    """
    Training orchestrator for GlucoCast models.

    Handles population training (cross-patient) and patient-specific
    fine-tuning as two distinct phases.

    Phase 1 — Population Training:
        Train on all patients (leave-one-out or full cohort).
        Objective: learn general glucose dynamics, IOB/COB patterns.

    Phase 2 — Patient Fine-tuning (optional):
        Starting from population checkpoint, fine-tune on individual
        patient data with low learning rate. Objective: adapt to patient-
        specific insulin sensitivity, meal absorption rate, circadian pattern.

    Usage:
        trainer = GlucocCastTrainer(model, config)
        trainer.train(train_loader, val_loader)
        trainer.fine_tune(patient_train_loader, patient_val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        prediction_horizons: List[int] = None,
        cgm_norm_params: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.prediction_horizons = prediction_horizons or [6, 12, 24]
        self.cgm_norm_params = cgm_norm_params   # (mean, std) for denormalisation
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.model = self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        # Loss
        self.criterion = MultiHorizonQuantileLoss(
            quantiles=config.quantiles,
            horizon_weights=config.horizon_weights,
            huber_delta=config.huber_delta,
        ).to(self.device)

        # Optimiser
        self.optimiser = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimiser,
            T_0=config.scheduler_t0,
            T_mult=config.scheduler_t_mult,
            eta_min=config.scheduler_eta_min,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and self.device.type == "cuda" else None

        self.state = TrainingState()
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingState:
        """
        Full training loop with early stopping.

        Monitors `config.early_stopping_metric` on the validation set.
        Saves the best model checkpoint when the metric improves.

        Returns:
            TrainingState with complete training history.
        """
        logger.info(
            f"Starting training: {self.config.max_epochs} max epochs, "
            f"patience={self.config.early_stopping_patience}, "
            f"metric={self.config.early_stopping_metric}"
        )

        for epoch in range(1, self.config.max_epochs + 1):
            self.state.epoch = epoch
            t0 = time.time()

            # --- Train Epoch ---
            train_loss = self._train_epoch(train_loader)

            # --- Validation ---
            val_metrics = self._validate(val_loader)
            val_loss = val_metrics["val_loss"]
            val_mae = val_metrics.get(self.config.early_stopping_metric, val_loss)

            # LR scheduler step
            self.scheduler.step()
            current_lr = self.optimiser.param_groups[0]["lr"]

            # --- Logging ---
            elapsed = time.time() - t0
            self.state.train_losses.append(train_loss)
            self.state.val_losses.append(val_loss)
            self.state.learning_rates.append(current_lr)

            logger.info(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | {self.config.early_stopping_metric}={val_mae:.2f} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s"
            )

            # --- Early Stopping ---
            if val_mae < self.state.best_val_metric:
                self.state.best_val_metric = val_mae
                self.state.best_epoch = epoch
                self.state.patience_counter = 0
                self._save_checkpoint("best_model.pt", epoch, val_metrics)
                logger.info(f"  ↑ New best model saved (epoch {epoch})")
            else:
                self.state.patience_counter += 1
                if self.state.patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(best was epoch {self.state.best_epoch})"
                    )
                    break

            # --- Periodic Checkpoint ---
            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, val_metrics)

        logger.info(
            f"Training complete. Best epoch: {self.state.best_epoch}, "
            f"best {self.config.early_stopping_metric}: {self.state.best_val_metric:.2f}"
        )
        return self.state

    def fine_tune(
        self,
        patient_train_loader: DataLoader,
        patient_val_loader: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> TrainingState:
        """
        Patient-specific fine-tuning from a population checkpoint.

        Optionally freezes the base network and only trains output layers
        (config.freeze_base=True) for low-data regimes.

        Args:
            patient_train_loader: DataLoader for single patient training data
            patient_val_loader:   DataLoader for single patient validation data
            checkpoint_path:      Path to population checkpoint (or None to use best)
        """
        # Load population checkpoint
        ckpt_path = checkpoint_path or str(self.checkpoint_dir / "best_model.pt")
        self._load_checkpoint(ckpt_path)
        logger.info(f"Fine-tuning from checkpoint: {ckpt_path}")

        if self.config.freeze_base:
            self._freeze_base_layers()
            logger.info("Froze base model layers — training output heads only")

        # Lower LR for fine-tuning
        for pg in self.optimiser.param_groups:
            pg["lr"] = self.config.finetune_lr

        # Reset scheduler for short fine-tuning run
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimiser,
            T_0=self.config.finetune_epochs,
            T_mult=1,
            eta_min=self.config.scheduler_eta_min,
        )

        # Reset early stopping state
        self.state = TrainingState()
        old_max_epochs = self.config.max_epochs
        old_patience = self.config.early_stopping_patience
        self.config.max_epochs = self.config.finetune_epochs
        self.config.early_stopping_patience = max(5, self.config.finetune_epochs // 4)

        result = self.train(patient_train_loader, patient_val_loader)

        self.config.max_epochs = old_max_epochs
        self.config.early_stopping_patience = old_patience
        return result

    def _train_epoch(self, loader: DataLoader) -> float:
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = self._to_device(batch)
            targets_norm = batch["targets_norm"]   # [B, n_horizons]

            self.optimiser.zero_grad()

            if self.scaler is not None:
                with autocast():
                    output = self._forward(batch)
                    predictions = output["predictions"]   # [B, n_horizons, n_q]
                    loss = self.criterion(predictions, targets_norm)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                output = self._forward(batch)
                predictions = output["predictions"]
                loss = self.criterion(predictions, targets_norm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimiser.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validation pass. Computes loss and per-horizon MAE in mg/dL."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = [[] for _ in self.prediction_horizons]  # Per-horizon predictions
        all_targets = [[] for _ in self.prediction_horizons]

        for batch in loader:
            batch = self._to_device(batch)
            targets_raw = batch["targets"]         # [B, n_horizons] in mg/dL
            targets_norm = batch["targets_norm"]

            output = self._forward(batch)
            predictions = output["predictions"]    # [B, n_horizons, n_q]

            loss = self.criterion(predictions, targets_norm)
            total_loss += loss.item()
            n_batches += 1

            # Extract median predictions (quantile index 1 for [0.1, 0.5, 0.9])
            if predictions.shape[-1] > 1:
                median_pred = predictions[:, :, 1]   # [B, n_horizons]
            else:
                median_pred = predictions[:, :, 0]

            # Denormalise to mg/dL
            if self.cgm_norm_params is not None:
                mu, sigma = self.cgm_norm_params
                median_pred_mgdl = median_pred * sigma + mu
            else:
                median_pred_mgdl = median_pred

            for h_idx in range(len(self.prediction_horizons)):
                all_preds[h_idx].append(median_pred_mgdl[:, h_idx].cpu().numpy())
                all_targets[h_idx].append(targets_raw[:, h_idx].cpu().numpy())

        metrics = {"val_loss": total_loss / max(n_batches, 1)}

        horizon_names = ["30min", "60min", "120min"]
        for h_idx, h_name in enumerate(horizon_names[:len(self.prediction_horizons)]):
            preds = np.concatenate(all_preds[h_idx])
            tgts = np.concatenate(all_targets[h_idx])
            valid = ~np.isnan(preds) & ~np.isnan(tgts)
            if valid.sum() > 0:
                mae = float(np.abs(preds[valid] - tgts[valid]).mean())
                rmse = float(np.sqrt(((preds[valid] - tgts[valid]) ** 2).mean()))
                metrics[f"val_mae_{h_name}"] = mae
                metrics[f"val_rmse_{h_name}"] = rmse

        return metrics

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Model-agnostic forward pass.

        Detects model type by checking its expected inputs.
        TFT expects (static, historical, future).
        LSTM/TCN/NBEATS expect (historical, future).
        """
        historical = batch["historical"]
        future = batch["future"]
        model_type = type(self.model).__name__

        if "TemporalFusionTransformer" in model_type:
            # TFT requires static covariates — use zeros if not available
            B = historical.shape[0]
            n_static = getattr(self.model, "num_static_vars", 5) if hasattr(self.model, "static_var_embeddings") else 5
            static = batch.get("static", torch.zeros(B, n_static, device=self.device))
            return self.model(static=static, historical=historical, future=future)
        else:
            return self.model(historical=historical, future=future)

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _save_checkpoint(
        self, filename: str, epoch: int, metrics: Dict
    ) -> None:
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "training_state": self.state,
            "config": self.config,
            "cgm_norm_params": self.cgm_norm_params,
        }, path)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "cgm_norm_params" in ckpt:
            self.cgm_norm_params = ckpt["cgm_norm_params"]
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")

    def _freeze_base_layers(self) -> None:
        """Freeze all parameters except the final output projection layers."""
        for name, param in self.model.named_parameters():
            if "output" not in name and "head" not in name and "proj" not in name:
                param.requires_grad = False
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Frozen base layers. Trainable parameters: {n_trainable:,}")

    @classmethod
    def from_checkpoint(cls, path: str, model: nn.Module, **kwargs) -> "GlucocCastTrainer":
        """Load a trainer from a saved checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        config = ckpt.get("config", TrainingConfig())
        trainer = cls(model=model, config=config, **kwargs)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.cgm_norm_params = ckpt.get("cgm_norm_params")
        return trainer
