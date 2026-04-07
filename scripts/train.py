#!/usr/bin/env python3
"""
GlucoCast Training Script.

Usage:
    # Population model (all patients)
    python scripts/train.py --config configs/ohio_config.yaml --model tft

    # Single patient
    python scripts/train.py --config configs/ohio_config.yaml --model tft --patient 559

    # Patient fine-tuning from population checkpoint
    python scripts/train.py --config configs/ohio_config.yaml --model tft \
        --patient 559 --checkpoint checkpoints/population_tft.pt --finetune

    # Leave-one-patient-out cross-validation
    python scripts/train.py --config configs/ohio_config.yaml --model tft --lopocv

    # With custom hyperparameters (override config)
    python scripts/train.py --config configs/ohio_config.yaml \\
        --model tft --lr 1e-3 --batch-size 128 --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import numpy as np

from src.data.ohio_dataset import OhioT1DM
from src.training.trainer import GlucocCastTrainer, TrainingConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GlucoCast glucose prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--model", default="tft",
        choices=["tft", "lstm", "tcn", "nbeats"],
        help="Model architecture"
    )
    parser.add_argument("--patient", default=None, help="Single patient ID (None = all patients)")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune from checkpoint")
    parser.add_argument("--lopocv", action="store_true", help="Leave-one-patient-out CV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Override config values
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs override")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Primary horizon in minutes (30|60|120)")
    parser.add_argument("--output-dir", default="checkpoints", help="Checkpoint output directory")

    return parser.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_name: str, cfg: dict, prediction_horizons: list):
    """Instantiate the requested model architecture."""
    if model_name == "tft":
        from src.models.temporal_fusion_transformer import TemporalFusionTransformer
        mc = cfg["model"]["tft"]
        return TemporalFusionTransformer(
            num_static_vars=mc["num_static_vars"],
            num_historical_vars=mc["num_historical_vars"],
            num_future_vars=mc["num_future_vars"],
            hidden_size=mc["hidden_size"],
            num_heads=mc["num_heads"],
            num_lstm_layers=mc["num_lstm_layers"],
            dropout=mc["dropout"],
            encoder_steps=cfg["data"]["encoder_steps"],
            prediction_horizons=prediction_horizons,
            quantiles=mc["quantiles"],
        )
    elif model_name == "lstm":
        from src.models.lstm_glucose import GlucoseLSTM
        mc = cfg["model"]["lstm"]
        return GlucoseLSTM(
            input_size=len(cfg["features"]["historical_features"]),
            hidden_size=mc["hidden_size"],
            encoder_steps=cfg["data"]["encoder_steps"],
            prediction_horizons=prediction_horizons,
            num_encoder_layers=mc["num_encoder_layers"],
            num_decoder_layers=mc["num_decoder_layers"],
            dropout=mc["dropout"],
            teacher_forcing_ratio=mc["teacher_forcing_ratio"],
        )
    elif model_name == "tcn":
        from src.models.tcn_glucose import TCNGlucose
        mc = cfg["model"]["tcn"]
        return TCNGlucose(
            input_size=len(cfg["features"]["historical_features"]),
            n_channels=mc["n_channels"],
            kernel_size=mc["kernel_size"],
            num_blocks=mc["num_blocks"],
            dropout=mc["dropout"],
            encoder_steps=cfg["data"]["encoder_steps"],
            prediction_horizons=prediction_horizons,
            multi_scale_dilations=mc["multi_scale_dilations"],
        )
    elif model_name == "nbeats":
        from src.models.nbeats_glucose import NBeatsGlucose
        mc = cfg["model"]["nbeats"]
        return NBeatsGlucose(
            input_size=cfg["data"]["encoder_steps"],
            prediction_horizons=prediction_horizons,
            hidden_size=mc["hidden_size"],
            n_blocks_per_stack=mc["n_blocks_per_stack"],
            n_fc_layers=mc["n_fc_layers"],
            trend_degree=mc["trend_degree"],
            n_harmonics=mc["n_harmonics"],
            n_generic_stacks=mc["n_generic_stacks"],
            theta_size=mc["theta_size"],
            dropout=mc["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_single_patient(args, cfg, patient_id: str, output_dir: Path):
    """Train or fine-tune on a single patient."""
    tc = cfg["training"]
    prediction_horizons = cfg["data"]["prediction_horizons"]

    # Build dataset
    dataset = OhioT1DM(
        data_dir=cfg["data"]["data_dir"],
        encoder_steps=cfg["data"]["encoder_steps"],
        prediction_horizons=prediction_horizons,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    train_loader, val_loader, test_loader = dataset.get_dataloaders(patient_id)
    cgm_norm_params = dataset._patient_cache.get(patient_id) and None   # Handled internally

    # Build model
    model = build_model(args.model, cfg, prediction_horizons)
    logger.info(f"Model: {type(model).__name__}, parameters: {model.num_parameters:,}")

    # Build training config
    train_config = TrainingConfig(
        learning_rate=args.lr or tc["learning_rate"],
        weight_decay=tc["weight_decay"],
        max_epochs=args.epochs or tc["max_epochs"],
        early_stopping_patience=tc["early_stopping_patience"],
        quantiles=tc["quantiles"],
        horizon_weights=tc["horizon_weights"],
        use_amp=tc["use_amp"],
        checkpoint_dir=str(output_dir / patient_id),
        finetune_lr=tc["finetune_lr"],
        finetune_epochs=tc["finetune_epochs"],
    )
    if args.batch_size:
        train_config.batch_size = args.batch_size

    trainer = GlucocCastTrainer(
        model=model,
        config=train_config,
        prediction_horizons=prediction_horizons,
    )

    if args.finetune and args.checkpoint:
        state = trainer.fine_tune(train_loader, val_loader, checkpoint_path=args.checkpoint)
    else:
        state = trainer.train(train_loader, val_loader)

    logger.info(
        f"Patient {patient_id} training complete. "
        f"Best epoch: {state.best_epoch}, "
        f"best val metric: {state.best_val_metric:.2f}"
    )
    return state


def train_population(args, cfg, output_dir: Path):
    """Train a population model on all patients."""
    tc = cfg["training"]
    prediction_horizons = cfg["data"]["prediction_horizons"]
    patient_ids = cfg["data"]["patient_ids"]

    dataset = OhioT1DM(
        data_dir=cfg["data"]["data_dir"],
        encoder_steps=cfg["data"]["encoder_steps"],
        prediction_horizons=prediction_horizons,
        batch_size=cfg["data"]["batch_size"],
    )

    # Combine all patients for population training
    import pandas as pd
    all_train_dfs = []
    all_val_dfs = []
    for pid in patient_ids:
        try:
            train_ds, val_ds, _ = dataset.get_patient_splits(str(pid))
            all_train_dfs.append(train_ds)
            all_val_dfs.append(val_ds)
        except Exception as e:
            logger.warning(f"Could not load patient {pid}: {e}")

    from torch.utils.data import ConcatDataset, DataLoader
    combined_train = ConcatDataset(all_train_dfs)
    combined_val = ConcatDataset(all_val_dfs)
    train_loader = DataLoader(combined_train, batch_size=cfg["data"]["batch_size"],
                              shuffle=True, num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(combined_val, batch_size=cfg["data"]["batch_size"],
                            shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = build_model(args.model, cfg, prediction_horizons)
    logger.info(
        f"Population model: {type(model).__name__}, "
        f"parameters: {model.num_parameters:,}, "
        f"training patients: {len(patient_ids)}"
    )

    train_config = TrainingConfig(
        learning_rate=args.lr or tc["learning_rate"],
        max_epochs=args.epochs or tc["max_epochs"],
        early_stopping_patience=tc["early_stopping_patience"],
        quantiles=tc["quantiles"],
        horizon_weights=tc["horizon_weights"],
        use_amp=tc["use_amp"],
        checkpoint_dir=str(output_dir / "population"),
    )

    trainer = GlucocCastTrainer(model=model, config=train_config,
                                prediction_horizons=prediction_horizons)
    state = trainer.train(train_loader, val_loader)

    logger.info(f"Population training complete. Best epoch: {state.best_epoch}")
    return state


def train_lopocv(args, cfg, output_dir: Path):
    """Leave-one-patient-out cross-validation."""
    prediction_horizons = cfg["data"]["prediction_horizons"]

    dataset = OhioT1DM(
        data_dir=cfg["data"]["data_dir"],
        encoder_steps=cfg["data"]["encoder_steps"],
        prediction_horizons=prediction_horizons,
        batch_size=cfg["data"]["batch_size"],
    )

    all_results = {}
    for train_loader, test_loader, test_patient_id in dataset.loocv_splits():
        logger.info(f"\n{'='*60}")
        logger.info(f"LOPOCV: Training with test patient = {test_patient_id}")
        logger.info(f"{'='*60}")

        model = build_model(args.model, cfg, prediction_horizons)
        tc = cfg["training"]
        train_config = TrainingConfig(
            learning_rate=args.lr or tc["learning_rate"],
            max_epochs=args.epochs or tc["max_epochs"],
            early_stopping_patience=tc["early_stopping_patience"],
            quantiles=tc["quantiles"],
            horizon_weights=tc["horizon_weights"],
            use_amp=tc["use_amp"],
            checkpoint_dir=str(output_dir / f"lopocv_test_{test_patient_id}"),
        )

        trainer = GlucocCastTrainer(model=model, config=train_config,
                                    prediction_horizons=prediction_horizons)
        state = trainer.train(train_loader, test_loader)
        all_results[test_patient_id] = state

    # Summary
    logger.info("\n" + "="*60)
    logger.info("LOPOCV Summary:")
    for pid, state in all_results.items():
        logger.info(f"  Patient {pid}: best val metric = {state.best_val_metric:.2f}")
    return all_results


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"GlucoCast Training")
    logger.info(f"  Model:    {args.model.upper()}")
    logger.info(f"  Config:   {args.config}")
    logger.info(f"  Patient:  {args.patient or 'all'}")
    logger.info(f"  Output:   {output_dir}")
    logger.info(f"  Device:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    if args.lopocv:
        train_lopocv(args, cfg, output_dir)
    elif args.patient:
        train_single_patient(args, cfg, str(args.patient), output_dir)
    else:
        train_population(args, cfg, output_dir)


if __name__ == "__main__":
    main()
