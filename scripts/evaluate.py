#!/usr/bin/env python3
"""
GlucoCast Evaluation Script.

Evaluates a trained model checkpoint on the OhioT1DM test set.
Produces clinical evaluation metrics including Clarke Error Grid analysis,
hypoglycemia detection performance, and per-patient safety profiles.

Usage:
    # Standard evaluation
    python scripts/evaluate.py \\
        --config configs/ohio_config.yaml \\
        --checkpoint checkpoints/best_model.pt

    # Full clinical safety analysis
    python scripts/evaluate.py \\
        --config configs/ohio_config.yaml \\
        --checkpoint checkpoints/best_model.pt \\
        --clarke-grid --hypo-analysis --per-patient

    # Compare multiple models
    python scripts/evaluate.py \\
        --config configs/ohio_config.yaml \\
        --checkpoints checkpoints/tft.pt checkpoints/lstm.pt \\
        --model-names TFT LSTM

    # Save results to JSON
    python scripts/evaluate.py \\
        --config configs/ohio_config.yaml \\
        --checkpoint checkpoints/best_model.pt \\
        --output results/evaluation_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml

from src.data.ohio_dataset import OhioT1DM
from src.evaluation.glucose_metrics import (
    evaluate_predictions,
    print_evaluation_table,
)
from src.evaluation.clinical_safety import (
    evaluate_hypo_alerts,
    patient_safety_analysis,
    print_safety_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GlucoCast glucose predictions"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint to evaluate")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Multiple checkpoints for comparison")
    parser.add_argument("--model-names", nargs="+", default=None,
                        help="Names for multiple checkpoints")
    parser.add_argument("--patient", default=None,
                        help="Evaluate on specific patient only")
    parser.add_argument("--clarke-grid", action="store_true",
                        help="Print Clarke Error Grid breakdown")
    parser.add_argument("--hypo-analysis", action="store_true",
                        help="Run hypoglycemia detection analysis")
    parser.add_argument("--per-patient", action="store_true",
                        help="Print per-patient breakdown")
    parser.add_argument("--output", default=None,
                        help="Save JSON results to this path")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save evaluation plots")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, cfg: dict):
    """Load model from checkpoint, inferring architecture from state dict."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    prediction_horizons = cfg["data"]["prediction_horizons"]

    # Infer model type from parameter keys
    if any("static_vsn" in k for k in state_dict.keys()):
        from src.models.temporal_fusion_transformer import TemporalFusionTransformer
        mc = cfg["model"]["tft"]
        model = TemporalFusionTransformer(
            num_static_vars=mc["num_static_vars"],
            num_historical_vars=mc["num_historical_vars"],
            num_future_vars=mc["num_future_vars"],
            hidden_size=mc["hidden_size"],
            num_heads=mc["num_heads"],
            num_lstm_layers=mc["num_lstm_layers"],
            dropout=0.0,
            encoder_steps=cfg["data"]["encoder_steps"],
            prediction_horizons=prediction_horizons,
            quantiles=mc["quantiles"],
        )
        model_name = "TFT"
    elif any("encoder_lstm" in k for k in state_dict.keys()):
        from src.models.lstm_glucose import GlucoseLSTM
        mc = cfg["model"]["lstm"]
        model = GlucoseLSTM(
            input_size=len(cfg["features"]["historical_features"]),
            hidden_size=mc["hidden_size"],
            encoder_steps=cfg["data"]["encoder_steps"],
            prediction_horizons=prediction_horizons,
            dropout=0.0,
        )
        model_name = "LSTM"
    else:
        raise ValueError(f"Cannot infer model type from checkpoint: {checkpoint_path}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    cgm_norm = ckpt.get("cgm_norm_params", (120.0, 40.0))
    return model, model_name, cgm_norm


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader,
    cgm_norm_params,
    prediction_horizons: list,
    device: torch.device,
) -> dict:
    """Run model inference on a DataLoader, collect predictions and targets."""
    mu, sigma = cgm_norm_params
    all_preds = []
    all_targets = []

    model.eval()
    model = model.to(device)

    for batch in loader:
        historical = batch["historical"].to(device)
        future = batch["future"].to(device)
        targets = batch["targets"].cpu().numpy()  # mg/dL

        model_type = type(model).__name__
        if "TemporalFusionTransformer" in model_type:
            B = historical.shape[0]
            static = torch.zeros(B, 5, device=device)
            output = model(static=static, historical=historical, future=future)
        else:
            output = model(historical=historical, future=future)

        preds_norm = output["predictions"].cpu().numpy()  # [B, n_h, n_q]
        # Denormalise
        preds_mgdl = preds_norm * sigma + mu

        all_preds.append(preds_mgdl)
        all_targets.append(targets)

    y_pred = np.concatenate(all_preds, axis=0)    # [N, n_horizons, n_quantiles]
    y_true = np.concatenate(all_targets, axis=0)  # [N, n_horizons]
    return {"predictions": y_pred, "targets": y_true}


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Evaluating on device: {device}")

    prediction_horizons = cfg["data"]["prediction_horizons"]
    horizons_min = cfg["evaluation"]["prediction_horizons_min"]

    # Collect checkpoints to evaluate
    checkpoints = []
    if args.checkpoint:
        checkpoints = [(args.checkpoint, "Model")]
    elif args.checkpoints:
        names = args.model_names or [Path(c).stem for c in args.checkpoints]
        checkpoints = list(zip(args.checkpoints, names))
    else:
        logger.error("Provide --checkpoint or --checkpoints")
        sys.exit(1)

    # Load dataset
    dataset = OhioT1DM(
        data_dir=cfg["data"]["data_dir"],
        encoder_steps=cfg["data"]["encoder_steps"],
        prediction_horizons=prediction_horizons,
        batch_size=cfg["data"]["batch_size"],
    )

    patient_ids = [str(args.patient)] if args.patient else [str(p) for p in cfg["data"]["patient_ids"]]
    all_results = {}

    for ckpt_path, model_name in checkpoints:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name} ({ckpt_path})")
        logger.info(f"{'='*60}")

        model, inferred_name, cgm_norm = load_model_from_checkpoint(ckpt_path, cfg)
        all_preds = []
        all_tgts = []
        patient_results = {}

        for pid in patient_ids:
            try:
                _, _, test_ds = dataset.get_patient_splits(pid)
                from torch.utils.data import DataLoader
                test_loader = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"],
                                        shuffle=False)

                result = run_evaluation(model, test_loader, cgm_norm,
                                        prediction_horizons, device)
                all_preds.append(result["predictions"])
                all_tgts.append(result["targets"])

                # Per-patient metrics
                metrics = evaluate_predictions(
                    result["targets"], result["predictions"], horizons_min
                )
                patient_results[pid] = metrics

                if args.per_patient:
                    print(f"\nPatient {pid}:")
                    print(print_evaluation_table(metrics))

                if args.hypo_analysis:
                    # Use 30-min predictions for alert analysis
                    cgm_true = result["targets"][:, 0]   # 30-min true values
                    cgm_pred_30 = result["predictions"][:, 0, 1]   # 30-min median
                    timestamps = np.arange(len(cgm_true), dtype=float) * 5 * 60   # Fake timestamps
                    alert_metrics = evaluate_hypo_alerts(
                        cgm_true, cgm_pred_30, timestamps.astype(object),
                        alert_horizon_min=30
                    )
                    logger.info(
                        f"  Patient {pid} Hypo Alert: "
                        f"Sens={alert_metrics.sensitivity:.1%}, "
                        f"Spec={alert_metrics.specificity:.1%}, "
                        f"FA/day={alert_metrics.false_alarm_rate_per_day:.1f}"
                    )

            except Exception as e:
                logger.warning(f"Could not evaluate patient {pid}: {e}")

        if all_preds:
            # Aggregate across patients
            y_pred_all = np.concatenate(all_preds, axis=0)
            y_true_all = np.concatenate(all_tgts, axis=0)

            logger.info(f"\nAggregate results ({len(patient_ids)} patients):")
            agg_metrics = evaluate_predictions(y_true_all, y_pred_all, horizons_min)
            print("\n" + print_evaluation_table(agg_metrics))

            if args.clarke_grid:
                print("\nClarke Error Grid Details:")
                for m in agg_metrics:
                    print(f"\n{m.horizon_min}-min Horizon:")
                    print(str(m.clarke_ega))

            all_results[model_name] = {
                "aggregate": [
                    {
                        "horizon_min": m.horizon_min,
                        "rmse": m.rmse,
                        "mae": m.mae,
                        "mard": m.mard,
                        "clarke_a": m.clarke_ega.zone_a,
                        "clarke_ab": m.clarke_ega.zone_ab,
                        "coverage_90": m.coverage_90,
                    }
                    for m in agg_metrics
                ],
                "per_patient": {
                    pid: [
                        {"horizon_min": m.horizon_min, "rmse": m.rmse, "mae": m.mae}
                        for m in pm
                    ]
                    for pid, pm in patient_results.items()
                },
            }

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
