#!/usr/bin/env python3
"""
GlucoCast Real-Time Prediction Script.

Demonstrates the real-time prediction pipeline using the GlucosePredictor.
Can read CGM data from:
  - A CSV file (simulated real-time replay)
  - Standard input (JSON lines for real integration)
  - The OhioT1DM test set (demo mode)

Usage:
    # Demo mode: replay OhioT1DM patient test data
    python scripts/predict.py \\
        --checkpoint checkpoints/best_model.pt \\
        --demo --patient 559

    # CSV replay (column: timestamp, cgm_value)
    python scripts/predict.py \\
        --checkpoint checkpoints/best_model.pt \\
        --csv data/my_cgm.csv

    # Streaming via stdin (JSON lines)
    cat cgm_stream.json | python scripts/predict.py \\
        --checkpoint checkpoints/best_model.pt \\
        --stdin

    # Show all alerts
    python scripts/predict.py \\
        --checkpoint checkpoints/best_model.pt \\
        --demo --patient 559 --show-alerts-only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from src.inference.realtime_predictor import (
    AlertLevel,
    AlertConfig,
    GlucosePredictor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")


def parse_args():
    parser = argparse.ArgumentParser(description="GlucoCast real-time prediction")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="configs/ohio_config.yaml")
    parser.add_argument("--demo", action="store_true", help="Demo mode with OhioT1DM data")
    parser.add_argument("--patient", default="559", help="Patient ID for demo mode")
    parser.add_argument("--csv", default=None, help="CSV file with CGM readings")
    parser.add_argument("--stdin", action="store_true", help="Read JSON lines from stdin")
    parser.add_argument("--show-alerts-only", action="store_true",
                        help="Only print when an alert is generated")
    parser.add_argument("--replay-speed", type=float, default=0.0,
                        help="Seconds between readings in demo (0=fastest)")
    # Alert thresholds
    parser.add_argument("--hypo-threshold", type=float, default=70.0)
    parser.add_argument("--hyper-threshold", type=float, default=250.0)
    return parser.parse_args()


def format_forecast(forecast, show_all: bool = True) -> str:
    """Format a GlucoseForecast for terminal display."""
    alert_symbols = {
        AlertLevel.NONE: "✓",
        AlertLevel.HYPO_RISK: "⚠ HYPO",
        AlertLevel.HYPO_SEVERE_RISK: "🚨 SEVERE HYPO",
        AlertLevel.HYPER_RISK: "⚠ HYPER",
        AlertLevel.RISING_FAST: "↑↑",
        AlertLevel.FALLING_FAST: "↓↓",
        AlertLevel.LOW_CONFIDENCE: "~",
    }

    roc_arrow = "→"
    if forecast.current_roc > 2.0:
        roc_arrow = "↑↑"
    elif forecast.current_roc > 1.0:
        roc_arrow = "↑"
    elif forecast.current_roc < -2.0:
        roc_arrow = "↓↓"
    elif forecast.current_roc < -1.0:
        roc_arrow = "↓"

    wu = "" if forecast.warm_up_complete else " [WARM-UP]"
    lines = [
        f"[{forecast.timestamp[:19]}]{wu} Current: {forecast.current_cgm:.0f} mg/dL {roc_arrow}"
        f"  ({forecast.current_roc:+.1f} mg/dL/min)"
    ]

    for h_name, pred in forecast.predictions.items():
        alert_str = alert_symbols.get(pred.alert, "")
        if pred.alert != AlertLevel.NONE or show_all:
            lines.append(
                f"  +{h_name:>6}: {pred.point:>5.0f} mg/dL "
                f"[{pred.lower:.0f}-{pred.upper:.0f}]  {alert_str}"
            )

    if forecast.max_alert != AlertLevel.NONE:
        lines.append(f"  >>> ALERT: {forecast.max_alert.value.upper()}")

    return "\n".join(lines)


def run_demo(args, predictor: GlucosePredictor, cfg: dict):
    """Demo mode: replay OhioT1DM patient test data."""
    from src.data.ohio_dataset import OhioT1DM
    from torch.utils.data import DataLoader

    dataset = OhioT1DM(
        data_dir=cfg["data"]["data_dir"],
        encoder_steps=cfg["data"]["encoder_steps"],
        prediction_horizons=cfg["data"]["prediction_horizons"],
        batch_size=1,
    )

    df = dataset.build_feature_dataframe(args.patient)
    logger.info(f"Replaying {len(df)} CGM readings for patient {args.patient}")

    n_alerts = 0
    n_total = 0

    for ts, row in df.iterrows():
        cgm_val = row["cgm"]
        if np.isnan(cgm_val):
            continue

        ts_str = ts.isoformat()
        forecast = predictor.update(
            cgm_value=float(cgm_val),
            timestamp=ts_str,
            iob=float(row.get("iob", 0)),
            cob=float(row.get("cob", 0)),
        )

        n_total += 1

        if forecast is not None:
            has_alert = forecast.max_alert != AlertLevel.NONE
            if has_alert:
                n_alerts += 1

            if not args.show_alerts_only or has_alert:
                print(format_forecast(forecast, show_all=not args.show_alerts_only))

        if args.replay_speed > 0:
            time.sleep(args.replay_speed)

    logger.info(f"Demo complete: {n_total} readings, {n_alerts} alerts generated")


def run_csv(args, predictor: GlucosePredictor):
    """Process CGM readings from a CSV file."""
    import csv

    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp") or row.get("time") or row.get("ts")
            cgm = float(row.get("cgm_value") or row.get("cgm") or row.get("glucose"))
            iob = float(row.get("iob", 0))
            cob = float(row.get("cob", 0))

            forecast = predictor.update(
                cgm_value=cgm,
                timestamp=ts,
                iob=iob,
                cob=cob,
            )
            if forecast is not None:
                has_alert = forecast.max_alert != AlertLevel.NONE
                if not args.show_alerts_only or has_alert:
                    print(format_forecast(forecast, show_all=not args.show_alerts_only))


def run_stdin(args, predictor: GlucosePredictor):
    """Process CGM readings from stdin (JSON lines format)."""
    print("Reading JSON lines from stdin. Format:")
    print('  {"timestamp": "2025-01-15T14:30:00", "cgm_value": 142.0, "iob": 1.5}')
    print("")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            forecast = predictor.update(
                cgm_value=float(data["cgm_value"]),
                timestamp=data["timestamp"],
                iob=float(data.get("iob", 0)),
                cob=float(data.get("cob", 0)),
                meal_carbs=float(data.get("meal_carbs", 0)),
                bolus_units=float(data.get("bolus_units", 0)),
            )
            if forecast is not None:
                # Output JSON for downstream processing
                output = {
                    "timestamp": forecast.timestamp,
                    "current_cgm": forecast.current_cgm,
                    "roc": forecast.current_roc,
                    "predictions": {
                        h: pred.to_dict() for h, pred in forecast.predictions.items()
                    },
                    "max_alert": forecast.max_alert.value,
                    "warm_up_complete": forecast.warm_up_complete,
                }
                print(json.dumps(output))
                sys.stdout.flush()
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Skipping invalid line: {e}")


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # Configure alerts
    alert_config = AlertConfig(
        hypo_alert_mgdl=args.hypo_threshold,
        hyper_alert_mgdl=args.hyper_threshold,
    )

    # Load predictor
    logger.info(f"Loading predictor from {args.checkpoint}")
    predictor = GlucosePredictor.from_checkpoint(
        args.checkpoint,
        alert_config=alert_config,
    )

    status = predictor.get_status()
    logger.info(f"Predictor status: {status}")

    print(f"\nGlucoCast Predictor | Model: {predictor.model_name}")
    print(f"Encoder window: {predictor.encoder_steps * 5} min | "
          f"Horizons: {predictor.horizon_minutes} min")
    print(f"Hypo alert: <{args.hypo_threshold} mg/dL | "
          f"Hyper alert: >{args.hyper_threshold} mg/dL")
    print("─" * 70)

    if args.demo:
        run_demo(args, predictor, cfg)
    elif args.csv:
        run_csv(args, predictor)
    elif args.stdin:
        run_stdin(args, predictor)
    else:
        logger.error("Provide --demo, --csv, or --stdin")
        sys.exit(1)


if __name__ == "__main__":
    main()
