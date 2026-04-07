"""
CGM-Specific Evaluation Metrics.

Standard time-series metrics (RMSE, MAE) are necessary but insufficient
for clinical CGM evaluation. This module adds:

1. Mean Absolute Relative Difference (MARD)
   The primary accuracy metric for regulatory approval of CGM devices.
   MARD = mean |CGM - Reference| / Reference * 100 (%)
   FDA requires MARD < 10% for CGM approval.
   Prediction algorithms don't face this exact bar, but it contextualises
   prediction errors relative to sensor accuracy.

2. Clarke Error Grid Analysis (EGA)
   Developed by Clarke et al. (1987) to assess clinical significance of
   glucose measurement errors. Classifies predictions into zones A-E based
   on the potential clinical outcome of acting on the prediction.

   Zones:
     A — Clinically accurate: correct treatment decision
     B — Benign error: deviation but no adverse treatment outcome
     C — Overcorrection risk: prediction would cause overcorrection
     D — Dangerous failure: prediction misses a dangerous condition
     E — Erroneous treatment: prediction would cause opposite treatment

   FDA guidance for CGM prediction (21 CFR 882): ≥ 99% in zones A+B.
   Research benchmark: ≥ 95% A+B is considered acceptable.

3. Parkes (Consensus) Error Grid
   Updated version of Clarke EGA with more refined zones and better
   applicability to Type 1 DM (Clarke was developed primarily for T2DM).
   Preferred by modern CGM evaluations.

4. Time Lag Analysis
   CGM predictions may be accurate in magnitude but temporally shifted.
   A prediction that is correct but 15 minutes early/late is clinically
   different from one that is correct at the right time.
   Lag is computed via cross-correlation between predicted and actual series.

5. Hypoglycemia-Specific Metrics
   Special attention to the clinically critical < 70 mg/dL zone:
   - Sensitivity: fraction of true hypo events detected
   - Specificity: fraction of non-hypo periods correctly classified
   - False alarm rate: false positives per day (alarm fatigue threshold)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clinical Constants
# ---------------------------------------------------------------------------

HYPO_L1 = 70      # mg/dL — Level 1 hypoglycemia (ADA 2023)
HYPO_L2 = 54      # mg/dL — Level 2 (severe) hypoglycemia
HYPER_L1 = 180    # mg/dL — Level 1 hyperglycemia
HYPER_L2 = 250    # mg/dL — Level 2 hyperglycemia


# ---------------------------------------------------------------------------
# Basic Regression Metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in mg/dL."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in mg/dL."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid])))


def mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Relative Difference (%).

    MARD = mean(|pred - true| / true) * 100

    Excludes zero reference values (undefined MARD).
    Note: true values (CGM or fingerstick) should be used as denominator,
    not predicted values.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred)) & (y_true > 0)
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid]) / y_true[valid]) * 100)


def coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Prediction interval coverage probability (PICP).

    Fraction of true values that fall within [lower, upper] bounds.
    For a 90% PI (10th-90th percentile), target coverage ≥ 0.90.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    covered = (y_true[valid] >= y_lower[valid]) & (y_true[valid] <= y_upper[valid])
    return float(covered.mean())


def mean_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Average prediction interval width (narrower is better for equal coverage)."""
    valid = ~(np.isnan(y_lower) | np.isnan(y_upper))
    return float(np.mean(y_upper[valid] - y_lower[valid]))


# ---------------------------------------------------------------------------
# Clarke Error Grid Analysis
# ---------------------------------------------------------------------------

@dataclass
class ClarkeEGAResult:
    """Results of Clarke Error Grid Analysis."""
    zone_a: float    # % clinically accurate
    zone_b: float    # % benign error
    zone_c: float    # % overcorrection risk
    zone_d: float    # % dangerous failure
    zone_e: float    # % erroneous treatment
    zone_ab: float   # % clinically acceptable (A+B)
    n_points: int
    raw_counts: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Clarke EGA (n={self.n_points}):\n"
            f"  Zone A (accurate):    {self.zone_a:6.2f}%\n"
            f"  Zone B (benign):      {self.zone_b:6.2f}%\n"
            f"  Zone C (overCorrect): {self.zone_c:6.2f}%\n"
            f"  Zone D (dangerous):   {self.zone_d:6.2f}%\n"
            f"  Zone E (erroneous):   {self.zone_e:6.2f}%\n"
            f"  ──────────────────────────────\n"
            f"  Zone A+B (acceptable):{self.zone_ab:6.2f}%\n"
        )


def clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ClarkeEGAResult:
    """
    Clarke Error Grid Analysis.

    Classifies each (reference, predicted) pair into zones A-E based on
    the original Clarke et al. (1987) boundary definitions.

    Zone A: |pred - true| ≤ 20% OR both < 70 mg/dL
    Zone B: Above/below zone A but not in C, D, or E
    Zone C: Overcorrection — pred is acceptable but true needs no treatment
            (or vice versa)
    Zone D: Dangerous failure — pred misses hypo/hyper condition
    Zone E: Erroneous treatment — pred is in opposite zone from true

    Reference:
        Clarke WL et al. (1987). Evaluating clinical accuracy of systems for
        self-monitoring of blood glucose. Diabetes Care 10(5):622-628.
        https://doi.org/10.2337/diacare.10.5.622

    Args:
        y_true: Reference glucose values (mg/dL) — fingerstick or CGM
        y_pred: Predicted glucose values (mg/dL)

    Returns:
        ClarkeEGAResult with zone percentages
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    n = len(y_true)

    if n == 0:
        return ClarkeEGAResult(0, 0, 0, 0, 0, 0, 0)

    zones = np.full(n, "B", dtype=object)

    for i in range(n):
        ref = y_true[i]
        pred = y_pred[i]
        zones[i] = _classify_clarke_zone(ref, pred)

    counts = {z: int((zones == z).sum()) for z in ["A", "B", "C", "D", "E"]}
    total = n

    return ClarkeEGAResult(
        zone_a=100 * counts["A"] / total,
        zone_b=100 * counts["B"] / total,
        zone_c=100 * counts["C"] / total,
        zone_d=100 * counts["D"] / total,
        zone_e=100 * counts["E"] / total,
        zone_ab=100 * (counts["A"] + counts["B"]) / total,
        n_points=total,
        raw_counts=counts,
    )


def _classify_clarke_zone(ref: float, pred: float) -> str:
    """
    Classify a single (reference, predicted) pair into Clarke zone A-E.

    Implements the original boundary conditions from Clarke et al. (1987).
    All thresholds in mg/dL.
    """
    # Zone A: Within 20% of reference, OR both < 70 mg/dL
    if (abs(pred - ref) / max(ref, 1e-6)) <= 0.20:
        return "A"
    if ref < 70 and pred < 70:
        return "A"

    # Zone E: Erroneous — one in hypo, other in hyper range
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"

    # Zone D: Dangerous failure
    # D upper: reference is hyperglycemic but prediction is normal/hypo
    if ref >= 240 and pred < 70:
        return "D"
    if ref < 70 and pred >= 180:
        return "D"
    # D lower: reference is hypoglycemic but prediction misses it
    if ref >= 70 and ref < 180 and pred < 70:
        return "D"
    if ref > 180 and pred >= 70 and pred < 180:
        return "D"

    # Zone C: Overcorrection
    if ref > 70 and ref < 180:
        if pred < 70:
            return "C"
        if pred > 180:
            return "C"

    # Zone B: Everything else
    return "B"


# ---------------------------------------------------------------------------
# Parkes (Consensus) Error Grid
# ---------------------------------------------------------------------------

def parkes_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    diabetes_type: int = 1,
) -> Dict[str, float]:
    """
    Parkes (Consensus) Error Grid for Type 1 or Type 2 diabetes.

    The Parkes grid uses polynomial boundary lines rather than the simpler
    linear zones of Clarke EGA. It was specifically designed for Type 1 DM
    insulin therapy decisions.

    Reference:
        Parkes JL et al. (2000). A New Consensus Error Grid to Evaluate the
        Clinical Significance of Inaccuracies in the Measurement of Blood
        Glucose. Diabetes Care 23(8):1143-1148.
        https://doi.org/10.2337/diacare.23.8.1143

    This simplified implementation uses the T1DM boundary approximation.
    Full implementation requires precise Parkes boundary coordinates.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    n = len(y_true)
    if n == 0:
        return {"zone_a": 0, "zone_b": 0, "zone_c": 0, "zone_d": 0, "zone_e": 0}

    zones = np.array([_parkes_t1dm_zone(r, p) for r, p in zip(y_true, y_pred)])
    counts = {z: int((zones == z).sum()) for z in ["A", "B", "C", "D", "E"]}
    return {f"zone_{z.lower()}": 100 * counts[z] / n for z in ["A", "B", "C", "D", "E"]}


def _parkes_t1dm_zone(ref: float, pred: float) -> str:
    """
    Classify (reference, predicted) pair into Parkes T1DM zone.

    Uses approximate boundary conditions from Parkes et al. (2000).
    For production use, replace with exact boundary coordinates.
    """
    error_pct = abs(pred - ref) / max(ref, 1e-6) * 100

    if ref < 70:   # Hypoglycemic reference
        if pred < 70 or error_pct <= 20:
            return "A"
        elif pred < 180:
            return "B"
        elif pred < 300:
            return "D"
        else:
            return "E"
    elif ref <= 180:   # Normoglycemic reference
        if error_pct <= 20:
            return "A"
        elif error_pct <= 40:
            return "B"
        elif pred < 70:
            return "D"
        else:
            return "C"
    else:   # Hyperglycemic reference
        if error_pct <= 20:
            return "A"
        elif error_pct <= 40:
            return "B"
        elif pred < 70:
            return "E"
        elif pred < 180:
            return "D"
        else:
            return "C"


# ---------------------------------------------------------------------------
# Lag Analysis
# ---------------------------------------------------------------------------

def temporal_lag_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lag_steps: int = 12,
    step_minutes: int = 5,
) -> Dict[str, float]:
    """
    Estimate systematic temporal lag between predicted and actual glucose.

    Cross-correlation measures similarity between two sequences as a function
    of time shift. The lag that maximises cross-correlation indicates whether
    the model predicts too early or too late.

    Positive lag: model predicts future events early (good — leads actual)
    Negative lag: model is behind actual signal (lag — needs improvement)
    Zero lag: prediction and actual are synchronous

    Args:
        y_true:         Actual CGM sequence
        y_pred:         Predicted CGM sequence
        max_lag_steps:  Maximum lag to search (in steps)
        step_minutes:   Minutes per step (default: 5)

    Returns:
        Dict with: optimal_lag_steps, optimal_lag_min, max_correlation,
                   mean_absolute_lag_min
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid] - y_true[valid].mean()
    yp = y_pred[valid] - y_pred[valid].mean()

    if len(yt) < 2 * max_lag_steps + 1:
        return {"optimal_lag_steps": 0, "optimal_lag_min": 0, "max_correlation": 0}

    # Normalised cross-correlation
    xcorr = signal.correlate(yt, yp, mode="full")
    lags = signal.correlation_lags(len(yt), len(yp), mode="full")

    # Focus on ±max_lag_steps
    center = len(xcorr) // 2
    lo = center - max_lag_steps
    hi = center + max_lag_steps + 1
    xcorr_clipped = xcorr[lo:hi]
    lags_clipped = lags[lo:hi]

    # Normalise to [-1, 1]
    norm = np.sqrt(np.sum(yt ** 2) * np.sum(yp ** 2))
    if norm > 0:
        xcorr_clipped = xcorr_clipped / norm

    optimal_idx = np.argmax(xcorr_clipped)
    optimal_lag = lags_clipped[optimal_idx]

    return {
        "optimal_lag_steps": int(optimal_lag),
        "optimal_lag_min": int(optimal_lag) * step_minutes,
        "max_correlation": float(xcorr_clipped[optimal_idx]),
        "mean_absolute_lag_min": float(np.abs(lags_clipped) @ np.abs(xcorr_clipped) / max(xcorr_clipped.sum(), 1e-9) * step_minutes),
    }


# ---------------------------------------------------------------------------
# Glycemia-Specific RMSE
# ---------------------------------------------------------------------------

def glycemia_specific_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Separate RMSE for hypoglycemia, normoglycemia, and hyperglycemia zones.

    Clinically, errors in the hypoglycemic range (< 70 mg/dL) are most
    dangerous. Errors in hyperglycemia (> 180 mg/dL) may lead to sub-optimal
    correction bolusing. Normal range errors have lower immediate risk.

    Returns:
        Dict with rmse_hypo, rmse_normal, rmse_hyper, n_hypo, n_normal, n_hyper
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]

    hypo_mask = yt < HYPO_L1
    normal_mask = (yt >= HYPO_L1) & (yt <= HYPER_L1)
    hyper_mask = yt > HYPER_L1

    def _rmse(mask):
        if mask.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))

    return {
        "rmse_hypo": _rmse(hypo_mask),
        "rmse_normal": _rmse(normal_mask),
        "rmse_hyper": _rmse(hyper_mask),
        "n_hypo": int(hypo_mask.sum()),
        "n_normal": int(normal_mask.sum()),
        "n_hyper": int(hyper_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Full Evaluation Report
# ---------------------------------------------------------------------------

@dataclass
class HorizonMetrics:
    """Evaluation metrics for a single prediction horizon."""
    horizon_min: int
    n_samples: int
    rmse: float
    mae: float
    mard: float
    clarke_ega: ClarkeEGAResult
    parkes: Dict[str, float]
    lag: Dict[str, float]
    glycemia_rmse: Dict[str, float]
    coverage_90: Optional[float] = None
    mean_interval_width_90: Optional[float] = None


def evaluate_predictions(
    y_true: np.ndarray,              # [N, n_horizons]
    y_pred: np.ndarray,              # [N, n_horizons] or [N, n_horizons, n_quantiles]
    prediction_horizons_min: List[int] = None,
) -> List[HorizonMetrics]:
    """
    Full clinical evaluation of multi-horizon glucose predictions.

    Args:
        y_true:                   Actual CGM values, shape [N, n_horizons]
        y_pred:                   Predictions, shape [N, n_horizons] or
                                  [N, n_horizons, n_quantiles]
        prediction_horizons_min:  Horizon labels in minutes [30, 60, 120]

    Returns:
        List of HorizonMetrics, one per horizon.
    """
    horizons_min = prediction_horizons_min or [30, 60, 120]
    n_horizons = y_true.shape[1]
    results = []

    for h_idx in range(n_horizons):
        yt = y_true[:, h_idx]
        horizon_min = horizons_min[h_idx] if h_idx < len(horizons_min) else (h_idx + 1) * 30

        if y_pred.ndim == 3:
            # Quantile predictions: extract median (index 1 for [0.1, 0.5, 0.9])
            yp_median = y_pred[:, h_idx, min(1, y_pred.shape[2] - 1)]
            yp_lower = y_pred[:, h_idx, 0]
            yp_upper = y_pred[:, h_idx, -1]
            cov = coverage(yt, yp_lower, yp_upper)
            miw = mean_interval_width(yp_lower, yp_upper)
        else:
            yp_median = y_pred[:, h_idx]
            cov = None
            miw = None

        valid = ~(np.isnan(yt) | np.isnan(yp_median))
        n_valid = int(valid.sum())

        metrics = HorizonMetrics(
            horizon_min=horizon_min,
            n_samples=n_valid,
            rmse=rmse(yt, yp_median),
            mae=mae(yt, yp_median),
            mard=mard(yt, yp_median),
            clarke_ega=clarke_error_grid(yt, yp_median),
            parkes=parkes_error_grid(yt, yp_median),
            lag=temporal_lag_analysis(yt, yp_median),
            glycemia_rmse=glycemia_specific_rmse(yt, yp_median),
            coverage_90=cov,
            mean_interval_width_90=miw,
        )
        results.append(metrics)

        logger.info(
            f"{horizon_min}-min: RMSE={metrics.rmse:.2f}, MAE={metrics.mae:.2f}, "
            f"MARD={metrics.mard:.1f}%, Clarke A+B={metrics.clarke_ega.zone_ab:.1f}%"
        )

    return results


def print_evaluation_table(results: List[HorizonMetrics]) -> str:
    """Format evaluation results as a readable table string."""
    lines = [
        "╔══════════════╦══════════╦═══════╦══════╦═══════╦═══════════════╗",
        "║ Horizon (min)║ N        ║  RMSE ║  MAE ║ MARD% ║ Clarke A+B %  ║",
        "╠══════════════╬══════════╬═══════╬══════╬═══════╬═══════════════╣",
    ]
    for m in results:
        lines.append(
            f"║ {m.horizon_min:>12} ║ {m.n_samples:>8,} ║ {m.rmse:>5.2f} ║ "
            f"{m.mae:>4.2f} ║ {m.mard:>5.1f} ║ {m.clarke_ega.zone_ab:>13.1f} ║"
        )
    lines.append(
        "╚══════════════╩══════════╩═══════╩══════╩═══════╩═══════════════╝"
    )
    return "\n".join(lines)
