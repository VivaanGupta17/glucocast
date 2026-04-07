"""
Clinical Safety Analysis for CGM Glucose Prediction.

For a glucose prediction algorithm to be used in a clinical or AID context,
it must be evaluated on safety-specific metrics beyond standard accuracy.

The key safety principle: errors are NOT symmetric.
  - FALSE NEGATIVE (missed hypoglycemia): Patient does not receive warning →
    hypoglycemia event → seizure, loss of consciousness, cardiac arrhythmia
  - FALSE POSITIVE (false alarm): Patient treats a non-existent hypo →
    hyperglycemia from overcorrection + alarm fatigue

Regulatory context:
  - FDA 21 CFR Part 882: CGM devices, including predictive features
  - IEC 62304: Medical device software lifecycle
  - ISO 15197:2013/2023: In vitro diagnostic test systems (accuracy standards)
  - FDA iCGM Special Controls (21 CFR 882.5860): Performance thresholds by zone

ADA Standards of Medical Care (2023) alert thresholds:
  - Level 1 hypoglycemia: < 70 mg/dL (requires prompt action)
  - Level 2 (severe): < 54 mg/dL (requires immediate action)
  - Level 1 hyperglycemia: > 180 mg/dL
  - Level 2 hyperglycemia: > 250 mg/dL

False alarm rate tolerance (from clinical literature):
  - Acceptable: ≤ 2 false alarms/day (alarms that disturb sleep: ≤ 1/night)
  - DiaMond study found alarm fatigue at > 3 alerts/day
  - Medtronic SmartGuard targets < 1 false alarm/night for PLGS
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert Thresholds
# ---------------------------------------------------------------------------

HYPO_L1_MGDL = 70    # Level 1 hypoglycemia
HYPO_L2_MGDL = 54    # Level 2 hypoglycemia (severe)
HYPER_L1_MGDL = 180  # Level 1 hyperglycemia
HYPER_L2_MGDL = 250  # Level 2 hyperglycemia (significant)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class HypoAlertMetrics:
    """
    Hypoglycemia alert detection performance.

    Computed over the full test set with a specific advance warning horizon.
    """
    threshold_mgdl: float
    alert_horizon_min: int      # How far ahead the alert is issued

    # Event-level detection
    n_true_hypo_events: int     # Total hypoglycemia events in test set
    n_detected: int             # Events with at least one correct alert
    n_missed: int               # Events with no alert (false negatives)
    n_false_alarms: int         # Alert issued but no subsequent hypo event

    # Rates
    sensitivity: float          # Recall = n_detected / n_true_hypo_events
    specificity: float          # True negative rate on non-hypo periods
    ppv: float                  # Precision = n_detected / (n_detected + n_false_alarms)
    false_alarm_rate_per_day: float

    # Timing
    median_lead_time_min: float  # Median time before event that alert fires
    p5_lead_time_min: float      # 5th percentile lead time
    p95_lead_time_min: float

    @property
    def f1(self) -> float:
        denom = self.sensitivity + self.ppv
        return 2 * self.sensitivity * self.ppv / denom if denom > 0 else 0.0


@dataclass
class PatientSafetyProfile:
    """Per-patient safety analysis."""
    patient_id: str
    n_cgm_readings: int
    n_hypo_l1_events: int    # < 70 mg/dL
    n_hypo_l2_events: int    # < 54 mg/dL
    n_hyper_l1_events: int   # > 180 mg/dL

    # Alert performance by horizon
    alert_30min: Optional[HypoAlertMetrics] = None
    alert_60min: Optional[HypoAlertMetrics] = None

    # Prediction quality in different glycemic ranges
    rmse_hypo: float = float("nan")
    rmse_normal: float = float("nan")
    rmse_hyper: float = float("nan")

    # Safety flags
    has_missed_severe_hypo: bool = False  # Any missed L2 hypo event
    has_high_false_alarm_rate: bool = False  # > 3 false alarms/day


# ---------------------------------------------------------------------------
# Event Detection
# ---------------------------------------------------------------------------

def detect_hypo_events(
    cgm: np.ndarray,
    timestamps: np.ndarray,
    threshold_mgdl: float = HYPO_L1_MGDL,
    min_duration_min: float = 15.0,
    refractory_period_min: float = 30.0,
) -> List[Dict]:
    """
    Detect discrete hypoglycemia events from a CGM time series.

    An "event" is defined as a continuous period where CGM < threshold,
    lasting at least min_duration_min. Brief recoveries within
    refractory_period_min are merged into a single event.

    Args:
        cgm:                    CGM values array (mg/dL)
        timestamps:             Corresponding timestamps (numpy datetime64 or similar)
        threshold_mgdl:         Hypoglycemia threshold (default: 70 mg/dL)
        min_duration_min:       Minimum duration to count as an event
        refractory_period_min:  Gap between two events (shorter gap → merge)

    Returns:
        List of event dicts: {start, end, nadir, duration_min, nadir_value}
    """
    below_threshold = cgm < threshold_mgdl
    # Replace NaN with False
    below_threshold = np.where(np.isnan(cgm), False, below_threshold)

    # Find contiguous runs below threshold
    labeled_array, n_events = label(below_threshold)
    events = []

    for event_id in range(1, n_events + 1):
        event_indices = np.where(labeled_array == event_id)[0]
        if len(event_indices) == 0:
            continue

        # Duration (5-min CGM intervals)
        duration_min = len(event_indices) * 5.0
        if duration_min < min_duration_min:
            continue

        nadir_idx = event_indices[np.nanargmin(cgm[event_indices])]
        events.append({
            "start": timestamps[event_indices[0]],
            "end": timestamps[event_indices[-1]],
            "start_idx": int(event_indices[0]),
            "end_idx": int(event_indices[-1]),
            "nadir_value": float(cgm[nadir_idx]),
            "nadir_idx": int(nadir_idx),
            "duration_min": duration_min,
        })

    # Merge events within refractory period
    events = _merge_nearby_events(events, refractory_period_min, timestamps)
    return events


def _merge_nearby_events(
    events: List[Dict],
    refractory_min: float,
    timestamps: np.ndarray,
) -> List[Dict]:
    """Merge hypoglycemia events separated by less than refractory_min."""
    if len(events) < 2:
        return events

    merged = [events[0]]
    for ev in events[1:]:
        prev = merged[-1]
        # Compute gap between events
        try:
            gap_min = (ev["start"] - prev["end"]).total_seconds() / 60
        except (AttributeError, TypeError):
            # numpy timedelta
            gap_min = float(ev["start"] - prev["end"]) / 1e9 / 60

        if gap_min <= refractory_min:
            # Merge
            merged[-1] = {
                "start": prev["start"],
                "end": ev["end"],
                "start_idx": prev["start_idx"],
                "end_idx": ev["end_idx"],
                "nadir_value": min(prev["nadir_value"], ev["nadir_value"]),
                "nadir_idx": (prev["nadir_idx"] if prev["nadir_value"] <= ev["nadir_value"]
                              else ev["nadir_idx"]),
                "duration_min": prev["duration_min"] + ev["duration_min"] + gap_min,
            }
        else:
            merged.append(ev)
    return merged


# ---------------------------------------------------------------------------
# Alert Performance Evaluation
# ---------------------------------------------------------------------------

def evaluate_hypo_alerts(
    cgm_true: np.ndarray,              # [N] actual CGM values
    cgm_pred: np.ndarray,              # [N] predicted CGM at horizon ahead
    timestamps: np.ndarray,            # [N] timestamps
    threshold_mgdl: float = HYPO_L1_MGDL,
    alert_horizon_min: int = 30,
    min_event_duration_min: float = 15.0,
    alarm_suppression_min: float = 60.0,
) -> HypoAlertMetrics:
    """
    Evaluate hypoglycemia alert detection performance.

    The model fires an alert at time t if it predicts glucose < threshold
    at time t + horizon. We then check whether a true hypoglycemia event
    actually occurs within [t, t + horizon + 30min] (grace window).

    Alert suppression: After an alert fires, suppress additional alerts for
    alarm_suppression_min minutes (to avoid alert flood during a single event).

    Args:
        cgm_true:             Actual CGM values
        cgm_pred:             Predicted CGM values at alert_horizon_min ahead
        timestamps:           Timestamps for CGM readings
        threshold_mgdl:       Alert trigger threshold (predicted glucose < this)
        alert_horizon_min:    Prediction lead time (30 or 60 min)
        min_event_duration_min: Min duration to count as hypo event
        alarm_suppression_min: Refractory period for alert suppression

    Returns:
        HypoAlertMetrics with complete safety analysis
    """
    n = len(cgm_true)
    # 5-min CGM intervals
    horizon_steps = alert_horizon_min // 5
    step_min = 5

    # Detect true hypoglycemia events in actual CGM
    true_events = detect_hypo_events(
        cgm_true, timestamps, threshold_mgdl, min_event_duration_min
    )

    # Generate alerts: fired when predicted glucose < threshold
    alert_fired = cgm_pred < threshold_mgdl
    alert_fired = np.where(np.isnan(cgm_pred), False, alert_fired)

    # Apply alert suppression (refractory period)
    suppressed_alerts = np.zeros(n, dtype=bool)
    suppression_counter = 0
    for i in range(n):
        if suppression_counter > 0:
            suppression_counter -= 1
            suppressed_alerts[i] = False
        elif alert_fired[i]:
            suppressed_alerts[i] = True
            suppression_counter = int(alarm_suppression_min / step_min)

    # Match alerts to events: was there a true event within the alert window?
    grace_steps = (alert_horizon_min + 30) // step_min

    detected_events = set()
    false_alarm_indices = []
    lead_times = []

    for alert_idx in np.where(suppressed_alerts)[0]:
        alert_end_idx = min(alert_idx + grace_steps, n - 1)
        # Is there a true hypo event in the window?
        window_true = cgm_true[alert_idx: alert_end_idx]
        event_in_window = np.any(window_true < threshold_mgdl)

        if event_in_window:
            # Find which true event this matched
            for ev_idx, ev in enumerate(true_events):
                if (ev["start_idx"] >= alert_idx and
                        ev["start_idx"] <= alert_end_idx):
                    detected_events.add(ev_idx)
                    # Lead time: how early did we alert before the event?
                    lead_min = (ev["start_idx"] - alert_idx) * step_min
                    lead_times.append(max(0, lead_min))
                    break
        else:
            false_alarm_indices.append(alert_idx)

    # Sensitivity and specificity
    n_true_events = len(true_events)
    n_detected = len(detected_events)
    n_missed = n_true_events - n_detected
    n_false_alarms = len(false_alarm_indices)

    sensitivity = n_detected / max(n_true_events, 1)

    # Specificity: fraction of non-hypo 5-min periods correctly not alarmed
    non_hypo_periods = np.sum(cgm_true >= threshold_mgdl)
    true_negatives = non_hypo_periods - n_false_alarms
    specificity = true_negatives / max(non_hypo_periods, 1)
    specificity = max(0.0, min(1.0, specificity))

    ppv = n_detected / max(n_detected + n_false_alarms, 1)

    # False alarm rate per day (total recording duration / n_false_alarms)
    total_days = n * step_min / (60 * 24)
    false_alarm_rate = n_false_alarms / max(total_days, 1e-9)

    lead_arr = np.array(lead_times) if lead_times else np.array([0.0])

    return HypoAlertMetrics(
        threshold_mgdl=threshold_mgdl,
        alert_horizon_min=alert_horizon_min,
        n_true_hypo_events=n_true_events,
        n_detected=n_detected,
        n_missed=n_missed,
        n_false_alarms=n_false_alarms,
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv,
        false_alarm_rate_per_day=false_alarm_rate,
        median_lead_time_min=float(np.median(lead_arr)),
        p5_lead_time_min=float(np.percentile(lead_arr, 5)),
        p95_lead_time_min=float(np.percentile(lead_arr, 95)),
    )


# ---------------------------------------------------------------------------
# Time-to-Alert Analysis
# ---------------------------------------------------------------------------

def time_to_alert_analysis(
    cgm_true: np.ndarray,
    cgm_pred_30: np.ndarray,
    cgm_pred_60: np.ndarray,
    timestamps: np.ndarray,
    threshold_mgdl: float = HYPO_L1_MGDL,
) -> Dict[str, float]:
    """
    Analyse how far in advance the model detects approaching hypoglycemia.

    Compares the first alert time from 30-min and 60-min predictions
    vs. the actual hypoglycemia onset. More lead time = more time for action.

    Args:
        cgm_true:      Actual CGM
        cgm_pred_30:   30-min ahead predictions
        cgm_pred_60:   60-min ahead predictions
        timestamps:    Timestamps
        threshold_mgdl: Alert threshold

    Returns:
        Dict with median/mean/std of advance warning time for each horizon
    """
    true_events = detect_hypo_events(cgm_true, timestamps, threshold_mgdl)
    step_min = 5

    results = {}
    for horizon_min, preds in [(30, cgm_pred_30), (60, cgm_pred_60)]:
        lead_times = []
        for ev in true_events:
            event_start = ev["start_idx"]
            # Search backward from event start for first alert
            look_back_steps = int((horizon_min + 30) / step_min)
            search_start = max(0, event_start - look_back_steps)
            alert_region = preds[search_start: event_start + 1]
            alert_steps = np.where(alert_region < threshold_mgdl)[0]
            if len(alert_steps) > 0:
                first_alert_offset = len(alert_region) - 1 - alert_steps[-1]
                lead_times.append(first_alert_offset * step_min)
            else:
                lead_times.append(0)  # No alert before event

        if lead_times:
            arr = np.array(lead_times)
            results[f"horizon_{horizon_min}min"] = {
                "median_advance_warning_min": float(np.median(arr)),
                "mean_advance_warning_min": float(np.mean(arr)),
                "std_advance_warning_min": float(np.std(arr)),
                "fraction_with_warning": float((arr > 0).mean()),
                "fraction_with_30min_warning": float((arr >= 30).mean()),
            }

    return results


# ---------------------------------------------------------------------------
# Per-Patient Safety Report
# ---------------------------------------------------------------------------

def patient_safety_analysis(
    patient_id: str,
    cgm_true: np.ndarray,
    cgm_pred_30: np.ndarray,
    cgm_pred_60: np.ndarray,
    timestamps: np.ndarray,
) -> PatientSafetyProfile:
    """
    Comprehensive safety analysis for a single patient.

    Args:
        patient_id:    Patient identifier
        cgm_true:      Actual CGM values [N]
        cgm_pred_30:   30-min ahead predictions [N]
        cgm_pred_60:   60-min ahead predictions [N]
        timestamps:    Timestamps [N]

    Returns:
        PatientSafetyProfile with complete safety metrics
    """
    from src.evaluation.glucose_metrics import glycemia_specific_rmse

    valid = ~np.isnan(cgm_true)

    # Event counts
    hypo_l1_events = detect_hypo_events(cgm_true, timestamps, HYPO_L1_MGDL)
    hypo_l2_events = detect_hypo_events(cgm_true, timestamps, HYPO_L2_MGDL)
    hyper_l1_events = detect_hypo_events(
        -cgm_true, timestamps, -HYPER_L1_MGDL
    )   # Invert to reuse function

    # Alert performance
    alert_30 = evaluate_hypo_alerts(cgm_true, cgm_pred_30, timestamps,
                                    HYPO_L1_MGDL, 30)
    alert_60 = evaluate_hypo_alerts(cgm_true, cgm_pred_60, timestamps,
                                    HYPO_L1_MGDL, 60)

    # Glycemia-specific RMSE (30-min horizon)
    gv_rmse = glycemia_specific_rmse(cgm_true[valid], cgm_pred_30[valid])

    # Safety flags
    severe_hypo_detected = all(
        any(alert_30.n_detected > 0 or alert_60.n_detected > 0
            for _ in [1])  # simplified
        for _ in hypo_l2_events
    )
    has_missed_severe = len(hypo_l2_events) > 0 and alert_30.n_missed > 0

    profile = PatientSafetyProfile(
        patient_id=patient_id,
        n_cgm_readings=int(valid.sum()),
        n_hypo_l1_events=len(hypo_l1_events),
        n_hypo_l2_events=len(hypo_l2_events),
        n_hyper_l1_events=len(hyper_l1_events),
        alert_30min=alert_30,
        alert_60min=alert_60,
        rmse_hypo=gv_rmse.get("rmse_hypo", float("nan")),
        rmse_normal=gv_rmse.get("rmse_normal", float("nan")),
        rmse_hyper=gv_rmse.get("rmse_hyper", float("nan")),
        has_missed_severe_hypo=has_missed_severe,
        has_high_false_alarm_rate=alert_30.false_alarm_rate_per_day > 3.0,
    )

    # Log safety warnings
    if profile.has_missed_severe_hypo:
        logger.warning(
            f"Patient {patient_id}: MISSED SEVERE HYPO EVENT(S)! "
            f"({profile.n_hypo_l2_events} L2 events, {alert_30.n_missed} missed)"
        )
    if profile.has_high_false_alarm_rate:
        logger.warning(
            f"Patient {patient_id}: High false alarm rate "
            f"({alert_30.false_alarm_rate_per_day:.1f}/day)"
        )

    return profile


def print_safety_report(profile: PatientSafetyProfile) -> str:
    """Format a patient safety profile as a readable report."""
    a30 = profile.alert_30min
    a60 = profile.alert_60min

    lines = [
        f"═══ Patient {profile.patient_id} Safety Report ═══",
        f"CGM readings:           {profile.n_cgm_readings:,}",
        f"Hypo L1 events (<70):   {profile.n_hypo_l1_events}",
        f"Hypo L2 events (<54):   {profile.n_hypo_l2_events}",
        f"Hyper L1 events (>180): {profile.n_hyper_l1_events}",
        "",
        "─── 30-min Alert Performance ───",
    ]

    if a30:
        lines += [
            f"  Sensitivity:       {a30.sensitivity:.1%}",
            f"  Specificity:       {a30.specificity:.1%}",
            f"  PPV (Precision):   {a30.ppv:.1%}",
            f"  F1 Score:          {a30.f1:.3f}",
            f"  False alarms/day:  {a30.false_alarm_rate_per_day:.1f}",
            f"  Median lead time:  {a30.median_lead_time_min:.0f} min",
        ]

    if a60:
        lines += [
            "",
            "─── 60-min Alert Performance ───",
            f"  Sensitivity:       {a60.sensitivity:.1%}",
            f"  Specificity:       {a60.specificity:.1%}",
            f"  False alarms/day:  {a60.false_alarm_rate_per_day:.1f}",
            f"  Median lead time:  {a60.median_lead_time_min:.0f} min",
        ]

    lines += [
        "",
        "─── Glycemia-Specific RMSE (30-min) ───",
        f"  Hypo (<70 mg/dL):  {profile.rmse_hypo:.2f} mg/dL",
        f"  Normal:            {profile.rmse_normal:.2f} mg/dL",
        f"  Hyper (>180):      {profile.rmse_hyper:.2f} mg/dL",
    ]

    if profile.has_missed_severe_hypo:
        lines += ["", "⚠ WARNING: Missed severe hypoglycemia event(s)!"]
    if profile.has_high_false_alarm_rate:
        lines += ["", "⚠ WARNING: High false alarm rate (> 3/day)"]

    return "\n".join(lines)
