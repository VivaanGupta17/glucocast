"""
CGM Signal Preprocessing.

CGM signals are noisy, gappy, and occasionally contain physiologically
impossible values. This module provides a pipeline to:

  1. Detect and classify gaps (sensor warm-up, transmission loss, calibration)
  2. Remove outliers (physiologically impossible glucose velocities)
  3. Impute short gaps (≤15 min) using forward fill + linear interpolation
  4. Detect and handle compression artifacts and sensor noise
  5. Normalise to patient-specific z-score statistics

CGM Signal Characteristics (Abbott FreeStyle Libre / Medtronic Enlite):
  - Measurement range: 39–401 mg/dL (reports 39 for LOW, 401 for HIGH)
  - Sampling interval: 5 min (Libre/Medtronic), 5 min (Dexcom G6/G7)
  - Accuracy: MARD ~9% for Libre 3, ~8% for Dexcom G7 vs. fingerstick
  - Compression artifacts: falsely low readings when sensor is compressed
    during sleep (common 2-4 AM)
  - Sensor warmup: 60 min (Dexcom G6) / 60 min (Libre 3) after insertion
  - Calibration drift: accuracy degrades in last 24h of 14-day Libre wear

Physiological bounds used for outlier detection:
  - Rate of change: ±4 mg/dL/min sustained (Clarke 1987)
    - Maximum physiological: ~5 mg/dL/min during rapid insulin action
    - CGM sensors typically clip at 3 mg/dL/min trend arrows
  - Absolute range: 20–600 mg/dL (beyond sensor range suggests device error)
  - Delta between consecutive readings: >70 mg/dL in 5 min is physiologically
    impossible without calibration error
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CGM_INTERVAL_MIN = 5          # Standard CGM sampling interval
MAX_ROC_MGDL_MIN = 4.0        # Maximum physiologically plausible rate of change
MAX_DELTA_5MIN = 50.0         # Maximum plausible 5-min delta (mg/dL)
ABS_GLUCOSE_MIN = 20.0        # Below this: sensor error
ABS_GLUCOSE_MAX = 600.0       # Above this: sensor error (not HI reading)
SHORT_GAP_THRESHOLD_MIN = 15  # Gaps ≤ 15 min: safe to interpolate
MEDIUM_GAP_THRESHOLD_MIN = 45 # Gaps 15-45 min: forward fill only
LONG_GAP_THRESHOLD_MIN = 120  # Gaps > 120 min: do not impute (mark as invalid segment)


class GapType(Enum):
    SHORT = "short"        # ≤ 15 min — cubic spline interpolation
    MEDIUM = "medium"      # 15-45 min — forward fill
    LONG = "long"          # 45-120 min — forward fill with uncertainty flag
    SENSOR_CHANGE = "sensor_change"  # > 120 min — likely sensor replacement


@dataclass
class GapInfo:
    start: pd.Timestamp
    end: pd.Timestamp
    duration_min: float
    gap_type: GapType
    n_missing_steps: int


@dataclass
class OutlierInfo:
    timestamp: pd.Timestamp
    value: float
    reason: str
    replaced_with: Optional[float] = None


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class CGMPreprocessor:
    """
    CGM signal preprocessing pipeline.

    Steps:
        1. Resample to regular 5-min grid
        2. Clip physiologically impossible values (LOW/HIGH sensor flags)
        3. Detect outliers (velocity-based + Hampel filter)
        4. Detect and classify gaps
        5. Impute short/medium gaps
        6. Flag non-imputed segments for downstream handling

    Usage:
        preprocessor = CGMPreprocessor()
        clean_cgm = preprocessor.process(raw_cgm_series)
        report = preprocessor.last_report  # Gap/outlier summary
    """

    def __init__(
        self,
        max_roc: float = MAX_ROC_MGDL_MIN,
        max_delta_5min: float = MAX_DELTA_5MIN,
        short_gap_min: float = SHORT_GAP_THRESHOLD_MIN,
        medium_gap_min: float = MEDIUM_GAP_THRESHOLD_MIN,
        long_gap_min: float = LONG_GAP_THRESHOLD_MIN,
        hampel_window: int = 5,    # Steps on each side for Hampel filter
        hampel_threshold: float = 3.0,  # MAD multiplier for Hampel outlier
    ):
        self.max_roc = max_roc
        self.max_delta_5min = max_delta_5min
        self.short_gap_min = short_gap_min
        self.medium_gap_min = medium_gap_min
        self.long_gap_min = long_gap_min
        self.hampel_window = hampel_window
        self.hampel_threshold = hampel_threshold

        self.last_report: Dict = {}

    def process(self, cgm: pd.Series) -> pd.Series:
        """
        Full preprocessing pipeline.

        Args:
            cgm: Raw CGM series with DatetimeIndex, values in mg/dL.
                 May have irregular timestamps (Libre uploads) or 5-min grid.

        Returns:
            Cleaned CGM series on 5-min grid. NaN where data is genuinely
            missing and cannot be safely imputed.
        """
        cgm = cgm.copy()
        cgm.name = "cgm"

        # Step 1: Ensure 5-min regular grid
        cgm = self._resample_to_grid(cgm)
        n_original = (~cgm.isna()).sum()

        # Step 2: Clip sensor limits (39 = LOW, 401 = HIGH → NaN)
        cgm = self._clip_sensor_limits(cgm)

        # Step 3: Absolute range filter
        out_of_range = (cgm < ABS_GLUCOSE_MIN) | (cgm > ABS_GLUCOSE_MAX)
        n_oor = out_of_range.sum()
        if n_oor > 0:
            logger.debug(f"Removed {n_oor} out-of-range CGM values")
            cgm[out_of_range] = np.nan

        # Step 4: Velocity-based outlier detection
        cgm, outlier_info = self._remove_velocity_outliers(cgm)

        # Step 5: Hampel filter (robust outlier removal)
        cgm, hampel_info = self._hampel_filter(cgm)

        # Step 6: Gap detection and classification
        gaps = self._detect_gaps(cgm)

        # Step 7: Imputation
        cgm = self._impute_gaps(cgm, gaps)

        # Step 8: Smooth sensor noise (optional, mild Savitzky-Golay)
        cgm = self._mild_smoothing(cgm)

        self.last_report = {
            "n_original_readings": n_original,
            "n_out_of_range": int(n_oor),
            "n_velocity_outliers": len(outlier_info),
            "n_hampel_outliers": len(hampel_info),
            "gaps": gaps,
            "n_short_gaps": sum(1 for g in gaps if g.gap_type == GapType.SHORT),
            "n_medium_gaps": sum(1 for g in gaps if g.gap_type == GapType.MEDIUM),
            "n_long_gaps": sum(1 for g in gaps if g.gap_type == GapType.LONG),
            "n_sensor_changes": sum(1 for g in gaps if g.gap_type == GapType.SENSOR_CHANGE),
            "final_missing_pct": cgm.isna().mean() * 100,
        }

        logger.info(
            f"Preprocessing: {n_original} → {(~cgm.isna()).sum()} readings "
            f"({len(gaps)} gaps, {self.last_report['final_missing_pct']:.1f}% missing)"
        )

        return cgm

    def _resample_to_grid(self, cgm: pd.Series) -> pd.Series:
        """Resample to strict 5-minute DatetimeIndex."""
        if not isinstance(cgm.index, pd.DatetimeIndex):
            raise ValueError("CGM series must have DatetimeIndex")
        if len(cgm) == 0:
            return cgm
        start = cgm.index.min().floor("5T")
        end = cgm.index.max().ceil("5T")
        grid = pd.date_range(start, end, freq="5T")
        # For each grid point, take the nearest reading within ±2.5 min
        cgm_resampled = cgm.reindex(grid, method="nearest", tolerance=pd.Timedelta("2min30s"))
        return cgm_resampled

    def _clip_sensor_limits(self, cgm: pd.Series) -> pd.Series:
        """
        Convert sensor LOW/HIGH flags to NaN.

        Abbott Libre reports 39 mg/dL for readings below sensor range.
        Medtronic Enlite reports 39 mg/dL (LOW) and 401 mg/dL (HIGH).
        These exact values indicate sensor saturation, not true glucose.
        """
        cgm = cgm.copy()
        cgm[cgm == 39.0] = np.nan   # Sensor LOW flag
        cgm[cgm == 401.0] = np.nan  # Sensor HIGH flag
        return cgm

    def _remove_velocity_outliers(
        self, cgm: pd.Series
    ) -> Tuple[pd.Series, List[OutlierInfo]]:
        """
        Remove CGM values with physiologically impossible rate of change.

        A CGM reading is flagged as an outlier if it requires an implausibly
        rapid glucose change FROM the preceding AND subsequent readings.
        Single-sided checks miss compression artifacts followed by recovery.

        Algorithm:
          For each timestep t:
            roc_prev = (cgm[t] - cgm[t-1]) / 5 min
            roc_next = (cgm[t+1] - cgm[t]) / 5 min
            If |roc_prev| > max_roc AND |roc_next| > max_roc AND
               sign(roc_prev) != sign(roc_next):  → spike/compression artifact
        """
        cgm = cgm.copy()
        outliers = []
        values = cgm.values.copy()
        n = len(values)

        for i in range(1, n - 1):
            if np.isnan(values[i]):
                continue
            prev_val = values[i - 1]
            next_val = values[i + 1]
            if np.isnan(prev_val) or np.isnan(next_val):
                continue

            roc_prev = (values[i] - prev_val) / CGM_INTERVAL_MIN
            roc_next = (next_val - values[i]) / CGM_INTERVAL_MIN

            is_spike = (abs(roc_prev) > self.max_roc and
                       abs(roc_next) > self.max_roc and
                       np.sign(roc_prev) != np.sign(roc_next))

            if is_spike:
                outliers.append(OutlierInfo(
                    timestamp=cgm.index[i],
                    value=float(values[i]),
                    reason="velocity_spike",
                ))
                values[i] = np.nan

            # Simple single-side delta check
            elif abs(values[i] - prev_val) > self.max_delta_5min:
                outliers.append(OutlierInfo(
                    timestamp=cgm.index[i],
                    value=float(values[i]),
                    reason="excessive_delta",
                ))
                values[i] = np.nan

        cgm.iloc[:] = values
        return cgm, outliers

    def _hampel_filter(
        self, cgm: pd.Series
    ) -> Tuple[pd.Series, List[OutlierInfo]]:
        """
        Hampel identifier: robust outlier detection using local MAD.

        For each point x_t, compute the median and MAD over a window of
        ±k steps. Flag x_t as an outlier if:
          |x_t - median| > threshold * MAD / 0.6745

        0.6745 is the scaling factor to make MAD a consistent estimator
        of the normal distribution standard deviation.

        Effective for CGM compression artifacts and calibration spikes.
        """
        cgm = cgm.copy()
        outliers = []
        values = cgm.values.copy()
        n = len(values)
        k = self.hampel_window

        for i in range(k, n - k):
            if np.isnan(values[i]):
                continue
            window = values[max(0, i - k): i + k + 1]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) < 3:
                continue
            med = np.median(valid_window)
            mad = np.median(np.abs(valid_window - med))
            sigma_hat = mad / 0.6745
            if sigma_hat > 0 and abs(values[i] - med) > self.hampel_threshold * sigma_hat:
                outliers.append(OutlierInfo(
                    timestamp=cgm.index[i],
                    value=float(values[i]),
                    reason="hampel",
                    replaced_with=float(med),
                ))
                values[i] = np.nan

        cgm.iloc[:] = values
        return cgm, outliers

    def _detect_gaps(self, cgm: pd.Series) -> List[GapInfo]:
        """
        Identify all NaN gaps in the CGM series and classify them.

        A "gap" is a contiguous run of NaN values. We classify gaps
        by duration because different imputation strategies are appropriate:
          - SHORT (≤15 min): Sensor transmission hiccup. Safe to interpolate.
          - MEDIUM (≤45 min): Activity/Bluetooth outage. Forward fill only.
          - LONG (≤120 min): Sensor calibration / brief removal.
          - SENSOR_CHANGE (>120 min): New sensor insertion. Never impute.
        """
        gaps = []
        n = len(cgm)
        i = 0

        while i < n:
            if pd.isna(cgm.iloc[i]):
                gap_start_idx = i
                while i < n and pd.isna(cgm.iloc[i]):
                    i += 1
                gap_end_idx = i - 1
                gap_start = cgm.index[gap_start_idx]
                gap_end = cgm.index[gap_end_idx]
                duration_min = (gap_end - gap_start).total_seconds() / 60 + CGM_INTERVAL_MIN
                n_missing = gap_end_idx - gap_start_idx + 1

                if duration_min <= self.short_gap_min:
                    gap_type = GapType.SHORT
                elif duration_min <= self.medium_gap_min:
                    gap_type = GapType.MEDIUM
                elif duration_min <= self.long_gap_min:
                    gap_type = GapType.LONG
                else:
                    gap_type = GapType.SENSOR_CHANGE

                gaps.append(GapInfo(
                    start=gap_start,
                    end=gap_end,
                    duration_min=duration_min,
                    gap_type=gap_type,
                    n_missing_steps=n_missing,
                ))
            else:
                i += 1

        return gaps

    def _impute_gaps(self, cgm: pd.Series, gaps: List[GapInfo]) -> pd.Series:
        """
        Impute CGM gaps based on their type.

        SHORT gaps: Cubic spline through the surrounding ±3 valid readings.
                    Cubic spline preserves the smooth kinetics of CGM signal
                    better than linear interpolation.

        MEDIUM gaps: Forward fill from last known value.
                     Linear extrapolation would diverge; simple carry-forward
                     is more conservative and avoids false predictions.

        LONG / SENSOR_CHANGE gaps: Leave as NaN. These will be handled
                                   by the training pipeline (window exclusion).
        """
        cgm = cgm.copy()

        for gap in gaps:
            if gap.gap_type == GapType.SHORT:
                # Find flanking valid indices for spline
                start_pos = cgm.index.get_loc(gap.start)
                end_pos = cgm.index.get_loc(gap.end)
                context_start = max(0, start_pos - 3)
                context_end = min(len(cgm) - 1, end_pos + 3)

                context = cgm.iloc[context_start: context_end + 1].dropna()
                if len(context) >= 2:
                    # Fit cubic spline on valid context
                    t_valid = np.array([(ts - context.index[0]).total_seconds()
                                        for ts in context.index])
                    cs = CubicSpline(t_valid, context.values, extrapolate=False)
                    # Evaluate at gap timesteps
                    for pos in range(start_pos, end_pos + 1):
                        t_gap = (cgm.index[pos] - context.index[0]).total_seconds()
                        if 0 <= t_gap <= t_valid[-1]:
                            interp_val = float(cs(t_gap))
                            # Clip to physiological range
                            cgm.iloc[pos] = np.clip(interp_val, ABS_GLUCOSE_MIN, ABS_GLUCOSE_MAX)

            elif gap.gap_type in (GapType.MEDIUM, GapType.LONG):
                # Forward fill
                start_pos = cgm.index.get_loc(gap.start)
                if start_pos > 0:
                    last_valid = cgm.iloc[start_pos - 1]
                    if not np.isnan(last_valid):
                        end_pos = cgm.index.get_loc(gap.end)
                        cgm.iloc[start_pos: end_pos + 1] = last_valid

            # SENSOR_CHANGE: leave as NaN

        return cgm

    def _mild_smoothing(self, cgm: pd.Series) -> pd.Series:
        """
        Apply mild Savitzky-Golay smoothing to reduce sensor noise.

        Only applied to non-NaN segments. Window=5 (25 min) with polynomial
        degree 2 removes high-frequency noise while preserving physiological
        features (meal peaks, trend direction).

        Note: Not applied during inference (prediction window) — only
        historical data is smoothed. Future predictions use raw CGM.
        """
        cgm = cgm.copy()
        values = cgm.values.copy()
        valid_mask = ~np.isnan(values)

        # Apply smoothing only within continuous valid segments
        segments = self._find_valid_segments(valid_mask)
        for seg_start, seg_end in segments:
            if seg_end - seg_start < 5:   # Need at least window_length points
                continue
            seg = values[seg_start:seg_end + 1]
            # window_length must be odd and ≤ segment length
            win = min(5, len(seg) if len(seg) % 2 == 1 else len(seg) - 1)
            if win < 3:
                continue
            smoothed = sp_signal.savgol_filter(seg, window_length=win, polyorder=2)
            values[seg_start:seg_end + 1] = smoothed

        cgm.iloc[:] = values
        return cgm

    def _find_valid_segments(
        self, valid_mask: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Find contiguous segments of valid (non-NaN) values."""
        segments = []
        in_segment = False
        seg_start = 0
        for i, v in enumerate(valid_mask):
            if v and not in_segment:
                seg_start = i
                in_segment = True
            elif not v and in_segment:
                segments.append((seg_start, i - 1))
                in_segment = False
        if in_segment:
            segments.append((seg_start, len(valid_mask) - 1))
        return segments

    def normalise_patient(
        self, cgm: pd.Series
    ) -> Tuple[pd.Series, Tuple[float, float]]:
        """
        Patient-specific z-score normalisation.

        Returns normalised series and (mean, std) tuple for denormalisation.
        Always compute stats on the TRAINING set, apply to val/test.
        """
        mu = float(cgm.dropna().mean())
        sigma = float(cgm.dropna().std())
        if sigma < 1e-6:
            sigma = 1.0
        normalised = (cgm - mu) / sigma
        return normalised, (mu, sigma)

    @staticmethod
    def denormalise(normalised: pd.Series, mu: float, sigma: float) -> pd.Series:
        """Inverse of z-score normalisation."""
        return normalised * sigma + mu


SENSOR_WARMUP_MINUTES = 120  # first 2 hours after sensor insertion are unreliable


def skip_sensor_warmup(cgm_df, sensor_start_time=None, warmup_min: int = SENSOR_WARMUP_MINUTES):
    """
    Drop readings from the sensor warmup period.
    During warmup, glucose readings are often inaccurate due to tissue equilibration.
    """
    if sensor_start_time is None:
        sensor_start_time = cgm_df["timestamp"].min()

    cutoff = sensor_start_time + pd.Timedelta(minutes=warmup_min)
    filtered = cgm_df[cgm_df["timestamp"] >= cutoff].copy()
    n_dropped = len(cgm_df) - len(filtered)
    if n_dropped > 0:
        import logging
        logging.getLogger("glucocast.preprocessing").info(
            "skipped %d warmup readings (first %d min)", n_dropped, warmup_min
        )
    return filtered
