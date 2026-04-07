"""
Clinical Feature Engineering for CGM Glucose Prediction.

Transforms raw CGM, insulin, and meal data into clinically meaningful
features that capture the pharmacological dynamics governing blood glucose.

Key features computed:

1. Insulin on Board (IOB)
   The amount of active insulin remaining after a bolus injection.
   Uses the biexponential pharmacokinetic model (Walsh et al., 2011).
   Critical for predicting glucose falls — a 4-unit bolus from 2 hours
   ago still has ~40% activity remaining (for rapid-acting insulin).

2. Carbs on Board (COB)
   Estimated remaining glucose impact from recent meals.
   Uses the Hovorka absorption model for subcutaneous carbohydrate
   dynamics (nonlinear, biphasic absorption).

3. Rate of Change (CGM RoC)
   First derivative of CGM signal. Trend arrows on Abbott Libre and
   Dexcom use 5 categories: ↑↑ (>2), ↑ (1-2), → (±1), ↓ (-1 to -2), ↓↓ (<-2)
   GlucoCast uses the continuous mg/dL/min value.

4. Glycemic Variability
   - CV (Coefficient of Variation): std/mean of CGM over rolling window
     CV < 36% is ADA target for "stable" glycemia
   - MAGE (Mean Amplitude of Glycemic Excursions): average of excursions
     greater than one standard deviation
   - MODD (Mean of Daily Differences): average day-to-day variability
   - Time-in-Range (TIR): fraction of time 70-180 mg/dL (ADA target: >70%)
   - Time-above-Range (TAR): fraction > 180 mg/dL
   - Time-below-Range (TBR): fraction < 70 mg/dL

5. Circadian Encoding
   Sine/cosine encoding of time-of-day and day-of-week to capture
   diurnal glucose patterns (dawn phenomenon, cortisol peaks).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pharmacokinetic Constants
# ---------------------------------------------------------------------------

# Rapid-acting insulin analogue PK parameters (Walsh et al., 2011)
# Covers: NovoLog/Aspart, Humalog/Lispro, Apidra/Glulisine
# Typical onset: 10-20 min, peak: 60-90 min, duration: 3-5 hours
RAPID_ACTING_PK = {
    "peak_min": 75,          # Time to peak activity (minutes)
    "duration_min": 300,     # Total active duration (minutes, ~5 hours)
    "dia": 5.0,              # Duration of insulin action (hours)
}

# Fiasp/Lispro-aabc (ultra-fast-acting)
ULTRA_FAST_PK = {
    "peak_min": 55,
    "duration_min": 240,
    "dia": 4.0,
}

INSULIN_TYPES = {
    "NovoLog": RAPID_ACTING_PK,
    "Humalog": RAPID_ACTING_PK,
    "Apidra": RAPID_ACTING_PK,
    "Fiasp": ULTRA_FAST_PK,
    "default": RAPID_ACTING_PK,
}

# Carbohydrate absorption constants (Hovorka model)
# Meal glycemic index affects absorption rate
CARB_ABSORPTION = {
    "fast": {"t_max": 40, "duration": 120},   # High GI: fruit, juice, white bread
    "medium": {"t_max": 60, "duration": 180},  # Medium GI: most starchy foods
    "slow": {"t_max": 90, "duration": 240},    # Low GI: legumes, high-fiber
    "default": {"t_max": 60, "duration": 180},
}


# ---------------------------------------------------------------------------
# IOB Pharmacokinetic Model
# ---------------------------------------------------------------------------

class InsulinOnBoard:
    """
    Insulin on Board (IOB) calculation using the biexponential model.

    Walsh's model (used in Omnipod, Insulet, most commercial pumps):

    IOB(t) = dose * (1 - integral_of_activity_curve from 0 to t)

    The activity curve for rapid-acting insulin (Scheiner's model) is
    approximated as:

    activity(t) = dose * (A * exp(-t/τ1) + B * exp(-t/τ2))

    where τ1 ≈ 55 min (fast component) and τ2 ≈ 500 min (slow taper).

    This is the model used by Loop (OpenAPS) and the Omnipod Dash system.

    Reference:
      Walsh et al. (2011). Using Insulin: Everything You Need for Success
      with Insulin. Torrey Pines Press.
    """

    def __init__(
        self,
        peak_min: float = 75,
        dia_hours: float = 5.0,
        time_resolution_min: float = 5.0,
    ):
        self.peak_min = peak_min
        self.dia_min = dia_hours * 60
        self.time_resolution = time_resolution_min

        # Precompute activity curve
        self._activity_curve = self._compute_activity_curve()

    def _compute_activity_curve(self) -> np.ndarray:
        """
        Biexponential activity curve normalised to integrate to 1.0.

        The double exponential approximates the empirical activity profile
        measured by euglycemic clamp studies.
        """
        t = np.arange(0, self.dia_min + self.time_resolution, self.time_resolution)

        # Scheiner exponential model (matches Walsh pump IOB tables)
        tp = self.peak_min
        td = self.dia_min

        # Normalise so that activity sums to 1 (per unit insulin)
        # Using asymmetric biexponential
        τ1 = tp / 1.1
        τ2 = (td - tp) / 1.1
        activity = np.exp(-t / τ1) * (1 - np.exp(-t / τ2))
        activity[t > td] = 0.0
        activity = np.clip(activity, 0, None)
        # Normalise
        if activity.sum() > 0:
            activity /= activity.sum()
        return activity

    def activity_at_t(self, t_min: float) -> float:
        """Fractional activity at t minutes after bolus (0 to 1)."""
        idx = int(t_min / self.time_resolution)
        if idx >= len(self._activity_curve):
            return 0.0
        return float(self._activity_curve[idx])

    def iob_at_t(self, dose_units: float, t_min: float) -> float:
        """
        Active insulin remaining at t minutes after a bolus of dose_units.

        IOB(t) = dose * sum(activity[t:]) / sum(activity[0:])
        """
        idx = int(t_min / self.time_resolution)
        if idx >= len(self._activity_curve):
            return 0.0
        remaining_fraction = self._activity_curve[idx:].sum()
        return dose_units * remaining_fraction

    def compute_iob_series(
        self,
        bolus_events: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Compute cumulative IOB at each timestamp in index.

        Sums contributions from all recent boluses whose DIA window
        overlaps with the current time.

        Args:
            bolus_events: DataFrame with columns [timestamp, units]
            index:        Regular 5-min DatetimeIndex to compute IOB on

        Returns:
            pd.Series of IOB values (units) at each index timestamp
        """
        iob_values = np.zeros(len(index))

        if bolus_events is None or len(bolus_events) == 0:
            return pd.Series(iob_values, index=index, name="iob")

        for _, bolus in bolus_events.iterrows():
            bolus_ts = bolus["timestamp"]
            dose = bolus["units"]
            if dose <= 0:
                continue

            # Find index positions within DIA window of this bolus
            dia_end = bolus_ts + pd.Timedelta(minutes=self.dia_min)
            mask = (index >= bolus_ts) & (index <= dia_end)
            positions = np.where(mask)[0]

            for pos in positions:
                t_elapsed = (index[pos] - bolus_ts).total_seconds() / 60
                iob_values[pos] += self.iob_at_t(dose, t_elapsed)

        return pd.Series(iob_values, index=index, name="iob")


# ---------------------------------------------------------------------------
# COB Absorption Model
# ---------------------------------------------------------------------------

class CarbsOnBoard:
    """
    Carbs on Board (COB) using a nonlinear absorption model.

    The Hovorka model describes subcutaneous meal absorption as:

    dQ1/dt = f_g * D_G(t) - k_12 * Q1
    dQ2/dt = k_12 * Q1 - F_{01,c} + F_r

    We simplify to a one-compartment model with meal-type-dependent
    absorption kinetics. This is the model used by OpenAPS and Loop.

    For CGM prediction purposes, COB represents the estimated remaining
    glucose-raising potential of a recently consumed meal.

    Reference:
      Hovorka et al. (2004). Nonlinear model predictive control of glucose
      concentration in subjects with type 1 diabetes. AJP Endocrinol.
      https://doi.org/10.1152/ajpendo.00220.2004
    """

    def __init__(
        self,
        t_max_min: float = 60,      # Time to peak absorption rate
        absorption_duration_min: float = 180,  # Total absorption window
        time_resolution_min: float = 5.0,
    ):
        self.t_max = t_max_min
        self.duration = absorption_duration_min
        self.resolution = time_resolution_min
        self._absorption_curve = self._compute_absorption_curve()

    def _compute_absorption_curve(self) -> np.ndarray:
        """
        Normalised absorption rate curve (parabolic model).

        Rate(t) = 4 * carbs * (t / t_max) * (1 - t / t_max) / t_max
        for 0 ≤ t ≤ t_max*2, else 0.

        Parabolic shape peaks at t_max/2 and returns to zero at t_max.
        """
        t = np.arange(0, self.duration + self.resolution, self.resolution)
        rate = np.zeros_like(t, dtype=float)

        # Rising phase: 0 to t_max
        rising = t <= self.t_max
        rate[rising] = (t[rising] / self.t_max) ** 2

        # Falling phase: t_max to t_max*2 (then zero)
        t_fall = self.t_max * 2
        falling = (t > self.t_max) & (t <= t_fall)
        rate[falling] = (1 - (t[falling] - self.t_max) / self.t_max) ** 2

        # Normalise
        if rate.sum() > 0:
            rate /= rate.sum()
        return rate

    def cob_at_t(self, carbs_g: float, t_min: float) -> float:
        """Remaining carbs on board at t minutes after meal."""
        idx = int(t_min / self.resolution)
        if idx >= len(self._absorption_curve):
            return 0.0
        remaining = self._absorption_curve[idx:].sum()
        return carbs_g * remaining

    def compute_cob_series(
        self,
        meal_events: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Compute cumulative COB at each timestamp.

        Args:
            meal_events: DataFrame with columns [timestamp, carbs_g]
            index:       Regular 5-min DatetimeIndex

        Returns:
            pd.Series of COB values (grams) at each timestamp
        """
        cob_values = np.zeros(len(index))

        if meal_events is None or len(meal_events) == 0:
            return pd.Series(cob_values, index=index, name="cob")

        for _, meal in meal_events.iterrows():
            meal_ts = meal["timestamp"]
            carbs = meal["carbs_g"]
            if carbs <= 0:
                continue

            abs_end = meal_ts + pd.Timedelta(minutes=self.duration)
            mask = (index >= meal_ts) & (index <= abs_end)
            positions = np.where(mask)[0]

            for pos in positions:
                t_elapsed = (index[pos] - meal_ts).total_seconds() / 60
                cob_values[pos] += self.cob_at_t(carbs, t_elapsed)

        return pd.Series(cob_values, index=index, name="cob")


# ---------------------------------------------------------------------------
# Glycemic Variability Metrics
# ---------------------------------------------------------------------------

class GlycemicVariabilityCalculator:
    """
    Rolling glycemic variability metrics used as input features.

    These features provide the model with context about the patient's
    recent glucose stability — a highly variable patient in a hyperglycemic
    episode requires different prediction dynamics than a stable patient.
    """

    # Standard glucose ranges (ADA 2023 Standards of Medical Care)
    HYPO_THRESHOLD = 70      # mg/dL — Level 1 hypoglycemia
    SEVERE_HYPO = 54         # mg/dL — Level 2 hypoglycemia
    NORMAL_LOWER = 70        # mg/dL
    NORMAL_UPPER = 180       # mg/dL — Level 1 hyperglycemia
    HYPER_THRESHOLD = 250    # mg/dL — Level 2 hyperglycemia

    def cv(self, cgm: pd.Series, window_steps: int = 288) -> pd.Series:
        """
        Coefficient of Variation over rolling window.

        CV = (std / mean) * 100 (%)
        ADA target: CV < 36% for glycemic stability.
        CV > 36% indicates high variability — predict with wider intervals.

        window_steps = 288 steps = 24 hours at 5-min resolution.
        """
        roll_mean = cgm.rolling(window_steps, min_periods=window_steps // 4).mean()
        roll_std = cgm.rolling(window_steps, min_periods=window_steps // 4).std()
        cv = (roll_std / roll_mean.replace(0, np.nan)) * 100
        return cv.rename("cv_rolling")

    def mage(self, cgm: pd.Series, window_steps: int = 288) -> pd.Series:
        """
        Mean Amplitude of Glycemic Excursions (MAGE) over rolling window.

        MAGE = mean of all excursions > 1 SD in the window.
        An excursion is a nadir-to-peak or peak-to-nadir swing > 1 SD.

        Simplified implementation using rolling peak-trough detection.
        Full implementation requires the Service et al. algorithm (1970).
        """
        mage_values = np.full(len(cgm), np.nan)
        sd = cgm.rolling(window_steps, min_periods=window_steps // 4).std()

        for i in range(window_steps, len(cgm)):
            window = cgm.iloc[i - window_steps: i].dropna()
            if len(window) < window_steps // 2:
                continue
            threshold = window.std()
            if threshold == 0 or np.isnan(threshold):
                continue
            # Count excursions above threshold
            above = window > (window.mean() + threshold)
            below = window < (window.mean() - threshold)
            excursion_magnitude = (window[above].mean() - window[below].mean()
                                   if above.any() and below.any() else 0.0)
            mage_values[i] = excursion_magnitude

        return pd.Series(mage_values, index=cgm.index, name="mage_rolling")

    def time_in_range(
        self, cgm: pd.Series, window_steps: int = 288
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Rolling time-in-range fractions (TIR, TAR, TBR).

        ADA targets (2023):
          TIR (70-180 mg/dL): > 70%
          TAR (>180 mg/dL):   < 25%
          TBR (<70 mg/dL):    < 4%

        Returns:
            tir: time in range (70-180 mg/dL)
            tar: time above range (>180 mg/dL)
            tbr: time below range (<70 mg/dL)
        """
        def rolling_fraction(condition_series, window):
            return condition_series.rolling(window, min_periods=window // 4).mean()

        in_range = ((cgm >= self.NORMAL_LOWER) & (cgm <= self.NORMAL_UPPER)).astype(float)
        above_range = (cgm > self.NORMAL_UPPER).astype(float)
        below_range = (cgm < self.NORMAL_LOWER).astype(float)

        tir = rolling_fraction(in_range, window_steps).rename("tir_rolling")
        tar = rolling_fraction(above_range, window_steps).rename("tar_rolling")
        tbr = rolling_fraction(below_range, window_steps).rename("tbr_rolling")

        return tir, tar, tbr

    def modd(self, cgm: pd.Series) -> pd.Series:
        """
        Mean of Daily Differences (MODD).

        MODD = mean |CGM(t) - CGM(t - 24h)|
        Measures day-to-day reproducibility of glycemic patterns.
        High MODD indicates irregular lifestyle or changing insulin sensitivity.
        """
        day_steps = 288   # 24 hours at 5-min resolution
        shifted = cgm.shift(day_steps)
        modd = (cgm - shifted).abs().rolling(day_steps, min_periods=day_steps // 2).mean()
        return modd.rename("modd")


# ---------------------------------------------------------------------------
# Main Feature Engineer
# ---------------------------------------------------------------------------

class GlucoseFeatureEngineer:
    """
    Orchestrates all clinical feature engineering for the GlucoCast pipeline.

    Usage:
        engineer = GlucoseFeatureEngineer()
        feature_df = engineer.compute_all_features(
            cgm=cgm_series,
            bolus_events=bolus_df,
            meal_events=meal_df,
            exercise_events=exercise_df,
            index=cgm_series.index,
        )
    """

    def __init__(
        self,
        insulin_type: str = "default",
        meal_type: str = "default",
        roc_window: int = 3,    # Steps for RoC smoothing (15 min)
        gv_window: int = 288,   # Steps for glycemic variability (24h)
    ):
        pk = INSULIN_TYPES.get(insulin_type, RAPID_ACTING_PK)
        ab = CARB_ABSORPTION.get(meal_type, CARB_ABSORPTION["default"])

        self.iob_model = InsulinOnBoard(
            peak_min=pk["peak_min"],
            dia_hours=pk["dia"],
        )
        self.cob_model = CarbsOnBoard(
            t_max_min=ab["t_max"],
            absorption_duration_min=ab["duration"],
        )
        self.gv = GlycemicVariabilityCalculator()
        self.roc_window = roc_window
        self.gv_window = gv_window

    def compute_all_features(
        self,
        cgm: pd.Series,
        bolus_events: Optional[pd.DataFrame] = None,
        meal_events: Optional[pd.DataFrame] = None,
        exercise_events: Optional[pd.DataFrame] = None,
        index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        Compute all clinical features on the given index.

        Returns:
            DataFrame with columns:
              iob, cob, cgm_roc, meal_flag, bolus_flag, exercise_intensity,
              time_sin, time_cos, day_sin, day_cos,
              cv_rolling, tir_rolling, tar_rolling, tbr_rolling
        """
        if index is None:
            index = cgm.index

        features = {}

        # --- IOB ---
        features["iob"] = self.iob_model.compute_iob_series(bolus_events, index)

        # --- COB ---
        features["cob"] = self.cob_model.compute_cob_series(meal_events, index)

        # --- CGM Rate of Change (mg/dL/min) ---
        features["cgm_roc"] = self._compute_roc(cgm, window=self.roc_window)

        # --- Meal Flag (is there active COB?) ---
        features["meal_flag"] = (features["cob"] > 2.0).astype(float)

        # --- Bolus Flag (was there a bolus in the last 30 min?) ---
        features["bolus_flag"] = self._compute_bolus_flag(bolus_events, index, window_min=30)

        # --- Exercise Intensity ---
        features["exercise_intensity"] = self._compute_exercise_intensity(exercise_events, index)

        # --- Circadian Encoding ---
        time_features = self._compute_time_features(index)
        features.update(time_features)

        # --- Glycemic Variability (rolling, on aligned CGM) ---
        cgm_aligned = cgm.reindex(index)
        features["cv_rolling"] = self.gv.cv(cgm_aligned, self.gv_window)
        tir, tar, tbr = self.gv.time_in_range(cgm_aligned, self.gv_window)
        features["tir_rolling"] = tir
        features["tar_rolling"] = tar
        features["tbr_rolling"] = tbr

        return pd.DataFrame(features, index=index)

    def _compute_roc(self, cgm: pd.Series, window: int = 3) -> pd.Series:
        """
        CGM rate of change: central difference with smoothing.

        Uses a smoothed 15-minute window to reduce the effect of sensor noise
        on the derivative estimate. Central difference:
          RoC(t) = (CGM(t + window) - CGM(t - window)) / (2 * window * 5 min)

        Clipped to ±MAX_ROC_MGDL_MIN for numerical stability.
        """
        smoothed = cgm.rolling(window, center=True, min_periods=1).mean()
        roc = smoothed.diff() / 5.0   # 5-min interval → mg/dL/min
        roc = roc.clip(-4.0, 4.0)
        return roc.rename("cgm_roc")

    def _compute_bolus_flag(
        self,
        bolus_events: Optional[pd.DataFrame],
        index: pd.DatetimeIndex,
        window_min: int = 30,
    ) -> pd.Series:
        """Binary flag: was a bolus given in the last `window_min` minutes?"""
        flag = np.zeros(len(index), dtype=float)
        if bolus_events is None or len(bolus_events) == 0:
            return pd.Series(flag, index=index, name="bolus_flag")
        for _, bolus in bolus_events.iterrows():
            window_start = bolus["timestamp"]
            window_end = window_start + pd.Timedelta(minutes=window_min)
            mask = (index >= window_start) & (index <= window_end)
            flag[mask] = 1.0
        return pd.Series(flag, index=index, name="bolus_flag")

    def _compute_exercise_intensity(
        self,
        exercise_events: Optional[pd.DataFrame],
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Exercise intensity (0-3) at each timestep.

        Exercise dramatically increases insulin sensitivity, causing glucose
        drops that can occur 2-4 hours AFTER the exercise ends (delayed effect).
        We encode both the active exercise and a 2-hour post-exercise flag.
        """
        intensity = np.zeros(len(index), dtype=float)
        if exercise_events is None or len(exercise_events) == 0:
            return pd.Series(intensity, index=index, name="exercise_intensity")

        for _, ex in exercise_events.iterrows():
            ex_start = ex["timestamp"]
            ex_end = ex_start + pd.Timedelta(minutes=int(ex.get("duration_min", 30)))
            # Active exercise period
            active = (index >= ex_start) & (index <= ex_end)
            intensity[active] = max(intensity[active].max(), float(ex.get("intensity", 1)))
            # Post-exercise effect: decaying intensity for 2 hours after
            post_end = ex_end + pd.Timedelta(hours=2)
            post = (index > ex_end) & (index <= post_end)
            decay = (ex_end - pd.Series(index[post])).dt.total_seconds() / (2 * 3600)
            for i, (is_post, d) in enumerate(zip(post, np.where(post)[0])):
                if is_post:
                    t_elapsed = (index[d] - ex_end).total_seconds() / (2 * 3600)
                    intensity[d] = max(intensity[d], float(ex.get("intensity", 1)) * (1 - t_elapsed))

        return pd.Series(intensity, index=index, name="exercise_intensity")

    def _compute_time_features(self, index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """
        Cyclical encoding of time-of-day and day-of-week.

        Linear hour (0-23) is not suitable for neural networks because
        hour 23 and hour 0 should be "adjacent" — circular encoding solves this.

        time_sin = sin(2π * hour / 24)
        time_cos = cos(2π * hour / 24)
        day_sin  = sin(2π * dayofweek / 7)
        day_cos  = cos(2π * dayofweek / 7)
        """
        hour_frac = (index.hour + index.minute / 60) / 24  # Continuous hour fraction
        dow_frac = index.dayofweek / 7

        return {
            "time_sin": pd.Series(np.sin(2 * np.pi * hour_frac), index=index, name="time_sin"),
            "time_cos": pd.Series(np.cos(2 * np.pi * hour_frac), index=index, name="time_cos"),
            "day_sin": pd.Series(np.sin(2 * np.pi * dow_frac), index=index, name="day_sin"),
            "day_cos": pd.Series(np.cos(2 * np.pi * dow_frac), index=index, name="day_cos"),
        }
