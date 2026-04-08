"""
OhioT1DM Dataset Loader.

The OhioT1DM dataset (Marling & Bunescu, 2020) contains 8 weeks of
longitudinal data for 12 people with Type 1 Diabetes. Each patient's data
is stored in an XML file with synchronised streams:

  <glucose_level>      5-minute Medtronic Enlite CGM readings (mg/dL)
  <finger_stick>       Reference BG fingerstick values
  <basal>              Pump basal rate (units/hr)
  <temp_basal>         Temporary basal rate adjustments
  <bolus>              Insulin bolus events (type, units, carbs if meal bolus)
  <meal>               Self-reported carbohydrate estimates
  <sleep>              Sleep quality and duration logs
  <work>               Work activity logs
  <stressors>          Self-reported stress events
  <hypo_event>         Clinician-annotated hypoglycemia events
  <exercise>           Exercise type and intensity
  <basis_heart_rate>   Wrist PPG heart rate (Intel Basis Band)
  <basis_gsr>          Galvanic skin response
  <basis_skin_temp>    Wrist skin temperature
  <basis_steps>        Accelerometer step count

Dataset access: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html
Requires a signed data use agreement.

Patient IDs (2018 cohort): 559, 563, 570, 575, 588, 591
Patient IDs (2020 cohort): 540, 544, 552, 567, 584, 596
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.preprocessing import CGMPreprocessor
from src.data.feature_engineering import GlucoseFeatureEngineer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PatientData:
    """All time-series streams for a single OhioT1DM patient."""
    patient_id: str
    cgm: pd.Series                     # mg/dL, 5-min intervals (NaN for gaps)
    finger_sticks: pd.Series           # mg/dL, irregular timing
    basal: pd.Series                   # units/hr, insulin pump basal rate
    bolus_events: pd.DataFrame         # columns: timestamp, units, type, bgInput, carbInput
    meal_events: pd.DataFrame          # columns: timestamp, carbs_g
    exercise_events: pd.DataFrame      # columns: timestamp, type, intensity, duration_min
    sleep_events: pd.DataFrame         # columns: timestamp, quality, duration_min
    hypo_events: pd.DataFrame          # clinician-annotated: timestamp, level
    demographics: Dict                 # age, weight, insulin_type, etc. (from XML header)

    def __post_init__(self):
        # Ensure datetime index
        if not isinstance(self.cgm.index, pd.DatetimeIndex):
            self.cgm.index = pd.to_datetime(self.cgm.index)
        if not isinstance(self.basal.index, pd.DatetimeIndex):
            self.basal.index = pd.to_datetime(self.basal.index)


# ---------------------------------------------------------------------------
# XML Parser
# ---------------------------------------------------------------------------

class OhioXMLParser:
    """
    Parses OhioT1DM XML files into structured Python objects.

    XML format example:
    <patient id="559" weight="..." insulin_type="...">
        <glucose_level>
            <event ts="2018-01-01 00:00:00" value="145"/>
            ...
        </glucose_level>
        <bolus>
            <event ts="..." type="normal" dose="2.5" bgInput="162" carbInput="40"/>
        </bolus>
    </patient>
    """

    VALID_CGM_RANGE = (39, 401)   # mg/dL: Enlite sensor measurement range

    def parse_file(self, xml_path: Path) -> PatientData:
        """Parse a single OhioT1DM XML file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        patient_id = root.attrib.get("id", xml_path.stem)
        demographics = {
            "weight_kg": float(root.attrib.get("weight", 0)),
            "insulin_type": root.attrib.get("insulin_type", "unknown"),
            "hba1c": float(root.attrib.get("hba1c", 0)) if "hba1c" in root.attrib else None,
        }

        cgm = self._parse_glucose(root.find("glucose_level"))
        finger_sticks = self._parse_fingerstick(root.find("finger_stick"))
        basal = self._parse_basal(root, cgm.index)
        bolus_events = self._parse_bolus(root.find("bolus"))
        meal_events = self._parse_meals(root.find("meal"))
        exercise_events = self._parse_exercise(root.find("exercise"))
        sleep_events = self._parse_sleep(root.find("sleep"))
        hypo_events = self._parse_hypo_events(root.find("hypo_event"))

        logger.info(
            f"Parsed patient {patient_id}: {len(cgm)} CGM readings, "
            f"{len(bolus_events)} boluses, {len(meal_events)} meals, "
            f"{cgm.isna().sum()} missing values"
        )

        return PatientData(
            patient_id=patient_id,
            cgm=cgm,
            finger_sticks=finger_sticks,
            basal=basal,
            bolus_events=bolus_events,
            meal_events=meal_events,
            exercise_events=exercise_events,
            sleep_events=sleep_events,
            hypo_events=hypo_events,
            demographics=demographics,
        )

    def _parse_glucose(self, node: Optional[ET.Element]) -> pd.Series:
        """Parse CGM glucose readings, converting OOB values to NaN."""
        if node is None:
            return pd.Series(dtype=float, name="cgm")
        records = []
        for event in node.findall("event"):
            ts = pd.to_datetime(event.attrib["ts"])
            val = float(event.attrib["value"])
            # Sensor reports 39 mg/dL for LOW and 401 for HIGH
            if val < self.VALID_CGM_RANGE[0] or val > self.VALID_CGM_RANGE[1]:
                val = float("nan")
            records.append((ts, val))
        if not records:
            return pd.Series(dtype=float, name="cgm")
        ts_idx, values = zip(*records)
        s = pd.Series(values, index=pd.DatetimeIndex(ts_idx), name="cgm")
        # Resample to strict 5-min grid (fill any missing timestamps with NaN)
        s = s.resample("5T").last()
        return s

    def _parse_fingerstick(self, node: Optional[ET.Element]) -> pd.Series:
        if node is None:
            return pd.Series(dtype=float, name="fingerstick")
        records = {
            pd.to_datetime(e.attrib["ts"]): float(e.attrib["value"])
            for e in node.findall("event")
        }
        return pd.Series(records, name="fingerstick")

    def _parse_basal(
        self,
        root: ET.Element,
        cgm_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Reconstruct continuous basal rate series by forward-filling basal events.

        The pump reports basal changes; between events the rate is constant.
        Temp basal overrides are applied on top.
        """
        records = {}
        basal_node = root.find("basal")
        if basal_node is not None:
            for event in basal_node.findall("event"):
                ts = pd.to_datetime(event.attrib["ts"])
                rate = float(event.attrib["value"])
                records[ts] = rate

        temp_node = root.find("temp_basal")
        if temp_node is not None:
            for event in temp_node.findall("event"):
                ts = pd.to_datetime(event.attrib["ts"])
                rate = float(event.attrib.get("value", 0))
                duration = int(event.attrib.get("duration", 30))  # minutes
                # Apply temp basal for its duration
                for i in range(0, duration, 5):
                    records[ts + pd.Timedelta(minutes=i)] = rate

        if not records:
            return pd.Series(0.0, index=cgm_index, name="basal_rate")

        basal_series = pd.Series(records, name="basal_rate")
        basal_series = basal_series.reindex(cgm_index).ffill().fillna(0.0)
        return basal_series

    def _parse_bolus(self, node: Optional[ET.Element]) -> pd.DataFrame:
        if node is None:
            return pd.DataFrame(columns=["timestamp", "units", "bolus_type", "bg_input", "carb_input"])
        records = []
        for event in node.findall("event"):
            records.append({
                "timestamp": pd.to_datetime(event.attrib["ts"]),
                "units": float(event.attrib.get("dose", 0)),
                "bolus_type": event.attrib.get("type", "normal"),
                "bg_input": float(event.attrib.get("bgInput", 0)),
                "carb_input": float(event.attrib.get("carbInput", 0)),
            })
        return pd.DataFrame(records)

    def _parse_meals(self, node: Optional[ET.Element]) -> pd.DataFrame:
        if node is None:
            return pd.DataFrame(columns=["timestamp", "carbs_g"])
        records = [
            {"timestamp": pd.to_datetime(e.attrib["ts"]), "carbs_g": float(e.attrib.get("carbs", 0))}
            for e in node.findall("event")
        ]
        return pd.DataFrame(records)

    def _parse_exercise(self, node: Optional[ET.Element]) -> pd.DataFrame:
        if node is None:
            return pd.DataFrame(columns=["timestamp", "exercise_type", "intensity", "duration_min"])
        records = []
        for e in node.findall("event"):
            records.append({
                "timestamp": pd.to_datetime(e.attrib["ts"]),
                "exercise_type": e.attrib.get("type", "unknown"),
                "intensity": int(e.attrib.get("intensity", 0)),
                "duration_min": int(e.attrib.get("duration", 30)),
            })
        return pd.DataFrame(records)

    def _parse_sleep(self, node: Optional[ET.Element]) -> pd.DataFrame:
        if node is None:
            return pd.DataFrame(columns=["timestamp", "quality", "duration_min"])
        records = [
            {
                "timestamp": pd.to_datetime(e.attrib["ts"]),
                "quality": int(e.attrib.get("quality", 0)),
                "duration_min": int(e.attrib.get("duration", 0)),
            }
            for e in node.findall("event")
        ]
        return pd.DataFrame(records)

    def _parse_hypo_events(self, node: Optional[ET.Element]) -> pd.DataFrame:
        if node is None:
            return pd.DataFrame(columns=["timestamp", "level"])
        records = [
            {"timestamp": pd.to_datetime(e.attrib["ts"]), "level": e.attrib.get("level", "mild")}
            for e in node.findall("event")
        ]
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Dataset Splits
# ---------------------------------------------------------------------------

class OhioT1DM:
    """
    High-level interface for the OhioT1DM dataset.

    Usage:
        dataset = OhioT1DM(data_dir="data/ohio/")
        patient_ids = dataset.patient_ids

        # Single patient splits
        train, val, test = dataset.get_patient_splits("559")

        # Leave-one-patient-out cross-validation
        for train_loader, test_loader, test_id in dataset.loocv_splits():
            ...
    """

    # Chronological split: 6 weeks train, 1 week val, 1 week test
    TRAIN_FRAC = 0.75
    VAL_FRAC = 0.125
    TEST_FRAC = 0.125

    def __init__(
        self,
        data_dir: str,
        encoder_steps: int = 72,       # 6-hour lookback at 5-min resolution
        prediction_horizons: List[int] = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.encoder_steps = encoder_steps
        self.prediction_horizons = prediction_horizons or [6, 12, 24]
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.parser = OhioXMLParser()
        self.preprocessor = CGMPreprocessor()
        self.feature_engineer = GlucoseFeatureEngineer()

        self._patient_cache: Dict[str, PatientData] = {}
        self._find_patients()

    def _find_patients(self):
        """Discover patient XML files in data_dir."""
        xml_files = list(self.data_dir.glob("*.xml"))
        self.patient_ids = [f.stem for f in xml_files]
        logger.info(f"Found {len(self.patient_ids)} patients: {self.patient_ids}")

    def load_patient(self, patient_id: str) -> PatientData:
        """Load and cache patient data."""
        if patient_id not in self._patient_cache:
            xml_path = self.data_dir / f"{patient_id}.xml"
            if not xml_path.exists():
                # Try train/test subdirectories (OhioT1DM distribution structure)
                xml_path = self.data_dir / "train" / f"{patient_id}-ws-training.xml"
                if not xml_path.exists():
                    xml_path = self.data_dir / "test" / f"{patient_id}-ws-testing.xml"
            self._patient_cache[patient_id] = self.parser.parse_file(xml_path)
        return self._patient_cache[patient_id]

    def build_feature_dataframe(self, patient_id: str) -> pd.DataFrame:
        """
        Build the full feature matrix for a patient.

        Columns in output DataFrame:
          cgm, cgm_norm, basal_rate, iob, cob, cgm_roc,
          meal_flag, bolus_flag, exercise_intensity,
          time_sin, time_cos, day_sin, day_cos,
          cv_rolling, mage_rolling, time_in_range_rolling,
          hypo_flag (target label for alert model)
        """
        pdata = self.load_patient(patient_id)

        # 1. Preprocess CGM signal
        cgm_clean = self.preprocessor.process(pdata.cgm)

        # 2. Build base DataFrame on CGM index
        df = pd.DataFrame({"cgm": cgm_clean}, index=cgm_clean.index)

        # 3. Add basal rate
        df["basal_rate"] = pdata.basal.reindex(df.index).ffill().fillna(0.0)

        # 4. Feature engineering
        features = self.feature_engineer.compute_all_features(
            cgm=df["cgm"],
            bolus_events=pdata.bolus_events,
            meal_events=pdata.meal_events,
            exercise_events=pdata.exercise_events,
            index=df.index,
        )
        df = pd.concat([df, features], axis=1)

        # 5. Drop rows with persistent NaN (gaps > imputation threshold)
        df = df.dropna(subset=["cgm"])
        df = df.fillna(0.0)   # Zero-fill remaining NaN (e.g., no meal/bolus)

        return df

    def get_patient_splits(
        self, patient_id: str
    ) -> Tuple["CGMWindowDataset", "CGMWindowDataset", "CGMWindowDataset"]:
        """
        Build train/val/test DataLoaders for a single patient.

        Split is chronological (no shuffling across time boundaries):
          - Train: first 75% of readings
          - Val:   next 12.5%
          - Test:  last 12.5%

        This mimics real-world deployment: model trained on historical data,
        evaluated on future unseen data.
        """
        df = self.build_feature_dataframe(patient_id)
        n = len(df)
        n_train = int(n * self.TRAIN_FRAC)
        n_val = int(n * self.VAL_FRAC)

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train: n_train + n_val]
        test_df = df.iloc[n_train + n_val:]

        # Compute normalization statistics on TRAINING set only
        cgm_mean = train_df["cgm"].mean()
        cgm_std = train_df["cgm"].std()
        norm_params = {"cgm_mean": cgm_mean, "cgm_std": cgm_std}

        train_ds = CGMWindowDataset(train_df, self.encoder_steps, self.prediction_horizons, norm_params)
        val_ds = CGMWindowDataset(val_df, self.encoder_steps, self.prediction_horizons, norm_params)
        test_ds = CGMWindowDataset(test_df, self.encoder_steps, self.prediction_horizons, norm_params)

        return train_ds, val_ds, test_ds

    def get_dataloaders(
        self, patient_id: str
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Convenience wrapper returning DataLoaders."""
        train_ds, val_ds, test_ds = self.get_patient_splits(patient_id)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader, test_loader

    def loocv_splits(
        self,
    ) -> Iterator[Tuple[DataLoader, DataLoader, str]]:
        """
        Leave-one-patient-out cross-validation (LOPOCV).

        The standard evaluation protocol for the OhioT1DM dataset.
        For each patient, all other patients form the training set and
        the held-out patient forms the test set.

        Yields: (train_loader, test_loader, test_patient_id)
        """
        for test_id in self.patient_ids:
            train_ids = [pid for pid in self.patient_ids if pid != test_id]

            # Collect train data from all other patients
            train_dfs = []
            for pid in train_ids:
                try:
                    df = self.build_feature_dataframe(pid)
                    train_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Skipping patient {pid}: {e}")

            if not train_dfs:
                continue

            all_train = pd.concat(train_dfs, ignore_index=True)
            # Compute global normalization from population training set
            cgm_mean = all_train["cgm"].mean()
            cgm_std = all_train["cgm"].std()
            norm_params = {"cgm_mean": cgm_mean, "cgm_std": cgm_std}

            # Population train loader
            train_ds = CGMWindowDataset(all_train, self.encoder_steps,
                                        self.prediction_horizons, norm_params)
            train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                      shuffle=True, num_workers=self.num_workers)

            # Test: held-out patient with same population norm params
            test_df = self.build_feature_dataframe(test_id)
            test_ds = CGMWindowDataset(test_df, self.encoder_steps,
                                       self.prediction_horizons, norm_params)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size,
                                     shuffle=False, num_workers=self.num_workers)

            logger.info(f"LOPOCV: train={len(train_dfs)} patients, test={test_id}, "
                        f"train_windows={len(train_ds)}, test_windows={len(test_ds)}")
            yield train_loader, test_loader, test_id


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CGMWindowDataset(Dataset):
    """
    Sliding window dataset for CGM sequence prediction.

    Each sample is a tuple (window, targets, static) where:
      - window:  [encoder_steps + max_horizon, n_features] — the input window
                 (encoder_steps of history + max_horizon of known future features)
      - targets: [n_horizons] — CGM values at each prediction horizon
                 (normalised, in mg/dL units for unnormalised evaluation)
      - static:  [n_static] — patient-level static covariates (if available)

    Gap handling: Windows that span CGM gaps > 15 minutes are excluded
    (imputation doesn't extend beyond short gaps for safety).
    """

    MAX_GAP_STEPS = 3   # Skip windows with gaps > 15 min (3 x 5min)

    FEATURE_COLUMNS = [
        "cgm_norm",         # Normalised CGM
        "basal_rate",       # Insulin pump basal rate (units/hr)
        "iob",              # Insulin on board (units)
        "cob",              # Carbs on board (g)
        "cgm_roc",          # CGM rate of change (mg/dL/min)
        "meal_flag",        # Active meal absorption (0/1)
        "bolus_flag",       # Recent bolus (0/1)
        "exercise_intensity", # 0-3
        "time_sin",         # Circadian: sin(2π * hour/24)
        "time_cos",         # Circadian: cos(2π * hour/24)
        "day_sin",          # Weekly: sin(2π * dow/7)
        "day_cos",          # Weekly: cos(2π * dow/7)
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        encoder_steps: int,
        prediction_horizons: List[int],
        norm_params: Dict[str, float],
    ):
        self.encoder_steps = encoder_steps
        self.prediction_horizons = prediction_horizons
        self.max_horizon = max(prediction_horizons)
        self.norm_params = norm_params

        # Normalise CGM
        df = df.copy()
        mu, sigma = norm_params["cgm_mean"], norm_params["cgm_std"]
        df["cgm_norm"] = (df["cgm"] - mu) / (sigma + 1e-8)

        # Select and order feature columns
        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        missing = [c for c in self.FEATURE_COLUMNS if c not in df.columns]
        if missing:
            logger.debug(f"Missing features (will zero-fill): {missing}")
            for c in missing:
                df[c] = 0.0

        self.features = df[self.FEATURE_COLUMNS].values.astype(np.float32)  # [N, F]
        self.cgm_raw = df["cgm"].values.astype(np.float32)   # For denorm evaluation
        self.cgm_norm = df["cgm_norm"].values.astype(np.float32)

        # Build valid window indices (exclude windows spanning large gaps)
        self._build_valid_indices()

    def _build_valid_indices(self):
        """Pre-compute valid starting indices for sliding window."""
        n = len(self.features)
        total_steps = self.encoder_steps + self.max_horizon
        valid = []
        for start in range(n - total_steps + 1):
            end = start + total_steps
            # Check for CGM gaps within window
            window_cgm = self.cgm_raw[start:end]
            # Count consecutive NaN runs
            nans = np.isnan(window_cgm)
            if nans.sum() == 0:
                valid.append(start)
            elif nans.sum() <= self.MAX_GAP_STEPS:
                valid.append(start)
            # Skip windows with too many missing values
        self.valid_indices = valid
        logger.debug(f"Dataset: {n} total rows, {len(valid)} valid windows")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.valid_indices[idx]
        enc_end = start + self.encoder_steps
        fut_end = enc_end + self.max_horizon

        # Historical window (encoder input)
        historical = torch.from_numpy(self.features[start:enc_end])     # [T_enc, F]

        # Future window (for TFT future inputs)
        future = torch.from_numpy(self.features[enc_end:fut_end])       # [T_fut, F]

        # Targets: raw CGM values at each horizon
        targets = torch.tensor(
            [self.cgm_raw[enc_end + h - 1] for h in self.prediction_horizons],
            dtype=torch.float32,
        )   # [n_horizons]

        # Normalised targets (for training loss)
        targets_norm = torch.tensor(
            [self.cgm_norm[enc_end + h - 1] for h in self.prediction_horizons],
            dtype=torch.float32,
        )

        return {
            "historical": historical,
            "future": future,
            "targets": targets,
            "targets_norm": targets_norm,
        }


import pytz as _pytz


def normalise_timestamps_to_utc(df, local_tz: str = "America/New_York"):
    """
    Convert naive local timestamps to UTC-aware datetimes.
    OhioT1DM timestamps are stored in local time without timezone info,
    which causes off-by-one-hour bugs around DST transitions.
    """
    tz = _pytz.timezone(local_tz)

    if df["timestamp"].dt.tz is None:
        # assume local time, localise then convert
        df = df.copy()
        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            .dt.tz_convert("UTC")
        )
    elif str(df["timestamp"].dt.tz) != "UTC":
        df = df.copy()
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    return df
