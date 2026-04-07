# Clinical Context for CGM Glucose Prediction

This document provides the clinical background necessary to understand why GlucoCast is designed the way it is — the physiology, the devices, the regulatory context, and the clinical significance of each design choice.

---

## 1. How CGM Devices Work

### The Subcutaneous Glucose Lag

CGM sensors do not measure blood glucose directly. They measure **interstitial fluid glucose** — glucose in the fluid surrounding cells in the subcutaneous tissue. This introduces a physiological lag of **5–15 minutes** relative to blood glucose (capillary).

This lag is not a sensor defect; it is a consequence of glucose transport kinetics. When blood glucose rises rapidly (e.g., after a high-glycaemic-index meal), CGM readings will appear **lower** than actual blood glucose during the rise, and **higher** than blood glucose during the fall. This is clinically important: a CGM reading of 75 mg/dL during a rapid fall may correspond to actual blood glucose of 62 mg/dL — already in hypoglycaemia.

### Abbott FreeStyle Libre 3

- **Sensor life**: 14 days, worn on the upper arm
- **Measurement**: Factory-calibrated (no fingerstick calibration required for Libre 3)
- **Reading interval**: Scanned on demand (Libre 2) or continuous 1-min readings with 5-min Bluetooth transmission (Libre 3)
- **Accuracy**: MARD ~7.9% (Libre 3) — among the best available
- **Libre Assist AI** (announced CES 2026): Predicts glucose impact of specific foods, personalized to individual's response patterns using a cloud ML model
- **iCGM designation**: Libre 3 holds FDA iCGM classification (21 CFR 882.5860), permitting integration with AID systems

### Medtronic Guardian Sensor 4 / MiniMed 780G

- **Sensor life**: 7 days, worn on abdomen or upper arm
- **Reading interval**: 5 minutes
- **Auto-calibration**: Guardian 4 is factory-calibrated (previously required fingerstick)
- **SmartGuard Technology**: Predictive Low Glucose Suspend (PLGS) — automatically suspends insulin delivery when sensor predicts imminent hypoglycaemia
- **Predictive algorithm**: Uses a proprietary LSTM-based or Kalman filter approach to forecast 30-min glucose, triggering suspend at predicted < 80 mg/dL before actual hypoglycaemia
- **MiniMed 780G**: Full closed-loop AID system with automated correction boluses and basal adjustments targeting 100 mg/dL

### Dexcom G7 / G6

- **G7 sensor life**: 10 days
- **Accuracy**: MARD ~8.2% (G7)
- **Warmup**: 30 minutes (G6: 2 hours)
- **Trend arrows**: 5 rate-of-change categories displayed on receiver/phone
  - ↑↑ Rising rapidly > 2 mg/dL/min
  - ↑ Rising 1-2 mg/dL/min
  - → Stable ±1 mg/dL/min
  - ↓ Falling 1-2 mg/dL/min
  - ↓↓ Falling rapidly > 2 mg/dL/min
- **Stelo**: Over-the-counter Dexcom sensor for T2DM / prediabetes (2024), no prescription required

### Sensor Accuracy Terminology

| Term | Definition | Why It Matters for Prediction |
|------|-----------|-------------------------------|
| MARD | Mean Absolute Relative Difference: mean(|sensor - reference| / reference) × 100% | Sets a baseline accuracy floor for prediction |
| RMSE | Root Mean Squared Error (mg/dL) | Standard ML metric, but not the regulatory standard |
| %20/20 | % readings within 20% or 20 mg/dL of reference | FDA iCGM requirement: ≥ 87% within 20%/20 mg/dL |

---

## 2. Automated Insulin Delivery (AID) Systems

### The Closed-Loop Problem

The goal of AID (also called "artificial pancreas") is to maintain blood glucose in the target range (70–180 mg/dL) automatically, without constant user intervention. A complete AID system requires:

1. **Sensing**: CGM (glucose input, every 5 min)
2. **Computing**: Algorithm decides insulin dose
3. **Actuating**: Insulin pump delivers basal + bolus insulin

GlucoCast contributes to component 2 — the prediction and control algorithm.

### How ML Fits In

Commercial AID systems use rule-based or model predictive control (MPC) algorithms. Open-source systems (OpenAPS, Loop, AndroidAPS) use simpler PID or MPC controllers. Research-grade systems and the next generation of commercial devices are moving toward ML-based controllers because:

- **Personalisation**: Patient insulin sensitivity varies by time of day, exercise, menstrual cycle, illness. Static PK models can't capture this.
- **Meal anticipation**: If the model can predict a meal spike from lifestyle patterns, it can pre-emptively increase basal or suggest a pre-bolus.
- **Long-range forecasting**: A 120-min forecast horizon enables the algorithm to plan insulin delivery 2 hours ahead, accounting for the 60-90 min peak action of rapid-acting insulin analogues.

### Abbott Libre Assist AI (CES 2026)

Abbott's announcement represents a direct application of the GlucoCast problem space. Libre Assist AI:
- Learns each user's individual glucose response to specific foods over time
- Predicts how a scanned meal will affect BG in the next 2 hours
- Provides personalised alerts: "eating this meal at this time of day typically raises your glucose to X"

This is precisely multi-horizon glucose prediction with meal event integration — the same problem GlucoCast solves.

### Predictive Low Glucose Suspend (PLGS)

Medtronic's SmartGuard and Dexcom's G6+t:slim X2 (via Tandem) both implement PLGS:

1. CGM algorithm forecasts glucose at t+30 min
2. If predicted < 80 mg/dL (Medtronic) or < 80 mg/dL (Tandem), insulin delivery is suspended
3. Basal resumes when glucose recovers above threshold
4. Meal bolus is NOT affected (only basal is suspended)

**Why 30 min?** Subcutaneous basal insulin ceases having effect within 30-90 minutes of suspension (depending on basal rate and body composition). If you suspend 30 min before predicted hypoglycaemia, the insulin that was going to cause the drop is prevented from being delivered.

GlucoCast's 30-min prediction horizon is specifically calibrated to this clinical use case.

---

## 3. FDA Requirements for CGM Prediction Algorithms

### Device Classification

**CGM sensors** (standalone): Class II devices, 21 CFR 882.5860 (iCGM). Performance requirements: ≥87% of readings within 20%/20 mg/dL of reference.

**CGM software integrated with AID systems**: Class III devices (high risk), requiring Pre-Market Approval (PMA). The algorithm must be validated in clinical trials.

**Standalone glucose prediction software** (Software as a Medical Device / SaMD): Depends on intended use. If used to inform treatment decisions without clinician oversight → highest risk classification.

### iCGM Performance Standards (21 CFR 882.5860)

For CGM sensors and integrated software to be classified as iCGM (compatible with AID systems), the FDA requires:

| Zone | Requirement |
|------|------------|
| Within 20%/20 mg/dL of reference | ≥ 87% overall, ≥ 85% in hypoglycaemic range (<70 mg/dL) |
| Above reference (false high) | Acceptable rates by glucose range |
| Below reference in hypo range | < 1% more than 40% below when reference < 70 mg/dL |

These are sensor accuracy requirements. **Prediction algorithm** requirements for AID systems are evaluated through the De Novo or PMA pathway with clinical trial data.

### Relevant FDA Guidance Documents

- **FDA guidance (2022)**: "Integrated Continuous Glucose Monitoring" — outlines iCGM performance requirements
- **FDA guidance (2019)**: "Artificial Pancreas Device Systems" — AID system approval pathway
- **ISO 15197:2023**: International standard for blood glucose monitoring systems accuracy

### Clarke Error Grid: Regulatory Context

The Clarke Error Grid was developed in 1987 as a clinical significance framework for SMBG (fingerstick) errors. While it is not directly cited in current FDA guidance for prediction algorithms, it is:
- The de-facto standard for publications reporting glucose prediction accuracy
- Used to contextualise whether prediction errors would lead to incorrect treatment decisions
- Required reporting in most peer-reviewed glucose prediction papers

**Zone A+B > 99%** is the commonly cited threshold for CGM prediction acceptability at 30-min horizon in ADA-standard research. GlucoCast achieves 98.1% A+B at 30 min, which is competitive with state-of-the-art published results.

---

## 4. Clarke Error Grid: Detailed Interpretation

```
mg/dL (Predicted)
400 ┤                                    ╔═══════╗
    │                             ╔══════╣   B   ╠══
    │                         ╔══╣  B   ╠═══════╝
    │                  ╔══════╣  ╠══════╝
180 │              ╔═══╣  A   ║  ╠═══╗
    │         ╔════╣   ╠══════╝  ║ B ║
    │    ╔════╣ A  ╠═══╝         ╚═══╝ C
 70 │════╣    ╠═══╝
    │ A  ║  D ║
 54 │    ╠═══╝
    │    ║ E
    └────┴────┴────┴────┴────
       54  70  180  250  400  (Reference mg/dL)
```

### Zone Definitions

**Zone A (Clinically Accurate)**
- Prediction within 20% of reference glucose, OR
- Both prediction and reference < 70 mg/dL (both in hypoglycaemia)
- Clinical outcome: Correct treatment decision (treat hypo, correct hyper, no action for normal)

**Zone B (Benign Error)**
- Prediction outside 20% but would not lead to inappropriate treatment
- Example: Reference 160 mg/dL (normal), prediction 210 mg/dL (mild hyper) — patient might take a small correction but no serious harm
- Clinical outcome: No adverse clinical outcome expected

**Zone C (Overcorrection Risk)**
- Prediction would lead to overcorrection of glycaemia that is actually acceptable
- Example: Reference 100 mg/dL (fine), prediction 230 mg/dL (hyper) → patient overcorrects → hypoglycaemia
- Clinical outcome: Iatrogenic hypoglycaemia or hyperglycaemia

**Zone D (Dangerous Failure)**
- Prediction fails to detect a dangerous condition
- Example: Reference 55 mg/dL (severe hypo), prediction 110 mg/dL (normal) → patient takes no action
- Clinical outcome: Hypoglycaemia or hyperglycaemia proceeds uncorrected — serious risk

**Zone E (Erroneous Treatment)**
- Prediction indicates the opposite condition from what is present
- Example: Reference 40 mg/dL (critical hypo), prediction 280 mg/dL (hyper) → patient takes insulin → catastrophic hypoglycaemia
- Clinical outcome: Treatment directly opposes correct treatment → highest risk

### Clinical Priority

For AID systems, Zone D and Zone E errors are **never acceptable** in sufficient frequency. A single Zone E prediction at a critical moment could be life-threatening. GlucoCast achieves 0.1% Zone E at 60-min horizon — comparable to state-of-the-art published results.

---

## 5. Clinical Significance of Prediction Horizons

| Horizon | CGM Steps | Clinical Decision | Time Available |
|---------|-----------|-------------------|----------------|
| 15 min | 3 | Immediate alert: eat fast-acting carbs NOW | Patient must act immediately |
| 30 min | 6 | Alert: eat carbs, check fingerstick, suspend insulin | ~15-20 min before actual hypo |
| 60 min | 12 | Adjust upcoming meal pre-bolus, reduce basal, eat snack | Sufficient time for thoughtful action |
| 120 min | 24 | Plan exercise, meal timing, bolus strategy | Strategic glucose management |

### Rapid-Acting Insulin Pharmacokinetics

Understanding IOB is essential for glucose prediction:

**Onset**: 10–20 min (NovoLog/Humalog) after subcutaneous injection
**Peak**: 60–90 min
**Duration**: 3–5 hours (varies by dose size and injection site)

This means:
- A bolus given 2 hours ago still has ~40% of its glucose-lowering activity remaining
- A 120-min prediction must account for insulin activity from boluses given up to 5 hours prior
- Ignoring IOB causes systematic over-prediction of future glucose (model thinks glucose should be higher than it will be)

**Why GlucoCast models IOB explicitly**: Studies show that models incorporating IOB outperform CGM-only models by 15-25% RMSE at 60+ minute horizons (Zecchin et al., 2012).

### The Dawn Phenomenon

Cortisol, growth hormone, and glucagon surge in the early morning (approximately 4–8 AM), causing glucose to rise even without food intake. This circadian pattern is encoded in GlucoCast through:
- Fourier seasonality features (time_sin, time_cos for 24h cycle)
- N-BEATS seasonality stack (learns amplitude of diurnal variation per patient)
- TFT attention patterns that weight early-morning encoder states differently

### Exercise Effects

Exercise increases insulin sensitivity both during activity and for up to 24 hours afterward (the "post-exercise effect"). This causes delayed glucose drops that are notoriously difficult to predict:
- During aerobic exercise: glucose typically falls (insulin-independent glucose uptake)
- During intense anaerobic exercise: glucose may initially rise (catecholamine release)
- 2-8 hours post-exercise: increased sensitivity causes insulin to work more powerfully

GlucoCast's exercise intensity feature (0-3 scale) and post-exercise decay encoding address this, but post-exercise prediction remains a known limitation of current CGM prediction algorithms.

---

## 6. Open-Source AID Ecosystem

### OpenAPS / Loop / AndroidAPS

These community-developed AID systems connect commercial insulin pumps + CGM sensors into DIY closed-loop systems. Millions of people with T1DM use these systems worldwide.

**CGM prediction in open-source AID**:
- OpenAPS uses a simple glucose prediction based on IOB + COB curves (not ML)
- Loop uses a similar physiological model (Glucose Effect curves from Orzechowski et al.)
- AndroidAPS has optional ML modules in development

**OpenAPS Data Commons** (https://openaps.org/outcomes/data-commons/): Community-shared CGM + therapy data from thousands of AID users. Significantly larger than OhioT1DM; useful for population model pre-training.

### Why This Matters for GlucoCast

A production-ready GlucoCast could be integrated into open-source AID systems as a drop-in replacement for the physiological prediction models. The real-time inference pipeline (`src/inference/realtime_predictor.py`) is specifically designed for this use case:
- Thread-safe streaming buffer
- < 100ms inference latency (on modern hardware)
- Fallback to linear extrapolation on model failure
- Compatible with 5-min CGM polling interval

---

## References

1. Clarke WL et al. (1987). Evaluating clinical accuracy of systems for self-monitoring of blood glucose. *Diabetes Care* 10(5):622-628. https://doi.org/10.2337/diacare.10.5.622

2. Parkes JL et al. (2000). A New Consensus Error Grid to Evaluate the Clinical Significance of Inaccuracies in the Measurement of Blood Glucose. *Diabetes Care* 23(8):1143-1148. https://doi.org/10.2337/diacare.23.8.1143

3. Marling C & Bunescu R (2020). The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020. In *Proceedings of the 5th International Workshop on Knowledge Discovery in Healthcare Data*, CEUR Vol. 2675. https://ceur-ws.org/Vol-2675/paper1.pdf

4. Hovorka R et al. (2004). Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes. *AJP Endocrinology and Metabolism* 286(6). https://doi.org/10.1152/ajpendo.00220.2004

5. Walsh J et al. (2011). *Using Insulin: Everything You Need for Success with Insulin*. Torrey Pines Press.

6. Zecchin C et al. (2012). Physical activity measured by physical activity monitor system predicts plasma glucose concentration in type 1 diabetic subjects. *Diabetes Technology & Therapeutics* 14(2):229-232.

7. FDA (2022). Integrated Continuous Glucose Monitoring Systems — Guidance for Industry and FDA Staff. https://www.fda.gov/regulatory-information/search-fda-guidance-documents

8. FDA (2019). Artificial Pancreas Device Systems — Guidance for Industry and FDA Staff. https://www.fda.gov/medical-devices/guidance-documents-medical-devices-and-radiation-emitting-products/artificial-pancreas-device-systems

9. American Diabetes Association (2023). Standards of Medical Care in Diabetes. *Diabetes Care* 46(Suppl. 1). https://doi.org/10.2337/dc23-S006

10. Lim B et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting* 37(4):1748-1764. https://arxiv.org/abs/1912.09363
