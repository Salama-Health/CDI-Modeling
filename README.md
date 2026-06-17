# Salama Health: Climate Disruption Index Modelling

Predicting health facility disruptions in South Sudan, 28 days in advance.

---

## What This Is

This repository contains the full modelling pipeline for the **Climate Disruption Index (CDI)**, a composite risk score between 0 and 1 assigned to each Primary Health Care Centre across Unity, Jonglei, and Upper Nile states in South Sudan.

The CDI estimates the probability that a facility will be cut off from the communities it serves within the next 28 days. It combines four components, each targeting a distinct climate-driven disruption pathway that prevents children from receiving vaccines on time.

| Component | What It Measures | Status |
|---|---|---|
| P(flood) | Flood probability from Sentinel-1 SAR and SRTM elevation | Active |
| P(cutoff) | Road accessibility under flood and rainfall conditions | Active |
| P(CCF) | Cold chain failure risk from temperature exceedance above 8°C | Active |
| P(disp) | Population displacement pressure from IOM DTM | Active |

```
CDI = 0.35 × P(flood) + 0.30 × P(cutoff) + 0.20 × P(CCF) + 0.15 × P(disp)
```

CDI scores feed directly into the Salama Health mobile application, where community health workers see a daily-updated priority visit list ranked by the Immunisation Gap Score for each registered child in their catchment area.

---

## Pilot Coverage

- 93 Primary Health Care Centres across Unity, Jonglei, and Upper Nile states
- 2021 to 2024 observation window
- 9,617 facility-week training observations
- 30 features across satellite, climate, and displacement data sources

---

## The Problem This Solves

In South Sudan, children miss life-saving vaccines not because the vaccines do not exist, but because climate events cut off health facilities from the communities they serve. Floods submerge roads. Heat destroys cold chain equipment. Displacement moves families out of their registered catchment areas. These disruptions are predictable from satellite and climate signals weeks in advance.

The CDI pipeline processes those signals daily and surfaces the result to community health workers as a ranked list of children to visit before the disruption window closes.

---

## Data Sources

| Dataset | Source | CDI Component |
|---|---|---|
| Sentinel-1 SAR VV Backscatter | Google Earth Engine | P(flood) |
| SRTM Digital Elevation Model | Google Earth Engine | P(flood), P(cutoff) |
| CHIRPS Rainfall | HDX / Climate Hazards Center UCSB | P(flood), P(cutoff) |
| ERA5 Temperature and Humidity | Open-Meteo API | P(CCF) |
| OCHA Flood Affected People | Humanitarian Data Exchange | Disruption label |
| IOM Displacement Tracking Matrix | Humanitarian Data Exchange | P(disp) |
| UNOSAT Flood Extents | Humanitarian Data Exchange | Ground truth labels (pending) |

---

## Model Architecture

The pipeline is built across three phases following the Salama Health whitepaper specification.

**Phase 1: XGBoost Baseline**

A gradient-boosted classifier trained on 30 tabular features covering SAR backscatter, CHIRPS rainfall, Open-Meteo temperature and humidity, elevation, and displacement. Trained on 2021 to 2023 data and evaluated on a 2024 holdout set. Handles the 19 to 1 class imbalance using the scale-positive-weight parameter.

**Phase 2: Weighted Ensemble**

Three models combined using AUC-weighted averaging.

- XGBoost on all 30 tabular features, weight 0.341
- LSTM on 7-week climate sequences per facility, weight 0.333
- Random Forest on spatial and location features, weight 0.326

The LSTM is implemented in PyTorch. It processes sequences of 7 weekly observations across 15 climate features to capture temporal patterns such as rising rainfall trends and sustained heat exposure that a single-week snapshot cannot see.

**Phase 3: Immunisation Gap Score**

Translates facility-level CDI scores into child-level priority rankings using the IGS formula:

```
IGS(c, f, h) = CDI(f, h) × VD(c, h) × A(c)^-1 × UW(c)
```

Where VD is Vaccination Debt (fraction of due doses not yet received), A is Accessibility (distance adjusted for flood season and road conditions), and UW is Age-Urgency Weight (higher for younger children in critical vaccination windows).

**Current Performance**

| Model | OOF AUC |
|---|---|
| XGBoost | 1.0000 |
| Random Forest | 0.9548 |
| LSTM | 0.9775 |

Note: These AUC values are inflated because the current disruption label is derived from the same Sentinel-1 SAR signal used as the primary feature. This is a known limitation. Once UNOSAT verified flood extent ground truth labels replace the proxy label, the AUC will drop to a realistic value in the range of 0.70 to 0.85 that represents genuine predictive power. See the modelling report for a full discussion.

---

## Repository Structure

```
salama-cdi/
│
├── notebooks/
│   ├── Salama_Health_Processing_Pipeline.ipynb     Data loading, cleaning, and feature engineering
│   ├── Salama_Health_Modelling_Phase1.ipynb         XGBoost baseline and Phase 1 CDI scores
│   ├── Salama_Health_Modelling_Phase2.ipynb         LSTM, Random Forest, and ensemble CDI scores
│   └── Salama_Health_Modelling_Phase3.ipynb         IGS formula and priority visit list generation
│
├── data/
│   ├── raw/                                         Raw downloaded datasets (CHIRPS, DTM, flood)
│   └── outputs/                                     CDI scores, priority lists, trained models
│
└── README.md
```

---

## Getting Started

### Requirements

```bash
pip install xgboost lightgbm scikit-learn torch pandas numpy matplotlib tqdm
```

### Data Preparation

The pipeline loads data from five sources. All are available as direct downloads with no API key required except for Sentinel-1 and SRTM which require a Google Earth Engine account.

```
Sentinel-1 SAR      Google Earth Engine   earthengine.google.com
SRTM Elevation      Google Earth Engine   earthengine.google.com
CHIRPS Rainfall     Humanitarian Data Exchange   data.humdata.org
ERA5 Climate        Open-Meteo API        archive-api.open-meteo.com
IOM DTM             Humanitarian Data Exchange   data.humdata.org
OCHA Flood Data     Humanitarian Data Exchange   data.humdata.org
```

### Running the Pipeline

Run the notebooks in order. Each notebook saves its outputs to `/kaggle/working` for the next notebook to load.

```
1. Salama_Health_Processing_Pipeline.ipynb
   Downloads and merges all datasets, engineers features, saves SSD_Training_Dataset_v2.csv

2. Salama_Health_Modelling_Phase1.ipynb
   Trains XGBoost baseline, produces CDI scores, saves SSD_CDI_Scores_Phase1.csv

3. Salama_Health_Modelling_Phase2.ipynb
   Trains LSTM and Random Forest, produces ensemble CDI scores, saves SSD_CDI_Scores_Phase2.csv

4. Salama_Health_Modelling_Phase3.ipynb
   Loads child registry, computes IGS for all children, saves SSD_Priority_Visit_List.csv
```

---

## Key Outputs

| File | Description |
|---|---|
| SSD_Training_Dataset_v2.csv | 9,617 rows, 30 features, binary disruption label |
| SSD_CDI_Scores_Phase1.csv | CDI scores for 93 facilities from XGBoost |
| SSD_CDI_Scores_Phase2.csv | CDI scores for 93 facilities from ensemble |
| SSD_Priority_Visit_List.csv | All children ranked by IGS with overdue vaccines and caregiver contacts |
| SSD_XGBoost_Flood_Model.pkl | Trained XGBoost model |
| SSD_LSTM_Model.pt | Trained LSTM model weights |
| SSD_RF_Model.pkl | Trained Random Forest model |

---

---

## About

Salama Health is built by **Makarere AI Research Lab**.
Contact: mubarakatanvebb88@gmail.com
