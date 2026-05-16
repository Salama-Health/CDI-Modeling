# Salama Health — Climate Disruption Index (CDI) Modelling

> Predicting health facility disruptions in South Sudan, 28 days in advance.

## What This Is

This repository contains the modelling notebooks for the **Climate Disruption Index (CDI)** — a composite risk score (0–1) assigned to each Primary Health Care Centre (PHCC) across Unity, Jonglei, and Upper Nile states in South Sudan.

The CDI estimates the probability that a facility will be cut off from the communities it serves within the next 28 days, combining four components:

| Component | Description | Status |
|---|---|---|
| P(flood) | Flood probability from Sentinel-1 SAR + SRTM elevation | ✅ Active |
| P(road cutoff) | Road accessibility under flood conditions | ✅ Active |
| P(cold chain failure) | Temperature-driven cold chain risk from ERA5 | 🔧 In progress |
| P(displacement) | Population movement pressure from IOM DTM | 🔧 Next phase |

**CDI Formula:**
```
CDI = w₁·P(flood) + w₂·P(road_cutoff) + w₃·P(cold_chain) + w₄·P(displacement)
```

---

## Pilot Coverage

- **93 Primary Health Care Centres** across Unity, Jonglei, and Upper Nile states
- **6 years of satellite observations** (2019–2024)
- **18,000+ data points** processed

---

## Data Sources

| Source | Data | Use |
|---|---|---|
| [Sentinel-1 SAR](https://sentinel.esa.int) | Radar backscatter | Flood signal detection |
| [SRTM DEM](https://www2.jpl.nasa.gov/srtm/) | Elevation | Road cutoff & flood extent |
| [ERA5 (ECMWF)](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) | Temperature | Cold chain failure risk |
| [NASA IMERG](https://gpm.nasa.gov/data/imerg) | Rainfall | Flood predictor (next phase) |
| [IOM DTM South Sudan](https://dtm.iom.int/south-sudan) | Displacement data | Displacement pressure (next phase) |
| [UNOSAT](https://unosat.org) | Verified flood extents | Ground truth labels (next phase) |
| [HDX — South Sudan Health Facilities](https://data.humdata.org) | Facility GPS coordinates | Master facility list |

---

## Model

**Algorithm:** XGBoost classifier

**Features:**
- 7-day rolling rainfall (NASA IMERG)
- Current NDWI — Normalized Difference Water Index (Sentinel-1)
- Distance to river (SRTM DEM)
- Elevation

**Target:** Historical flood disruption events

**Performance:**

| Metric | Score |
|---|---|
| AUC | 0.9999 |
| F1 Score | 0.96 |

> **Note:** Current labels are derived from proxy satellite signals. The next phase will replace these with verified UNOSAT flood extent data for independent ground truth validation.

---

## Getting Started

### Requirements

```bash
pip install earthengine-api xgboost pandas geopandas scikit-learn matplotlib
```

### Google Earth Engine Authentication

```python
import ee
ee.Authenticate()
ee.Initialize(project='your-project-id')
```

### Run the Pipeline

1. `01_gee_data_pipeline.ipynb` — pull satellite data for target facilities
2. `02_flood_model_xgboost.ipynb` — train and evaluate the flood model
3. `03_cdi_assembly.ipynb` — compute CDI scores for all 93 facilities
4. `04_facility_risk_ranking.ipynb` — generate ranked facility risk output

---

## Roadmap

- [ ] Replace proxy labels with verified UNOSAT flood extent ground truth
- [ ] Integrate NASA IMERG live rainfall as a model feature
- [ ] Fix ERA5 temperature dataset for cold chain component
- [ ] Activate IOM DTM displacement pressure component

---
