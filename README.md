# 🧭 The LogPose Crew - Parking-Induced Congestion Intelligence System

> *"Always pointing to the next hotspot."*

**Flipkart GridLock Hackathon 2.0 - Round 2 | Problem Statement 1**
Built for Bengaluru Traffic Police (BTP)

---

## 📊 At a Glance

| Metric | Value |
|--------|-------|
| Ensemble Accuracy | **94.83%** (LightGBM + XGBoost) |
| Hotspot Clusters | **164** (DBSCAN @ 300m radius) |
| Training Records | **141,409** (device-trust recovery) |
| Dashboard Features | **35** (Streamlit live app) |
| Raw BTP Records Analyzed | **298,450** |
| NaN Records Recovered | **26,009** |
| SCITA Sync Rate | **93.83%** |

---

## 👥 Team

| Member | Role | Contributions |
|--------|------|---------------|
| **Himanshu Mahajan** | Team Lead / AI & ML Lead | Overall pipeline design, ML model training and ensembling, DBSCAN clustering, feature importance analysis, presentation and documentation lead. Also led project strategy, co-developed the DBSCAN & LightGBM/XGBoost ML pipeline, built the data cleaning pipeline, and generated key datasets and maps. |
| **Imran Farhat** | System Architecture & DevOps | Data cleaning pipeline and device-trust NaN recovery, feature engineering architecture, leakage-free design enforcement, MongoDB integration, deployment configuration. Co-developed and refined ensemble models; led deployment to ensure the Streamlit dashboard was stable, hosted, and accessible. |
| **Mohit Bhootra** | Frontend Lead | Streamlit dashboard design and implementation (`app.py`), CSS theming system, all 7 dashboard pages, map embedding, and export functionality. Designed and managed the project's database architecture using MongoDB; led backend development and cross-component system integrations. |

**GitHub:** [github.com/HimanshuMahajan2111/The-LogPose-Crew](https://github.com/HimanshuMahajan2111/The-LogPose-Crew)

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [System Architecture](#4-system-architecture)
5. [Data Foundation & Cleaning Pipeline](#5-data-foundation--cleaning-pipeline)
6. [Feature Engineering](#6-feature-engineering)
7. [Violation Label Engineering](#7-violation-label-engineering)
8. [ML Pipeline: Hotspot Detection & Classification](#8-ml-pipeline-hotspot-detection--classification)
9. [Enforcement Priority Ranking](#9-enforcement-priority-ranking)
10. [Dashboard Data Pipeline](#10-dashboard-data-pipeline)
11. [Streamlit Dashboard - The LogPose App](#11-streamlit-dashboard--the-logpose-app)
12. [Key Insights & Operational Recommendations](#12-key-insights--operational-recommendations)
13. [Performance Evaluation](#13-performance-evaluation)
14. [Impact, Scalability & Future Scope](#14-impact-scalability--future-scope)
15. [Technical Stack](#15-technical-stack)
16. [Project Structure](#16-project-structure)
17. [Leakage-Free Methodology](#17-leakage-free-methodology)
18. [Installation & Reproduction Guide](#18-installation--reproduction-guide)

---

## 1. Executive Summary

The LogPose Crew built an end-to-end AI pipeline for the Bengaluru Traffic Police (BTP) that transforms raw parking violation records into targeted, data-driven enforcement decisions. Starting from **298,450 anonymized BTP records** spanning January to May 2024, the system:

- **Cleans and recovers 26,009 additional high-quality records** using a leakage-safe device-trust scoring mechanism, expanding the usable dataset from ~115K approved records to 141,409 rows.

- **Engineers 28 machine learning features** across 9 groups - including cyclical time encodings, geohash6-based spatial density, repeat offender scores, and cross-feature interaction terms.

- **Identifies 164 spatial hotspot clusters** using DBSCAN with a haversine metric at 300m radius, with a single dominant cluster (City Market / Majestic belt) accounting for **54.2% of all clustered violations**.

- **Trains a LightGBM + XGBoost soft-voting ensemble** achieving **94.83% weighted accuracy** across 3 enforcement-meaningful target classes.

- **Ranks all 55 police station jurisdictions** by a composite enforcement priority score combining violation frequency, severity, congestion impact, heavy vehicle presence, and habitual offender counts.

- **Delivers all insights** through a 35-feature Streamlit dashboard (The LogPose) with interactive maps, a live ML prediction demo, patrol gap analysis, and plain-English deployment directives.

**Upparpet jurisdiction**, with 18,791 violations (1.5× the second-highest zone), is the unambiguous #1 deployment priority. The top 10 zones together account for **39.7% of all violations** - meaning targeted patrol of those 10 zones beats blanket city-wide coverage.

---

## 2. Problem Statement

### 2.1 Operational Challenge

On-street illegal parking and spillover parking near commercial areas, metro stations, and event venues choke carriageways and intersections throughout Bengaluru. The root problem is not a lack of enforcement resources - it is the **absence of spatial intelligence**. Officers are deployed reactively, responding to complaints rather than proactively patrolling high-density violation zones.

### 2.2 Why Existing Approaches Fall Short

- Enforcement is entirely patrol-based and reactive - no predictive layer exists.
- No heatmap of parking violations vs. congestion impact is available to dispatch operators.
- Prioritization of enforcement zones is manual and experience-driven, not data-driven.
- The raw BTP dataset has a large proportion of NaN-status records that naive pipelines discard, shrinking the effective training pool by ~55%.

### 2.3 Dataset

| Attribute | Value |
|-----------|-------|
| Raw file | `jan_to_may_police_violation_anonymized791b166.csv` |
| Raw records | 298,450 rows, 21 columns |
| Date range | January to May 2024 |
| Source | Bengaluru Traffic Police (BTP) - anonymized |
| Geographic scope | Bengaluru city (lat 12.70-13.40°N, lon 77.35-77.85°E) |

### 2.4 Problem Statement Direction

> *"How can AI-driven parking intelligence detect illegal parking hotspots and quantify their impact on traffic flow to enable targeted enforcement?"*

---

## 3. Solution Overview

The LogPose system addresses the problem through a **four-stage operational loop**:

| Stage | What It Does | Key Output |
|-------|-------------|------------|
| **01 - Detect Hotspots** | DBSCAN spatial clustering on GPS coordinates at 300m radius identifies illegal-parking concentration zones. | 164 hotspot clusters with centroid, size, and density |
| **02 - Quantify Impact** | A composite 1-100 congestion impact score blends violation severity, vehicle congestion weight, frequency, and junction proximity. | Per-record `congestion_score`; per-cluster impact tier |
| **03 - Prioritize Zones** | An enforcement priority ranker aggregates 5 normalized signals across all 55 police-station jurisdictions. | Ranked leaderboard: Upparpet #1 with score 0.6636 |
| **04 - Empower Officers** | A live Streamlit dashboard converts every insight into interactive maps, on-ground directives, and a live ML prediction demo. | 35 interactive dashboard features; 22 pre-computed data artifacts |

The entire pipeline is built on **141,409 real BTP violation records** - not synthetic data - and is fully reproducible from the raw CSV using 6 sequential Python scripts.

---

## 4. System Architecture

The system is organized into four layers: an offline data pipeline, an offline ML pipeline, an offline dashboard data pipeline, and a live Streamlit app. Each stage writes its output to disk - every metric in this documentation traces back to a real script run.

### 4.1 Pipeline Stages

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| Data Cleaning | `data_cleaning.py` | Raw CSV (298,450 rows) | `dataset_cleaned.csv` (~141K rows) + audit JSON |
| Feature Engineering | `feature_engineering.py` | `dataset_cleaned.csv` | `dataset_features.csv` (81 cols) + `label_encoders.pkl` |
| Violation Labeling | `violation_features.py` | `dataset_features.csv` | `primary_violation_final` column + `ml_*` binary matrix |
| ML + Maps + Plots | `parking_intelligence_pipeline.py` | `dataset_features.csv` | DBSCAN clusters, LGB+XGB models, 3 Folium maps, 10 EDA plots |
| Dashboard Artifacts | `dashboard_data_pipeline.py` | All cleaned + model outputs | 22 pre-computed JSON/CSV artifacts |
| Live Dashboard | `app.py` (`streamlit run`) | All artifacts | 35-feature interactive web dashboard |

### 4.2 Directory Structure

```
Round-2/
├── data/
│   ├── jan to may police violation_anonymized791b166.csv  # Raw (298K rows)
│   ├── cleaned/
│   │   ├── dataset_cleaned.csv          # After device-trust recovery
│   │   ├── dataset_features.csv         # Full feature matrix (81 cols)
│   │   └── cleaning_audit.json
│   ├── class_weights.json
│   └── label_encoders.pkl
├── outputs/
│   ├── maps/           # 3 Folium interactive HTML maps
│   ├── plots/          # 10 EDA visualizations
│   ├── model/          # lgbm_model.txt + xgb_model.json
│   ├── dashboard_data/ # 22 pre-computed JSON/CSV artifacts
│   ├── enforcement_priority_ranked.csv
│   ├── hotspot_clusters.csv
│   └── model_summary.json
├── data_cleaning.py
├── feature_engineering.py
├── violation_features.py
├── parking_intelligence_pipeline.py
├── dashboard_data_pipeline.py
└── app.py
```

---

## 5. Data Foundation & Cleaning Pipeline

*Script: `data_cleaning.py` | 6-step leakage-free pipeline*

### 5.1 Pipeline Steps

#### Step 1 - Load Raw CSV

The raw file is read in **50,000-row chunks** using pandas with a dtype map (`float32` for latitude, longitude, `center_code`) to manage memory. Dead columns (`description`, `closed_datetime`, `action_taken_timestamp`) are dropped during ingest.

#### Step 2 - Deduplication

Exact duplicate violation IDs are removed (keep first), and any records with `validation_status == 'duplicate'` are dropped. This step removes ~320 rows to yield **298,130 de-duplicated records**.

#### Step 3 - Device-Trust NaN Recovery (Key Innovation)

The raw dataset contains ~115K approved records and ~183K NaN-validation records. Most pipelines discard the NaN records and train on ~115K rows. Our device-trust scoring does the following:

1. Compute `approval_rate = approved / (approved + rejected)` per `device_id` using **only labeled records**.
2. Tag each row with its device's approval rate.
3. Promote NaN-validation records from devices with `approval_rate >= 0.80` into the training pool.
4. Drop all remaining NaN, rejected, `processing`, and `created1` records.

**Leakage-free design:** NaN records are passive recipients of the trust score - they never influence the computation of their own score.

| Category | Count | Action |
|----------|-------|--------|
| Approved records (ground truth) | 115,400 | Kept - all |
| NaN recovered (device trust >= 0.80) | 26,009 | Kept - promoted to training pool |
| NaN dropped (low-trust devices) | ~156,721 | Dropped |
| Rejected / other | variable | Dropped |
| **Final training pool** | **141,409** | Used for all downstream steps |

#### Step 4 - Geo & Datetime Quality Filters

Rows missing `latitude`/`longitude` or `created_datetime` are dropped. Records outside the Bengaluru bounding box (lat 12.70-13.40°N, lon 77.35-77.85°E) are removed as geo-invalid.

#### Step 5 - Type Casting & Null-Fill

Datetimes are parsed as ISO8601 UTC. String columns are stripped of whitespace. Standard null fills:
- `junction_name` → `'No Junction'`
- `police_station` → `'Unknown'`
- `vehicle_type` → `'OTHERS'`
- `center_code` nulls → filled with per-station median

#### Step 6 - Save

The cleaned dataset is saved to `data/cleaned/dataset_cleaned.csv`. A JSON audit file records row counts at each stage for full traceability.

### 5.2 Data Recovery Summary

```
298,450 Raw Records
   ↓ deduplication
298,130 After Dedup
   ↓ device-trust NaN recovery
141,409 Final Training Pool  (+26,009 NaN records recovered)
```

---

## 6. Feature Engineering

*Script: `feature_engineering.py` | 28 ML features across 9 groups*

### 6.1 Feature Groups

| Group | Features | Description |
|-------|----------|-------------|
| **A - Temporal** | `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `is_night_shift`, `is_weekend`, `time_bucket_enc` | Cyclical sin/cos encoding of hour (period 24) and day-of-week (period 7) eliminates discontinuities at midnight/week boundaries. Night shift = 22:00-06:00 IST. |
| **B - Vehicle** | `vehicle_category_enc`, `vehicle_congestion_weight`, `is_heavy_vehicle` | 22 vehicle types explicitly mapped to 4 groups (TWO_WHEELER, FOUR_WHEELER, HEAVY_VEHICLE, OTHERS). `congestion_weight` is a domain-informed numeric score (HTV = 3.0, Car = 1.5, 2W = 0.5). |
| **C - Violation** | `violation_count`, `max_severity`, `avg_severity`, `total_severity`, `is_multi_label`, `is_high_severity` | Derived from the `violation_type` list field. **Blocked from ML training** to prevent leakage (see Section 17). |
| **D - Junction** | `is_junction`, `is_top_junction` | Binary: `is_junction = 1` if `junction_name` is non-null. Top 10 highest-frequency junctions flagged as `is_top_junction`. |
| **E - Spatial** | `geohash6_density`, `police_station_density` | Coordinates encoded to geohash6 (~1km² cells). Density = min-max normalized count of records per cell. Replaces raw lat/lon in the model. **These are the #1 and #2 most important features by LightGBM gain.** |
| **F - Repeat Offender** | `repeat_offender_score`, `is_repeat_offender`, `is_habitual_offender`, `repeat_offender_tier_enc` | `repeat_offender_score` = count of violations by same `vehicle_number`. Thresholds: `is_repeat_offender` >= 5 violations, `is_habitual_offender` >= 10 violations. |
| **G - Zone Risk** | `police_station_density` | Per-station violation density, frequency-encoded from grouped counts. |
| **H - Interaction** | `station_hour`, `veh_time` | Frequency-encoded cross-features. `station_hour` = how often a given (`police_station`, `hour`) combination appears. `veh_time` = how often a (`vehicle_category`, `time_bucket`) pair appears. |
| **I - Label Encoded** | `vehicle_category_enc`, `police_station_enc`, `junction_name_enc`, `repeat_offender_tier_enc`, `time_bucket_enc` | LabelEncoder applied to 5 categorical columns. Encoders saved to `label_encoders.pkl` for dashboard live prediction. |

### 6.2 Geohash6 Spatial Encoding

Raw latitude and longitude are converted to **geohash6 strings** (6-character geohash = ~1.2km × 0.6km cells) using `pygeohash`. The per-cell violation count is then min-max normalized to produce `geohash6_density`. This approach:

- Avoids the overfitting risk of using raw coordinates directly in the model.
- Captures spatial violation density as a structural feature that generalizes to unseen locations within known cells.
- **Ranks as the #1 most important feature** by LightGBM information gain - evidence the model is learning real spatial patterns, not noise.

### 6.3 Features Excluded from ML

Two feature groups are explicitly blocked from the ML training set:

- **Raw coordinates:** `latitude`, `longitude` - replaced by `geohash6_density`.
- **7 leakage columns:** `avg_severity`, `is_high_severity`, `is_multi_label`, `is_parking_violation`, `max_severity`, `total_severity`, `violation_count` - these are derived from the target label and would constitute data leakage if included.

---

## 7. Violation Label Engineering

*Script: `violation_features.py` | Builds `primary_violation_final` + `violation_category`*

### 7.1 Class Merging Strategy

The raw BTP data contains ~17 distinct violation types with highly imbalanced frequencies. A naive threshold merge would incorrectly absorb operationally distinct classes. Our explicit merge set targets 9 specific classes:

| Class Merged | Records | Reason |
|--------------|---------|--------|
| DOUBLE PARKING | 2,037 | Operationally identical to WRONG PARKING at junctions |
| PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS | 525 | Too few for stable class boundary |
| PARKING OTHER THAN BUS STOP | 242 | Low frequency |
| PARKING OPPOSITE TO ANOTHER PARKED VEHICLE | 486 | Low frequency |
| H T V PROHIBITED | 31 | Extremely rare |
| REFUSE TO GO FOR HIRE | 887 | Taxi regulation, not parking congestion |
| OBSTRUCTING DRIVER | 16 | Extremely rare |
| DEMANDING EXCESS FARE | 240 | Taxi regulation, not parking |
| FAIL TO USE SAFETY BELTS | 8 | Extremely rare, non-parking |

### 7.2 Classes Kept as Standalone

The following classes are **not merged** because they represent operationally distinct enforcement signals:

- **WRONG PARKING** - largest class; core enforcement target
- **NO PARKING** - second largest class; distinct zone violation
- **PARKING IN A MAIN ROAD** - road obstruction, distinct severity tier
- **DEFECTIVE NUMBER PLATE** - 7,848 records; vehicle compliance signal
- **PARKING ON FOOTPATH** - pedestrian safety, distinct signal
- **PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC** - 2,403 records; protected zone
- **PARKING NEAR ROAD CROSSING** - 1,687 records; junction hazard

### 7.3 Three-Class Enforcement Target

For the ML model, the 8 standalone violation classes are further grouped into 3 enforcement-meaningful target classes:

| Enforcement Class | Constituent Violations | Test Support | F1-Score |
|-------------------|----------------------|-------------|---------|
| **GENERIC_PARKING** | WRONG PARKING, NO PARKING, PARKING IN MAIN ROAD, PARKING ON FOOTPATH, OTHER_VIOLATION | 26,421 | 0.972 |
| **SEVERE_OBSTRUCTION** | PARKING NEAR ROAD CROSSING, PARKING NEAR BUSTOP/SCHOOL/HOSPITAL | 1,733 | 0.622 |
| **VEHICLE_COMPLIANCE** | DEFECTIVE NUMBER PLATE | 128 | 0.132 |

### 7.4 Multi-Label Binary Matrix

In addition to the primary target, `violation_features.py` builds a **binary matrix** of 8 `ml_*` columns (e.g., `ml_wrong_parking`, `ml_defective_number_plate`) for optional multi-output modeling.

---

## 8. ML Pipeline: Hotspot Detection & Classification

*Script: `parking_intelligence_pipeline.py` | DBSCAN + LightGBM + XGBoost*

### 8.1 DBSCAN Hotspot Clustering

Spatial clustering is performed using **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) with a haversine metric, which correctly accounts for the curvature of the Earth.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | DBSCAN | Does not require k; handles noise points; finds arbitrary-shaped clusters |
| Metric | Haversine | Correct geodesic distance on a sphere; avoids Euclidean distortion at Bengaluru's latitude |
| Epsilon (radius) | 300m (0.3/6371 radians) | ~3 city blocks; captures a single intersection or commercial zone |
| `min_samples` | 10 | Requires at least 10 violations to form a cluster; eliminates isolated incidents |
| Max input points | 120,000 (subsampled) | Memory-safe for haversine distance matrix computation |

### 8.2 Hotspot Results

| Metric | Value |
|--------|-------|
| Total clusters found | 164 |
| Noise points (non-clustered) | 917 |
| **Dominant cluster (Cluster 0)** | **76,639 violations (54.2%) - City Market / Majestic belt** |
| Cluster 0 centroid | 12.979°N, 77.577°E |
| Cluster 4 | 9,111 violations - Whitefield corridor (12.942°N, 77.696°E) |
| Cluster 1 | 4,819 violations - 12.918°N, 77.625°E |

A handful of zones drive most of the problem - exactly the kind of signal BTP needs to deploy patrols efficiently instead of blanket coverage.

### 8.3 LightGBM + XGBoost Soft-Voting Ensemble

Two gradient boosting models are trained independently and combined via **soft-voting** (average of predicted class probabilities). This approach:

- Reduces variance compared to either individual model.
- Avoids the overfitting tendency of a single model on the dominant class.
- Allows per-class threshold calibration to optimize precision/recall trade-offs for each enforcement class.

**Training Configuration:**

| Setting | LightGBM | XGBoost |
|---------|----------|---------|
| `n_estimators` | 1,000 (with early stopping at 50) | 500 |
| `max_depth` | 8 | 7 |
| `learning_rate` | 0.05 | 0.05 |
| Class imbalance handling | `class_weight` computed on train fold only | `sample_weight` per instance |
| Class weight cap | 30.0 (max) | Equivalent per-sample weight |
| Random state | 42 (both) | 42 (both) |
| Validation | StratifiedKFold split | StratifiedKFold split |

**Per-Class Decision Thresholds:**

| Class | Threshold | Effect |
|-------|-----------|--------|
| GENERIC_PARKING | 0.31 | Lower threshold increases recall for the dominant class |
| SEVERE_OBSTRUCTION | 0.51 | Standard threshold for moderate-frequency class |
| VEHICLE_COMPLIANCE | 0.52 | Conservative threshold given severe class imbalance (128 test samples) |

### 8.4 Congestion Impact Score

Each violation record receives a `congestion_score` on a 0-1 scale (displayed as 0-100 in the dashboard) computed as:

```
congestion_score = (0.4 × severity_norm)
                 + (0.3 × vehicle_weight_norm)
                 + (0.2 × frequency_norm)
                 + (0.1 × repeat_offender_norm)
                 + junction_bonus (0.1 if is_junction)
```

Where each component is min-max normalized to [0, 1] and the repeat offender component is capped at 1.0 (score / 55).

---

## 9. Enforcement Priority Ranking

### 9.1 Priority Score Formula

Each of the **55 police station jurisdictions** receives a `priority_score` aggregated from 5 normalized signals:

- `violation_total_norm` - total violation count in the jurisdiction, normalized to [0, 1]
- `avg_severity_norm` - average per-record severity score, normalized
- `avg_congestion_norm` - average congestion impact score, normalized
- `habitual_flag` - binary presence of habitual offenders (10+ violations by same vehicle)
- `heavy_vehicle_flag` - binary presence of heavy vehicles in the zone

### 9.2 Top 15 Enforcement Zones

| Rank | Police Station | Violations | Heavy Veh. | Habitual | Priority Score |
|------|----------------|-----------|-----------|---------|---------------|
| 1 | **Upparpet** | 18,791 | 802 | 172 | **0.6636** |
| 2 | Whitefield | 886 | 62 | 0 | 0.5837 |
| 3 | Chikkajala | 2,706 | 42 | 0 | 0.5617 |
| 4 | Mahadevapura | 2,828 | 170 | 0 | 0.5554 |
| 5 | HAL Old Airport | 9,284 | 336 | 38 | 0.5251 |
| 6 | Shivajinagar | 12,375 | 171 | 116 | 0.4923 |
| 7 | City Market | 7,505 | 686 | 1 | 0.3814 |
| 8 | Yeshwanthpura | 1,363 | 139 | 0 | 0.3674 |
| 9 | Kengeri | 307 | 25 | 0 | 0.3668 |
| 10 | No Police Station | 129 | 53 | 0 | 0.3607 |
| 11 | Madiwala | 637 | 81 | 0 | 0.3531 |
| 12 | Jeevanbheemanagar | 4,307 | 89 | 0 | 0.3498 |
| 13 | Chamarajpet | 1,523 | 224 | 0 | 0.3475 |
| 14 | Malleshwaram | 10,624 | 141 | 39 | 0.3398 |

**Insight:** Upparpet's #1 rank is driven primarily by raw violation volume (18,791 - nearly 1.5× the next highest zone) combined with 802 heavy vehicle violations and 172 habitual offenders. **Whitefield's #2 rank despite only 886 violations** reflects its high average severity and congestion scores, making it a disproportionate congestion contributor per violation.

---

## 10. Dashboard Data Pipeline

*Script: `dashboard_data_pipeline.py` | 22 pre-computed artifacts*

All dashboard data is pre-computed offline and saved as JSON/CSV files. This eliminates real-time computation on every user interaction, keeping the dashboard instantaneously responsive.

| Artifact | Type | Contents |
|----------|------|----------|
| `peak_time_forecast.csv` | CSV | Top 3 peak violation hours per police station |
| `global_peak_hours.csv` | CSV | City-wide peak hours ranked by violation count |
| `hour_vs_violation_matrix.csv` | CSV | Hour × violation type cross-tabulation matrix |
| `patrol_gap_analysis.csv` | CSV | Geohash5 zones ranked by enforcement gap score (violations - active devices) |
| `reactive_vs_proactive.json` | JSON | Night patrol (reactive) vs. AM/PM rush-hour (proactive) enforcement breakdown |
| `anomaly_detection.json` | JSON | Month-over-month anomaly flags for data collection gaps |
| `scita_sync.json` | JSON | SCITA Smart City integration sync status (sent vs. not sent count and %) |
| `vehicle_lookup_index.csv` | CSV | Per-vehicle lookup index with violation count, severity, and habitual flag |
| `multi_violation_profiles.csv` | CSV | Vehicles with multiple distinct violation types |
| `multi_violation_summary.json` | JSON | Aggregate multi-violation statistics |
| `vehicle_vs_violation_matrix.csv` | CSV | Vehicle type × violation type cross-tabulation |
| `geohash_grid_overlay.csv` | CSV | Geohash6 grid cells with density scores for map overlay |
| `recommendations.json` | JSON | Plain-English enforcement directives per priority zone |
| `time_block_shifts.csv` | CSV | Shift-wise violation breakdown (Night/Morning/Midday/Afternoon/Evening) |
| `habitual_offenders.csv` | CSV | List of vehicles with >= 10 violations |
| `station_reference.csv` | CSV | Police station metadata for dashboard filters |
| `junction_reference.csv` | CSV | Junction metadata and violation counts |
| `quick_view_presets.json` | JSON | IT Corridor, Commercial Belt, and Outskirts quick-view configurations |
| `day_of_week_trends.csv` | CSV | Violation counts by day of week (Thu/Weekend peaks) |
| `validation_status.json` | JSON | Data health: approved_pct, anomaly flags |
| `offence_filter_reference.csv` | CSV | Violation type filter reference for dashboard checkboxes |
| `live_prediction_config.json` | JSON | Feature ranges and encoder states for the live ML prediction demo |

---

## 11. Streamlit Dashboard - The LogPose App

*Script: `app.py` | Command: `streamlit run app.py` | Opens at: `http://localhost:8501`*

### 11.1 Overview

The LogPose dashboard is a **7-page Streamlit application** with a glassmorphism dark-mode UI (toggle to light mode available). It exposes **35 interactive features** across 6 functional categories, all powered by pre-computed data artifacts loaded at startup for zero-latency rendering.

### 11.2 Dashboard Pages & Features

| Page | Features | Highlights |
|------|----------|------------|
| 🏠 **Command Center** | Status strip, KPI cards, glance insights, map carousel, vehicle search, quick navigation | Data health %, SCITA sync %, anomaly banner, AI enforcement directives, 3-map mini carousel |
| 🗺️ **Zone Maps** | Folium heatmap, junction/mid-road toggle, station dropdown, hotspot radius, clickable pins, patrol gap overlay, quick-view presets | 3 embedded Folium maps + geohash6 grid overlay |
| 📊 **Priority Board** | Top 10 zones leaderboard, congestion impact score (1-100), offence severity color map, plain-English recommendations | Composite priority score with 5-signal formula |
| 🚨 **Offender Registry** | Habitual offender alert board, vehicle lookup search, heavy vs. 2-wheeler filter, multi-violation profiles, vehicle × violation matrix | 93.83% SCITA sync rate indicator |
| ⏰ **Shift & Timing** | Peak time forecaster per station, time-block shift filter, hour × violation heatmap, day-of-week trend, reactive vs. proactive gauge | Global peak: city-wide violations peak at specific IST hour |
| 🤖 **ML Explainability** | Feature importance chart, model evaluation dashboard, confusion matrix, live prediction demo (select time/vehicle/station → get class) | 22-feature live inference from saved LGB+XGB models |
| ⚙️ **System Controls** | Data quality toggle, violation type filter, CSV/PDF export, anomaly warning, dark/light theme toggle | Export buttons for `priority_ranked.csv` and `hotspot_clusters.csv` |

### 11.3 Theme & Design

The dashboard uses a dual-mode theme system:

- **Dark mode (default):** Background `#0A0A14`, card `#13132A`, accent red `#E94560`, accent blue `#4CC9F0` - a premium intelligence-operations aesthetic consistent with BTP's operational context.
- **Light mode:** Background `#F0F2F8`, card `#FFFFFF`, accent red `#C0392B`, accent blue `#2980B9` - for daytime field use.

---

## 12. Key Insights & Operational Recommendations

| # | Insight | Operational Action |
|---|---------|-------------------|
| 1 | **93.4% of all violations** are just two types: NO PARKING + WRONG PARKING | Focus officer training and signage resources on these two violation types first - highest ROI |
| 2 | **46.5% of violations** involve two-wheelers - single largest vehicle category | Design two-wheeler-specific enforcement lanes and parking-bay guidance near metro stations and commercial areas |
| 3 | **Thursday and weekends** are the highest-volume days for violations | Increase weekend patrol staffing in commercial belts; reduce weekday roving during low-volume periods |
| 4 | Only **18.8% of enforcement** happens during rush hours (8-10 AM, 5-8 PM) vs. **35.2% at night** | Shift enforcement windows to 8-10 AM and 5-8 PM IST for maximum congestion impact |
| 5 | **Cluster 0** (City Market / Majestic belt) holds 76,639 violations - 54.2% of all clustered records | Permanent static enforcement post or dedicated patrol sub-unit at this cluster |
| 6 | **Upparpet** has 172 habitual offenders (vehicles with 10+ violations) alongside 802 heavy vehicle incidents | Habitual offender vehicle registration alerts; oversized vehicle route restrictions in Upparpet |
| 7 | **Top 10 priority zones** account for 39.7% of all 141,409 violations | Deploy 10-zone targeted patrol instead of blanket city-wide coverage |
| 8 | **SCITA sync rate is 93.83%** | Resolve the 6.17% unsynchronized records before go-live; implement automated retry for failed syncs |

---

## 13. Performance Evaluation

### 13.1 Model Comparison

| Metric | LightGBM | XGBoost | **Ensemble** |
|--------|----------|---------|------------|
| Accuracy | 94.68% | 94.89% | **94.83%** |
| F1 (weighted) | ~94.7% | ~94.8% | **94.79%** |
| Precision (weighted) | - | - | **94.76%** |
| Recall (weighted) | - | - | **94.83%** |
| F1 (macro) | - | - | **57.88%** |

### 13.2 Per-Class Performance

| Class | Precision | Recall | F1-Score | Test Support |
|-------|-----------|--------|---------|-------------|
| GENERIC_PARKING | 97.16% | 97.35% | **97.26%** | 26,421 |
| SEVERE_OBSTRUCTION | 63.21% | 61.28% | **62.23%** | 1,733 |
| VEHICLE_COMPLIANCE | 13.08% | 13.28% | **13.18%** | 128 |

### 13.3 Honest Disclosure

`VEHICLE_COMPLIANCE` (128 test samples) has a low F1 of **0.132** due to severe class imbalance - GENERIC_PARKING makes up 93.4% of violations across the original 8 classes. We disclose this proactively rather than mask it with weighted metrics alone.

The macro F1 of 57.88% vs. weighted F1 of 94.79% directly reflects this imbalance. Mitigation strategies considered and partially implemented:

- Class weight cap at **30.0** to avoid extreme weighting for VEHICLE_COMPLIANCE.
- Per-class decision threshold calibration (VEHICLE_COMPLIANCE threshold: 0.52).
- Future: synthetic minority oversampling (SMOTE) or focal loss for this class.

### 13.4 Spatial Clustering Performance

| Metric | Value |
|--------|-------|
| Total clusters (DBSCAN) | 164 |
| Noise points | 917 |
| Dominant cluster size | 76,639 violations (54.2%) |
| Clustering algorithm | DBSCAN with haversine metric, eps=300m, min_samples=10 |

All numbers are generated by `parking_intelligence_pipeline.py` - reproducible end-to-end from the raw CSV, not hand-tuned.

---

## 14. Impact, Scalability & Future Scope

### 14.1 Immediate Impact

- **Top 10 priority zones = 39.7% of all 141,409 violations** - targeted patrol of 10 zones beats blanket city-wide coverage.
- **Plain-English deployment directives** built into the dashboard - no data science expertise required from field officers.
- **93.83% SCITA Smart City sync rate** - integration-ready for Bengaluru's city-wide monitoring infrastructure.
- **Habitual offender tracking** across 141,409 records allows proactive registration-level interventions.

### 14.2 Scalability

- **City-agnostic pipeline:** swap the source CSV and re-run the 6 scripts - no code changes required for a different city's BTP data.
- **Geohash6 + DBSCAN** generalize to any GPS-tagged violation dataset, regardless of city geometry.
- **Dashboard architecture** supports new police station zones and vehicle types with zero code changes - driven by data, not hardcoded logic.
- **MongoDB `push_to_mongo.py`** supports cloud-hosted deployment for multi-district access.

### 14.3 Future Scope

| Feature | Status | Description |
|---------|--------|-------------|
| Drone / satellite imagery verification | Phase 2 | Real-time hotspot ground-truth via aerial feeds |
| Live SCITA API sync | Phase 2 | Currently 93.83% offline-sync ready; live API call replaces batch upload |
| Multi-city rollout | Phase 3 | Chennai, Hyderabad, Mumbai BTP data via same pipeline |
| SMOTE / focal loss for VEHICLE_COMPLIANCE | Model v2 | Address severe class imbalance in minority class |
| Real-time violation ingestion | Phase 2 | Streaming pipeline replacing batch CSV ingest |
| Officer mobile app | Phase 3 | React Native or Flutter front-end consuming the dashboard API |

*Note: Drone/satellite integration and live SCITA sync are deliberate next-phase scope items - not gaps in the current prototype.*

---

## 15. Technical Stack

| Layer | Technology | Version / Notes |
|-------|-----------|----------------|
| Data Processing | pandas, numpy | pandas >= 2.0, numpy >= 1.24 |
| ML - Gradient Boosting | LightGBM + XGBoost | lgbm >= 4.0, xgb >= 2.0; soft-voting ensemble |
| ML - Clustering | scikit-learn DBSCAN | sklearn >= 1.3; haversine metric |
| Spatial Encoding | pygeohash | Geohash5 (patrol gap) + geohash6 (ML features) |
| Maps | Folium | 3 interactive HTML maps (heatmap, clusters, night vs. day) |
| Visualization | Plotly, Matplotlib, Seaborn | 10 EDA plots + 35 dashboard charts |
| Dashboard | Streamlit | >= 1.32.0; 7 pages, 35 features |
| Database | MongoDB Atlas (pymongo) | 26 collections; certifi for TLS |
| Language | Python | 3.10+ |
| Dev Environment | Ubuntu 24.04 (VirtualBox on Windows 10) | Development environment |

---

## 16. Project Structure

| File / Folder | Purpose |
|---------------|---------|
| `data_cleaning.py` | Step 1: Raw CSV → cleaned dataset. Device-trust NaN recovery, deduplication, geo filtering. |
| `feature_engineering.py` | Step 2: Cleaned data → 28 ML features across 9 groups. Label encoders saved to `.pkl`. |
| `violation_features.py` | Step 3: Builds `primary_violation_final` target column and `ml_*` binary matrix. |
| `parking_intelligence_pipeline.py` | Step 4: DBSCAN clustering, LGB+XGB ensemble, congestion scoring, enforcement priority ranking, 3 Folium maps, 10 EDA plots. |
| `dashboard_data_pipeline.py` | Step 5: 22 pre-computed JSON/CSV data artifacts for the dashboard. |
| `app.py` | Step 6: 7-page Streamlit dashboard. 35 interactive features. |
| `push_to_mongo.py` | Optional: Pushes all 26 data collections to MongoDB Atlas. |
| `data/cleaned/dataset_cleaned.csv` | Cleaned and device-trust-recovered dataset (141,409 rows). |
| `data/cleaned/dataset_features.csv` | Full feature matrix (81 columns including all 28 ML features). |
| `outputs/model/lgbm_model.txt` | Trained LightGBM model (text format). |
| `outputs/model/xgb_model.json` | Trained XGBoost model (JSON format). |
| `outputs/enforcement_priority_ranked.csv` | Per-station enforcement priority scores and metadata. |
| `outputs/hotspot_clusters.csv` | DBSCAN cluster centroids and point counts. |
| `outputs/maps/01_congestion_heatmap.html` | Folium violation density heatmap. |
| `outputs/maps/02_hotspot_clusters_priority.html` | Folium hotspot cluster map with priority color coding. |
| `outputs/maps/03_night_vs_day.html` | Folium night vs. day enforcement pattern map. |
| `outputs/dashboard_data/` | 22 pre-computed JSON/CSV artifacts. |
| `outputs/model_summary.json` | Model metrics, feature list, target classes, decision thresholds. |
| `feature_metadata.json` | 28 ML feature names, excluded columns, class weight cap, generated timestamp. |

---

## 17. Leakage-Free Methodology

This section documents every data-leakage safeguard in the pipeline. **All safeguards are code-enforced and verifiable in the repository.**

### 17.1 The 7 Blocked Leakage Columns

The following 7 columns are derived from the target label (violation types) and would constitute target leakage if included in the ML training set. They are explicitly excluded from `NUMERIC_FEATURES` in `feature_engineering.py`:

| Blocked Column | Why It's Leakage |
|----------------|----------------|
| `avg_severity` | Derived from violation severity lookup - encodes the target |
| `is_high_severity` | Binary flag derived from `avg_severity` threshold - target-derived |
| `is_multi_label` | Indicates multiple violation types - directly encodes the target |
| `is_parking_violation` | Binary flag for parking category - encodes target class membership |
| `max_severity` | Max severity across violation list - target-derived |
| `total_severity` | Sum of severity scores - target-derived |
| `violation_count` | Count of violations in the record - target-derived |

### 17.2 Additional Safeguards

| Safeguard | Implementation |
|-----------|----------------|
| Train-fold-only class weighting | `class_weight` dictionary fitted exclusively on `X_train`, `y_train` - never on validation or test data |
| Geohash density as structural feature | `geohash6_density` replaces raw lat/lon - it is a count of records per spatial zone, not derived from the target label |
| Device-trust leakage isolation | Approval rates for NaN recovery computed only from labeled (approved/rejected) records; NaN rows never influence their own score |
| Per-class threshold calibration | Decision thresholds tuned on validation set predictions, not on test set |
| NaN record passivity | NaN-validation records in Step 3 are passive recipients of trust scores; their own status is never used to compute the score |

---

## 18. Installation & Reproduction Guide

### 18.1 Prerequisites

- Python 3.10 or higher
- pip (standard Python package manager)
- Raw BTP CSV file: `jan to may police violation_anonymized791b166.csv`
- MongoDB Atlas account *(optional - only required for `push_to_mongo.py`)*

### 18.2 Install Dependencies

```bash
pip install pandas numpy xgboost lightgbm scikit-learn folium streamlit plotly
pip install pygeohash tqdm seaborn matplotlib pymongo certifi python-dotenv
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 18.3 Run Pipeline (in order)

**Step 1** - Clean raw data (device-trust NaN recovery):
```bash
python data_cleaning.py
```

**Step 2** - Engineer 28 ML features:
```bash
python feature_engineering.py
```

**Step 3** - Build violation target labels:
```bash
python violation_features.py
```

**Step 4** - Train ML models + generate maps and plots:
```bash
python parking_intelligence_pipeline.py
```

**Step 5** - Pre-compute dashboard data artifacts:
```bash
python dashboard_data_pipeline.py
```

**Step 6** - Launch the interactive dashboard:
```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**.

### 18.4 MongoDB Upload (Optional)

Create a `.env` file with your MongoDB URI and database name:

```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/
MONGO_DB=btp_db
```

Then run:

```bash
python push_to_mongo.py
```

This uploads all **26 collections** (22 dashboard artifacts + enforcement priority, hotspot clusters, model summary, and dataset features) to MongoDB Atlas with chunked streaming for large files.

---

## 📝 Final Notes

> *Built on 298,450 real BTP records · 141,409 clean rows · 164 hotspots · 94.83% accuracy · one clear answer for where to enforce next.*

**Flipkart GridLock Hackathon 2.0 - Round 2 | The LogPose Crew**

GitHub: [github.com/HimanshuMahajan2111/The-LogPose-Crew](https://github.com/HimanshuMahajan2111/The-LogPose-Crew)
