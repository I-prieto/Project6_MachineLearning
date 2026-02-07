# Optimal Service Level Selection for Aircraft Parts Shipments
### Master's in Data Analytics - Machine Learning Project

## 1. Project Overview

This project develops a machine learning solution to predict the optimal shipping service level for aircraft parts, minimizing costs while ensuring timely delivery.

**Our Company:** Aircraft parts distributor contracting shipping services

**The Challenge:** When booking a shipment, we choose between three service levels:
- **Normal (ROV)** - Cheapest, slowest
- **Urgent (CRV)** - Mid-price, mid-speed
- **Critical (AOV)** - Most expensive, fastest

Choosing too expensive wastes budget. Choosing too cheap risks late delivery and production delays.

---

## 2. Business Problem

**Historical analysis of 3,630 shipments reveals:**

- **38.2% of shipments over-serviced** (1,385 orders) — paid for premium service when a cheaper tier would have sufficed
- **Only 17.7% of shipments met their contracted SLA** (641 orders) — expensive services don't guarantee on-time delivery
- **94.3% of Critical orders** could have been downgraded (1,149 of 1,219)
- **72.6% of Urgent orders** could have been downgraded (236 of 325)

**Key Insight:** Even paying for Critical service doesn't guarantee on-time delivery. We need data-driven decisions.

---

## 3. Data

| Dataset | Records | Description |
|---------|---------|-------------|
| Orders.csv | 3,695 → 3,630 | Shipment transactions with timestamps, zones, service levels |
| LeadtimeService.csv | 18 | SLA targets per Zone + Service Level |
| airports.csv | 84,490 | Airport-to-country mapping for route enrichment |

**Zone Distribution & SLA Targets (hours):**

| Zone | Description | Critical (AOV) | Urgent (CRV) | Normal (ROV) |
|------|-------------|----------------|--------------|--------------|
| Z1 | ES (Spain) | 4h | 16h | 24h |
| Z2 | EEZ | 8h | 24h | 36h |
| Z3 | EUR | 24h | 48h | 72h |
| Z4 | N.A. | 24h | 48h | 72h |
| Z5 | LATAM | 24h | 48h | 72h |
| Z6 | Others | 24h | 48h | 72h |

---

## 4. Methodology

### 4.1 Data Pipeline
1. Merged 3 datasets (Orders + LeadtimeService + Airports)
2. Cleaned invalid records (33 service level errors, 31 timestamp errors)
3. Engineered `leadtime_real_hours` (actual delivery time) and `leadtime_ml_hours` (capped at 99th percentile for outlier handling)
4. Created `on_time` binary flag comparing actual vs contracted SLA

### 4.2 EDA - Downgrade Analysis
Retrospective analysis to identify over-serviced orders using zone-specific SLA targets:
- If SLA met → keep contracted level
- If SLA not met → recommend closest lower tier (Critical → Urgent → Normal)
- If out of all SLAs → recommend Normal (minimum cost for service failure)
- Never recommend upgrading (faster-than-expected delivery is a win)

### 4.3 Feature Engineering
**Features available at booking time:**
- **Numerical:** Gross weight (Gwgt), Pieces (Pcs)
- **Categorical:** Zone, DGR (dangerous goods), Direction, Route (grouped, rare routes < 20 occurrences → "OTHER")
- **Temporal:** Day of week, Month, Hour (extracted from notification timestamp)

**Target variable:** `sla_recommended` — the minimum sufficient service level

### 4.4 Preprocessing
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- ColumnTransformer pipeline
- Train/Test split: 80/20 with stratification (Train: 2,904 | Test: 726)

### 4.5 PCA Analysis
Principal Component Analysis on numerical features:
- PC1 captures shipment physical characteristics (weight-driven)
- PC2 represents delivery time behavior

---

## 5. Models & Results

Four models were trained and evaluated:

| Model | Accuracy | Macro Recall | Normal Recall | Urgent Recall | Critical Recall |
|-------|----------|--------------|---------------|---------------|-----------------|
| Dummy (Baseline) | 0.820 | 0.333 | 1.000 | 0.000 | 0.000 |
| Logistic Regression | 0.519 | 0.498 | 0.531 | 0.462 | 0.500 |
| Random Forest | 0.745 | 0.573 | 0.803 | 0.487 | 0.429 |
| **Gradient Boosting** | **0.829** | **0.450** | **0.960** | **0.248** | **0.143** |

### Selected Model: Gradient Boosting (83% accuracy)

**Classification Report:**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.86 | 0.96 | 0.91 |
| Urgent | 0.59 | 0.25 | 0.35 |
| Critical | 0.18 | 0.14 | 0.16 |

**Why Gradient Boosting:** Highest overall accuracy (83%) and best Normal class recall (96%), prioritizing cost-efficient classification. Outperforms baseline and all other models on the majority class.

**Limitation:** Severe class imbalance (Normal 82%, Urgent 16%, Critical 2%) makes minority class detection challenging.

---

## 6. Key Findings

1. **38% of shipments are over-serviced** — significant cost savings opportunity
2. **Only 18% of orders meet contracted SLA** — service level selection alone doesn't guarantee performance
3. **Gradient Boosting at 83% accuracy** correctly identifies Normal shipments 96% of the time
4. **Zone is the most impactful feature** for service level prediction, followed by weight
5. **Class imbalance** remains the main challenge — Critical orders represent only 2% of data

---

## 7. Repository Structure

```
├── ML.ipynb                # Main analysis notebook
│   ├── Data Loading & Cleaning
│   ├── Target Engineering
│   ├── Exploratory Data Analysis
│   ├── PCA Analysis
│   ├── Model Training (Dummy, LogReg, RF, GB)
│   └── Model Evaluation & Comparison
│
├── Orders.csv              # Shipment data (3,695 records)
├── LeadtimeService.csv     # SLA targets (18 records)
├── airports.csv            # Airport mapping (84,490 records)
└── README.md               # This file
```

---

## 8. Technical Requirements

```
pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter
```

---

**Academic Project:** Master's in Data Analytics
**Focus:** Machine Learning for Business Optimization
