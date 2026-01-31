# 📦 ML Project — Service Level Optimization for SLA Compliance

## 1. Project Overview

This project analyzes shipment punctuality to support data-driven decisions when selecting the most appropriate service level for each shipment.

The operational dataset includes three service level codes:

- **ROV** → Normal service  
- **CRV** → Urgent service  
- **AOV** → Critical service  

Although higher service levels are expected to improve delivery performance, they also imply higher operational costs. This project evaluates whether escalating from Normal to Urgent or Critical actually improves SLA compliance across different zones and shipment profiles.

The final objective is to determine **when service escalation adds real value and when it does not**.

---

## 2. Business Problem

Operational teams often assume that selecting **AOV (Critical)** or **CRV (Urgent)** guarantees SLA compliance. However, historical data suggests that this assumption does not hold consistently across all zones and shipment characteristics.

Key business questions addressed:

- Does upgrading from ROV to CRV or AOV improve delivery punctuality?
- Are there zones where service escalation does not significantly reduce lead time?
- Can SLA compliance be predicted in advance for each service option?
- Can we recommend the optimal service level per shipment based on data?

---

## 3. Problem Formulation

This project is framed as a **supervised Machine Learning problem** with two complementary objectives.

### 3.1 Regression Task

**Target variable:**  
- `leadtime_real` → actual delivery lead time (in hours or days)

**Goal:**  
Predict the expected lead time of a shipment given its zone, service level and operational characteristics.

---

### 3.2 Classification Task (Derived)

**Target variable:**  
- `on_time`  
  - 1 if `leadtime_real ≤ SLA`  
  - 0 if `leadtime_real > SLA`

**Goal:**  
Estimate the probability that a shipment will comply with its SLA under each service level.

---

## 4. Data Description

The dataset is built by integrating and cleaning multiple operational data sources.

### Service Level Encoding

The categorical variable `Service Level` uses the original operational codes:

- `ROV` → Normal  
- `CRV` → Urgent  
- `AOV` → Critical  

These values are preserved and encoded during preprocessing.

---

### Key Features (X)

Only variables known **before shipment execution** are used as predictors:

- Zone  
- Service Level (ROV, CRV, AOV)  
- Weight metrics (Gross Weight, Chargeable Weight)  
- Number of pieces  
- Dangerous Goods indicator (DGR)  
- Origin country or airport  
- Destination country or airport  
- Temporal features derived from notification timestamp:
  - Day of week
  - Month

### Target Construction

- `leadtime_real` is calculated as the difference between notification timestamp and actual delivery timestamp.
- `on_time` is derived by comparing real lead time against the SLA associated with each shipment.

Post-delivery variables are excluded to avoid data leakage.

---

## 5. Exploratory Data Analysis

Exploratory analysis is conducted prior to modeling to validate the business hypothesis.

Key analyses include:

- Average lead time by Zone and Service Level  
- SLA compliance rate by Zone and Service Level  
- Comparison of ROV vs CRV vs AOV performance  
- Identification of zones where escalation does not improve punctuality  

This step demonstrates whether higher service levels systematically translate into better outcomes.

---

## 6. Modeling Approach

### 6.1 Preprocessing

A unified preprocessing pipeline is applied using `ColumnTransformer`:

- Numerical features are standardized  
- Categorical features are one-hot encoded  
- Preprocessing is embedded within ML pipelines to ensure reproducibility and prevent data leakage  

---

### 6.2 Regression Models

Models evaluated include:

- Dummy Regressor (baseline)  
- Linear and Ridge Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor (if applicable)  

**Evaluation metrics:**

- MAE  
- RMSE  
- R²  

---

### 6.3 Classification Models (Optional Extension)

Models evaluated include:

- Dummy Classifier  
- Logistic Regression  
- Random Forest Classifier  

**Evaluation metrics:**

- Recall (delay detection)  
- F1-score  
- Confusion matrix  

---

## 7. Service Level Simulation

For a given shipment profile, three service scenarios are simulated:

- **ROV (Normal)**  
- **CRV (Urgent)**  
- **AOV (Critical)**  

For each scenario, the models predict:

- Expected lead time  
- Probability of SLA compliance  

This allows a direct comparison of service levels for the same shipment.

---

## 8. Decision Framework

A rule-based decision framework is derived from model outputs:

- Select **ROV** if SLA compliance probability meets the defined threshold  
- Escalate to **CRV** only if ROV does not meet the threshold  
- Use **AOV** when neither ROV nor CRV provide sufficient reliability  

This framework balances punctuality and operational efficiency.

---

## 9. Project Outputs

- End-to-end ML pipeline  
- Lead time prediction model  
- SLA compliance probability model  
- Service level simulation tool  
- Data-driven service selection recommendations  

---

## 10. Expected Conclusions

- Escalating to Critical service does not universally guarantee SLA compliance  
- Certain zones show limited benefit from service escalation  
- Predictive modeling enables more efficient service selection  
- Data-driven decisions can reduce unnecessary operational over-escalation  

---

## 11. Next Steps

- Incorporate cost data to extend the model into cost-benefit optimization  
- Deploy the model as a decision support tool  
- Integrate predictions into BI dashboards for operational monitoring  
