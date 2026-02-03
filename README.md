# 📦 Optimal Service Level Selection for Aircraft Parts Shipments
### Master's in Data Analytics - Machine Learning Project

## 1. Project Overview

This project develops a machine learning solution to predict the optimal shipping service level for aircraft parts, minimizing costs while ensuring timely delivery to avoid operational disruptions.

**Our Company:** Aircraft parts distributor contracting shipping services

**The Business Challenge:**

When booking a shipment, we must choose between three service levels:
- **Normal (ROV)** - Cheapest, but slowest
- **Urgent (CRV)** - Mid-price, mid-speed
- **Critical (AOV)** - Most expensive, fastest

**The Risks:**
- **Select too cheap (Normal when Critical needed)** → Late delivery → Production delays, customer complaints, lost sales
- **Select too expensive (Critical when Normal sufficient)** → Overpay for unnecessary speed → Wasted budget

**The Goal:**

Build a machine learning model that predicts the **minimum service level** needed to ensure on-time delivery for each shipment, optimizing cost while avoiding late deliveries.

---

## 2. Business Problem

### 2.1 The Cost of Wrong Decisions

**Historical Analysis (3,630 shipments) reveals:**

**Over-spending: 42.5% of shipments (1,544 orders) used more expensive service levels than needed**
- 1,219 Critical shipments could have used Normal → overpaid by 2 tiers
- 325 Urgent shipments could have used Normal → overpaid by 1 tier
- Significant budget wasted on unnecessary service escalation

**Under-performance: Current service level selection fails to ensure timely delivery**
- Zone Z1 Critical: **0% on-time rate** (0/10 met SLA)
- Zone Z2 Critical: **6.6% on-time rate** (54/813 met SLA)
- Zone Z2 Normal: **34.1% on-time rate** (358/1050 met SLA)

**Key Insight:** Even expensive service levels don't guarantee on-time delivery. We need better prediction to make informed decisions.

### 2.2 Current Decision Process Issues

**No data-driven selection method:**
- Decisions based on intuition or default rules
- Zone characteristics not considered
- Weight and destination patterns ignored
- Temporal factors (day of week, seasonality) not accounted for

**Result:**
- 42.5% overpayment rate
- Low SLA compliance across all service levels
- Unpredictable delivery performance

---

## 3. Machine Learning Objective

**Problem Type:** Multi-class Classification (with supporting Regression analysis)

### 3.1 Classification Approach (Primary)

**Input:** Shipment characteristics known at booking time
- Zone (Z1-Z6)
- Weight (Gross and Chargeable)
- Number of pieces
- Origin and destination countries
- Dangerous goods flag
- Shipment direction (Export/Import/Cross-trade)
- Temporal factors (day of week, month, hour)

**Output:** Predicted service level (Normal/Urgent/Critical) that will meet delivery requirements

**Target Label Creation:**
For each historical shipment, determine the **minimum service level** that would have delivered on time:
- If delivered within Normal SLA → label as "Normal"
- If exceeded Normal but within Urgent SLA → label as "Urgent"
- If exceeded Urgent but within Critical SLA → label as "Critical"
- If exceeded all SLAs → label as "Normal" (baseline, since even Critical failed)

### 3.2 Success Criteria

1. **Cost Reduction:** Reduce over-serviced orders from 42.5% to <20%
2. **Reliability:** Achieve >85% on-time delivery rate with predicted service levels
3. **Model Confidence:** Provide probability scores to flag uncertain predictions for manual review

---

## 4. Data

### 4.1 Data Sources & Processing

**Orders.csv** (3,695 → 3,630 after cleaning)
- Shipment transaction records with timestamps, zones, service levels
- Cleaning: Removed 65 records (33 invalid service levels, 31 timestamp errors, 1 missing data)

**LeadtimeService.csv** (18 records)
- SLA targets for each Zone + Service Level combination
- Defines success criteria: shipment meets SLA if `actual_time ≤ target_time`

**airports.csv** (84,490 records)
- Maps airport codes to countries for origin/destination enrichment

### 4.2 Service Level Distribution

| Service Level | Orders | Percentage | Avg Lead Time |
|---------------|--------|------------|---------------|
| Normal (ROV)  | 2,099  | 57.8%      | ~70h          |
| Critical (AOV)| 1,219  | 33.6%      | ~62h          |
| Urgent (CRV)  | 312    | 8.6%       | ~51h          |

**Observation:** Despite paying for Critical, average lead time only slightly better than Normal.

### 4.3 Zone Distribution & SLA Targets

| Zone | Description | Volume | Critical SLA | Urgent SLA | Normal SLA |
|------|-------------|--------|--------------|------------|------------|
| Z1   | ES (Spain)  | 21 (0.6%) | 4h        | 16h        | 24h        |
| Z2   | EEZ         | 2,080 (57.3%) | 8h    | 24h        | 36h        |
| Z3   | EUR         | 662 (18.2%) | 24h     | 48h        | 72h        |
| Z4   | N.A.        | 719 (19.8%) | 24h     | 48h        | 72h        |
| Z5   | Other       | 139 (3.8%) | 24h      | 48h        | 72h        |
| Z6   | Pending     | 41 (1.1%) | 100h      | 100h       | 100h       |

**Key Insight:** Z2 represents majority of shipments (57.3%) with tight SLA targets (8h/24h/36h).

### 4.4 Lead Time Statistics

- **Mean:** 83.87 hours (~3.5 days)
- **Median:** 65.62 hours (~2.7 days)
- **Min:** 0.12 hours (7 minutes)
- **Max:** 670.78 hours (28 days)
- **99th percentile:** 337.09 hours (14 days) - used for ML target capping

---

## 5. Completed Work

### 5.1 Data Integration & Cleaning ✅
- Merged 3 datasets (Orders, LeadtimeService, airports)
- Cleaned 65 invalid records
- Enriched with origin/destination countries

### 5.2 Target Engineering ✅
- **`leadtime_real_hours`:** Actual delivery time (for analysis)
- **`leadtime_ml_hours`:** Capped at 99th percentile for modeling (handles outliers)
- **`on_time`:** Binary indicator if shipment met contracted SLA
- **`sla_required`:** Minimum service level that would have met actual delivery time
- **`sla_recommended`:** Recommended service level considering zone-specific targets
- **`downgrade_possible`:** Flag indicating over-serviced orders (42.5% of dataset)

### 5.3 Exploratory Data Analysis ✅

**SLA Compliance by Zone + Service Level:**
- Calculated on-time rates for all zone-service combinations
- Identified zones where expensive services don't improve reliability

**Downgrade Opportunity Analysis:**
- 42.5% of shipments over-serviced (1,544 orders)
- 1,219 Critical → Normal (skipped Urgent entirely)
- 325 Urgent → Normal
- Zero Critical → Urgent (when Critical fails, delivery too slow even for Urgent)

**Performance Comparison:**
- Critical doesn't always outperform Normal
- Zone-specific patterns identified (e.g., Z2 has low compliance across all levels)

---

## 6. Proposed ML Approach

### 6.1 Problem Formulation

**Approach: Multi-class Classification**

Directly predict which service level (Normal/Urgent/Critical) will deliver on time for each shipment.

**Why Classification over Regression:**
- Final decision is categorical (which service level to book)
- Model learns directly what we need to decide
- Avoids two-step process (predict time → convert to service level)

**Target Label Engineering:**

```python
def create_optimal_service_label(row):
    """
    For each historical shipment, determine the MINIMUM service level
    that would have delivered within SLA.

    This becomes our ground truth: what we SHOULD have booked.
    """
    actual_delivery_time = row['leadtime_real_hours']
    zone = row['ZONE']

    # Get SLA targets for this zone
    normal_sla = get_sla_target(zone, 'Normal')
    urgent_sla = get_sla_target(zone, 'Urgent')
    critical_sla = get_sla_target(zone, 'Critical')

    # Assign minimum sufficient service level
    if actual_delivery_time <= normal_sla:
        return 'Normal'  # Cheapest option would have worked
    elif actual_delivery_time <= urgent_sla:
        return 'Urgent'  # Needed mid-tier
    elif actual_delivery_time <= critical_sla:
        return 'Critical'  # Needed premium
    else:
        return 'Normal'  # Even Critical wouldn't have worked → default to cheapest

# Apply to dataset
df['optimal_service_level'] = df.apply(create_optimal_service_label, axis=1)
```

### 6.2 Feature Engineering

**Features available at booking time:**

**Categorical:**
- `ZONE` - Geographic zone (Z1-Z6) - **Most important**
- `Direction` - Export/Import/Cross-trade
- `Type` - ROA/AIR/DSx (transport type)
- `DGR` - Dangerous goods flag
- `iso_country_origin` - Origin country code
- `iso_country_destination` - Destination country code

**Numerical:**
- `Gwgt` - Gross weight (kg)
- `Cwgt` - Chargeable weight (kg)
- `Pcs` - Number of pieces

**Temporal (derived from notification timestamp):**
- `day_of_week` - 0-6 (Monday-Sunday)
- `month` - 1-12 (seasonality)
- `hour` - 0-23 (time of day)

**Feature Preprocessing:**
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),
         ['Gwgt', 'Cwgt', 'Pcs', 'day_of_week', 'month', 'hour']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         ['ZONE', 'Direction', 'Type', 'DGR',
          'iso_country_origin', 'iso_country_destination'])
    ])
```

### 6.3 Train-Test Split Strategy

```python
from sklearn.model_selection import train_test_split

# Stratify by ZONE to ensure all zones represented in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=df['ZONE'],  # Maintain zone distribution
    random_state=42
)

# Result: Train=2,904 orders, Test=726 orders
```

### 6.4 Models to Evaluate

**1. Baseline - DummyClassifier**
```python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
# Expected accuracy: ~58% (majority class "Normal")
```

**2. Logistic Regression (Multi-class)**
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced',  # Handle class imbalance
    max_iter=1000
)
```
- **Pros:** Fast, interpretable coefficients, good baseline
- **Cons:** Assumes linear relationships

**3. Random Forest Classifier**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
```
- **Pros:** Handles non-linear patterns, feature importance, robust to overfitting
- **Cons:** Can be slow, less interpretable than logistic regression

**4. Gradient Boosting Classifier**
```python
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
```
- **Pros:** Typically best performance, sequential error correction
- **Cons:** Prone to overfitting if not tuned, longer training time

**5. XGBoost (Advanced)**
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=1,  # Adjust for class imbalance
    eval_metric='mlogloss',
    random_state=42
)
```
- **Pros:** State-of-the-art performance, handles missing values, built-in CV
- **Cons:** More complex, requires hyperparameter tuning

### 6.5 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Example for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='f1_weighted',  # Weighted F1-score
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

grid_search.fit(X_train_processed, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

### 6.6 Evaluation Metrics

**Primary Metrics:**

**1. Weighted F1-Score** (Primary metric for model selection)
```python
from sklearn.metrics import f1_score
f1_weighted = f1_score(y_test, y_pred, average='weighted')
```
- Balances precision and recall across all classes
- Weighted by class frequency (accounts for imbalance)

**2. Confusion Matrix** (Error pattern analysis)
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=['Normal', 'Urgent', 'Critical'])
disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Urgent', 'Critical'])
disp.plot()

# Key analysis:
# - Diagonal = correct predictions
# - Below diagonal = overestimated (predicted higher than needed) → Cost waste
# - Above diagonal = underestimated (predicted lower than needed) → Late delivery risk
```

**3. Class-Specific Metrics** (Per service level performance)
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred,
                           target_names=['Normal', 'Urgent', 'Critical']))

# Output:
#              precision    recall  f1-score   support
#    Normal       0.85      0.88      0.86       420
#    Urgent       0.72      0.65      0.68        63
#  Critical       0.68      0.70      0.69       243
```

**Business-Oriented Metrics:**

**4. Cost-Benefit Analysis** (Custom metric)
```python
def calculate_business_impact(y_true, y_pred):
    """
    Calculate cost impact of predictions.

    Assumptions:
    - Overestimating (predict higher): Waste $X per tier
    - Underestimating (predict lower): Risk late delivery, lose $Y (opportunity cost)
    """
    service_rank = {'Normal': 1, 'Urgent': 2, 'Critical': 3}
    tier_cost = 100  # $ per service tier difference
    late_delivery_cost = 500  # $ opportunity cost if predict too low

    total_overspend = 0
    total_late_risk = 0

    for true, pred in zip(y_true, y_pred):
        true_rank = service_rank[true]
        pred_rank = service_rank[pred]

        if pred_rank > true_rank:  # Overestimated
            total_overspend += (pred_rank - true_rank) * tier_cost
        elif pred_rank < true_rank:  # Underestimated
            total_late_risk += late_delivery_cost

    return {
        'total_overspend': total_overspend,
        'total_late_risk': total_late_risk,
        'net_cost': total_overspend + total_late_risk
    }
```

**5. Calibration Analysis** (Confidence reliability)
```python
from sklearn.calibration import calibration_curve

# For each class, check if predicted probabilities match actual outcomes
prob_true, prob_pred = calibration_curve(
    y_test_binary, y_prob_positive, n_bins=10
)
```

### 6.7 Feature Importance Analysis

```python
import matplotlib.pyplot as plt
import pandas as pd

# Extract feature importance (for tree-based models)
feature_importance = best_model.feature_importances_
feature_names = preprocessor.get_feature_names_out()

# Create DataFrame and sort
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(10, 8))
plt.barh(range(20), importance_df['importance'][:20])
plt.yticks(range(20), importance_df['feature'][:20])
plt.xlabel('Importance')
plt.title('Top 20 Features for Service Level Prediction')
plt.tight_layout()
plt.show()

# Expected top features:
# 1. ZONE (geographic patterns)
# 2. Gwgt/Cwgt (weight impacts delivery speed)
# 3. iso_country_destination (customs, distance)
# 4. day_of_week (weekday vs weekend)
```

### 6.8 Model Deployment Strategy

**Prediction Function:**

```python
def predict_optimal_service_level(shipment_data):
    """
    Predict optimal service level for a new shipment.

    Input: shipment_data (dict or DataFrame row)
    Output: dict with recommendation and confidence
    """
    # Preprocess input
    X_new = preprocessor.transform(shipment_data)

    # Get prediction and probabilities
    predicted_service = best_model.predict(X_new)[0]
    probabilities = best_model.predict_proba(X_new)[0]

    # Map probabilities to service levels
    prob_dict = {
        'Normal': probabilities[0],
        'Urgent': probabilities[1],
        'Critical': probabilities[2]
    }

    # Confidence check
    max_confidence = max(probabilities)

    # Safety rule: if confidence < 70%, recommend one tier higher
    if max_confidence < 0.70:
        if predicted_service == 'Normal':
            safety_recommendation = 'Urgent'
        elif predicted_service == 'Urgent':
            safety_recommendation = 'Critical'
        else:
            safety_recommendation = 'Critical'
    else:
        safety_recommendation = predicted_service

    return {
        'predicted_service': predicted_service,
        'safety_recommendation': safety_recommendation,
        'confidence': max_confidence,
        'probabilities': prob_dict,
        'flag_for_review': max_confidence < 0.70
    }

# Example usage:
new_shipment = {
    'ZONE': 'Z2',
    'Gwgt': 150.5,
    'Cwgt': 180.2,
    'Pcs': 3,
    'DGR': 'Non HAZ',
    'Direction': 'Export',
    'Type': 'AIR',
    'iso_country_origin': 'ES',
    'iso_country_destination': 'FR',
    'day_of_week': 2,  # Wednesday
    'month': 6,        # June
    'hour': 14         # 2 PM
}

result = predict_optimal_service_level(new_shipment)
print(f"Recommended: {result['predicted_service']}")
print(f"Confidence: {result['confidence']:.2%}")
if result['flag_for_review']:
    print(f"⚠️ Low confidence - consider: {result['safety_recommendation']}")
```

---

## 7. Expected Outcomes

### 7.1 Model Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Weighted F1-Score | >0.75 | Balance across all classes |
| Recall for Critical | >0.80 | Catch shipments needing premium service |
| Overall Accuracy | >0.70 | Better than baseline (~58%) |
| Precision for Normal | >0.85 | Minimize false "safe to use cheap" predictions |

### 7.2 Business Impact

**Cost Reduction:**
- Reduce over-serviced orders from 42.5% → <20%
- Estimated annual savings: $X per year (depends on volume and pricing)

**Reliability Improvement:**
- Increase on-time delivery rate from current levels to >85%
- Reduce operational disruptions from late deliveries

**Decision Confidence:**
- Provide probability scores for all predictions
- Flag uncertain cases (confidence <70%) for manual review
- Enable data-driven booking decisions

---

## 8. Next Steps (Implementation Plan)

### Week 1: Feature Engineering & Model Development
1. ✅ Complete target label creation (`optimal_service_level`)
2. 🔄 Build preprocessing pipeline with ColumnTransformer
3. 🔄 Implement train-test split with stratification
4. 🔄 Train baseline model (DummyClassifier)

### Week 2: Model Training & Tuning
5. 🔄 Train Logistic Regression (baseline linear model)
6. 🔄 Train Random Forest with hyperparameter tuning
7. 🔄 Train Gradient Boosting / XGBoost
8. 🔄 Compare models using cross-validation

### Week 3: Model Evaluation & Analysis
9. ⏳ Evaluate best model on test set
10. ⏳ Generate confusion matrix and classification report
11. ⏳ Analyze feature importance
12. ⏳ Calculate business impact (cost savings, risk reduction)
13. ⏳ Identify error patterns (which shipments are mispredicted?)

### Week 4: Deployment & Documentation
14. ⏳ Build prediction function with confidence thresholds
15. ⏳ Create decision support dashboard (optional: Streamlit)
16. ⏳ Write final report with recommendations
17. ⏳ Prepare presentation for stakeholders

---

## 9. Technical Requirements

```python
# Core libraries
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0

# Optional advanced models
xgboost>=1.5.0
lightgbm>=3.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Notebook
jupyter>=1.0.0
ipykernel>=6.0.0

# Model interpretation (optional)
shap>=0.40.0  # For SHAP value analysis
```

**Computational Requirements:**
- Standard laptop sufficient (3,630 records)
- GridSearchCV with 5-fold CV may take 10-30 minutes
- XGBoost faster than sklearn's GradientBoosting

---

## 10. Repository Structure

```
├── ML.ipynb                      # Main analysis notebook
│   ├── Data Loading & Cleaning
│   ├── Target Engineering
│   ├── Exploratory Data Analysis
│   ├── Feature Engineering
│   ├── Model Training & Evaluation
│   ├── Model Comparison
│   └── Business Recommendations
│
├── Orders.csv                    # Shipment transaction data (3,695 records)
├── LeadtimeService.csv           # SLA targets by zone+service (18 records)
├── airports.csv                  # Airport-country mapping (84,490 records)
└── README.md                     # This file
```

---

## 11. Key Takeaways

**Problem:** 42.5% of shipments over-serviced, low on-time rates across all service levels

**Solution:** ML classification model to predict minimum sufficient service level

**Approach:** Multi-class classification with zone-specific SLA targets

**Expected Impact:**
- 50% reduction in over-servicing (42.5% → <20%)
- >85% on-time delivery rate with predicted service levels
- Significant cost savings while maintaining reliability

---

**Academic Project:** Master's in Data Analytics
**Focus:** Machine Learning for Business Optimization
**Objective:** Develop data-driven service level selection system
