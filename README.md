# AI for Business — Churn, Energy Forecasting & Customer Segmentation

> Three applied machine-learning case studies on real business datasets: predicting bank-customer churn with a neural network, forecasting building energy load with regression/boosting, and segmenting wholesale customers with unsupervised learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-9ACD32)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

---

## Overview

This repository collects three business-oriented ML projects. Each ships as an **exploratory notebook with committed outputs** (the source of the verified numbers below) plus an **"advanced" Python script** that re-implements the task with deeper modeling (deep nets, LightGBM, multi-algorithm clustering). The notebook results are reproducible from this repo; the advanced scripts are provided as enhanced, runnable implementations whose artifacts are not pre-committed.

| Project | Task | Datasets in repo |
|---------|------|------------------|
| Churn prediction | Binary classification | `Churn_Modelling.csv` (10,000 customers) |
| Energy load forecasting | Regression / time series | `energydata_complete.csv` |
| Customer segmentation | Clustering | `customers.csv` (wholesale, 440 customers) |

---

## 1. Customer Churn Prediction

**Goal:** predict whether a bank customer will exit, from 10 demographic/account features.

**Notebook (`ANN.ipynb`)** — a Keras Sequential network (two ReLU hidden layers, dropout, sigmoid output) trained on an 80/20 split with label + one-hot encoding and standardized features.

**Verified result (from committed notebook outputs):** on the 2,000-record held-out test set, the confusion matrix is

```
                 Predicted
                 Stay   Exit
   Actual Stay  [1560    35]
          Exit  [ 285   120]
```

i.e. **1,680 / 2,000 correct → 84.0% test accuracy**, with high precision on churners (120 / 155 ≈ 77.4%) but limited recall (120 / 405 ≈ 29.6%) — the expected difficulty on an imbalanced churn target.

**Advanced script (`advanced_churn_prediction.py`)** deepens this with a 4-layer network (128→64→32→16) using BatchNormalization, 30% dropout, Adam + `ReduceLROnPlateau`, `EarlyStopping`, **balanced class weights**, and AUC reporting — directly targeting the recall gap above.

## 2. Energy Load Forecasting

**Goal:** predict appliance energy consumption (Wh) in a low-energy building from temperature, humidity, and time features (10-minute intervals).

**Notebook (`energy_load_forecasting.ipynb`)** — EDA (consumption by hour/weekday/month, correlation heatmaps, log-transform of the target) followed by Linear Regression, SVR, and Random Forest, evaluated on a held-out test set.

**Verified results (from committed notebook outputs):**

| Model | Test R² | Mean abs. error (°/log-Wh) |
|-------|---------|----------------------------|
| Linear Regression | 27.85% | 0.3104 |
| SVR (RBF) | 27.50% | 0.3324 |
| **Random Forest** | **64.85%** | **0.2092** |
| Random Forest (+ engineered features) | **68.19%** | 0.1983 |

Random Forest is the clear winner; adding engineered features lifts test R² from ~0.65 to ~0.68. (5-fold CV likewise ranks RF highest: R² ≈ 0.55 ± 0.45.)

**Advanced script (`advanced_energy_forecasting.py`)** replaces RF with **LightGBM**, adds cyclical time encodings, lag features (1–24 steps), rolling statistics, time-series cross-validation, and SHAP-based feature importance.

## 3. Customer Segmentation

**Goal:** segment 440 wholesale customers by annual spend across six product categories (Fresh, Milk, Grocery, Frozen, Detergents/Paper, Delicatessen).

**Notebook (`customer_segments.ipynb`)** — log-transform + scaling, outlier analysis, PCA (the first 2 components explain **~70.7%** of variance — and ~93.1% on the cleaned data), then clustering scored by silhouette coefficient.

**Verified result (from committed notebook outputs):**

| # Clusters | Silhouette score |
|-----------|------------------|
| **2** | **0.412** |
| 3 | 0.374 |

Two clusters give the best separation, corresponding to the classic **retail vs. HoReCa (hotel/restaurant/café)** customer split.

**Advanced script (`advanced_customer_segmentation.py`)** compares **K-Means, DBSCAN, Gaussian Mixture Models, and hierarchical clustering**, with PCA/UMAP projection and silhouette / Davies-Bouldin / Calinski-Harabasz scoring.

## Tech Stack & Tools

| Area | Libraries |
|------|-----------|
| Data & numerics | **pandas**, **NumPy** |
| Visualization | **Matplotlib**, **Seaborn** |
| Classical ML | **scikit-learn** (LinearRegression, SVR, RandomForest, KMeans/DBSCAN/GMM, PCA, metrics) |
| Deep learning | **TensorFlow / Keras** |
| Gradient boosting | **LightGBM** |
| Optional | **SHAP** (interpretability), **UMAP** (dimensionality reduction), **keras-tuner** (HPO) |

## Project Structure

```
AI_For_Business/
├── ANN.ipynb                              # churn — Keras DNN (verified 84% test accuracy)
├── advanced_churn_prediction.py           # deeper net + class weights + AUC
├── energy_load_forecasting.ipynb          # energy — LR/SVR/RF (verified R² up to 0.68)
├── advanced_energy_forecasting.py         # LightGBM + lag/rolling/cyclical features
├── customer_segments.ipynb                # segmentation — PCA + silhouette (0.412 @ k=2)
├── advanced_customer_segmentation.py      # KMeans/DBSCAN/GMM/Hierarchical
├── Churn_Modelling.csv                    # 10,000 bank customers
├── energydata_complete.csv               # building energy time series
├── customers.csv                          # 440 wholesale customers
├── CCAI_Seasonal_Forecasting_with_Exercises.ipynb
├── Census income classification with scikit-learn.ipynb
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/ejazfahil/AI_For_Business.git
cd AI_For_Business
pip install -r requirements.txt

# reproduce the notebooks (outputs already committed)
jupyter notebook ANN.ipynb

# run the enhanced implementations
python advanced_churn_prediction.py
python advanced_energy_forecasting.py
python advanced_customer_segmentation.py
```

## Future Work

- Churn: SMOTE / threshold tuning to lift churner **recall**; deploy as a REST API.
- Energy: Temporal Fusion Transformer / N-BEATS and weather-forecast integration.
- Segmentation: RFM analysis and customer-lifetime-value modeling on top of the clusters.

## Conclusion

A practical portfolio of business ML problems — each with a reproducible, output-backed baseline notebook and a more advanced re-implementation — spanning supervised classification, regression/time-series forecasting, and unsupervised segmentation.

---

*Author: Fahil Ejaz · [@ejazfahil](https://github.com/ejazfahil)*
