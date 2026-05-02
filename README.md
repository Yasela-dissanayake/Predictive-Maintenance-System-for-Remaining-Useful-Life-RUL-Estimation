# Predictive Maintenance System for Failure Detection & RUL Estimation

A machine learning-based predictive maintenance system designed to estimate equipment degradation and Remaining Useful Life (RUL) using multivariate time-series sensor data.

This project uses the NASA CMAPSS turbofan engine degradation dataset and implements a complete end-to-end ML pipeline including preprocessing, feature engineering, model training, evaluation, and prediction.

---

# Project Overview

Predictive maintenance aims to detect degradation patterns before equipment failure occurs.

This system analyzes historical sensor measurements from aircraft engines and predicts:

- Remaining Useful Life (RUL)
- Equipment degradation trends
- Failure progression patterns

The project focuses on building a realistic and modular ML workflow for industrial predictive maintenance applications.

---

# Features

- NASA CMAPSS dataset integration
- Remaining Useful Life (RUL) estimation
- Time-series feature engineering
- Rolling mean degradation features
- Feature scaling pipeline
- Engine-based train/test splitting
- XGBoost regression model
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Actual vs Predicted visualization
- Saved model + scaler pipeline
- End-to-end prediction workflow

---

# Project Structure

```text
predictive-maintenance/
│
├── app/
│   └── api.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
└── README.md

---

# Dataset

Dataset used:

* NASA CMAPSS Turbofan Engine Degradation Dataset

Dataset characteristics:

* Multivariate time-series sensor data
* Multiple engines operating until failure
* Operational settings + sensor measurements
* Realistic degradation progression simulation

Dataset source:

[https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

---

# Machine Learning Pipeline

## 1. Data Preprocessing

* Dataset loading
* Missing column removal
* Column naming
* RUL generation
* RUL clipping

---

## 2. Feature Engineering

The project includes rolling mean features to capture degradation trends over time.

Example:

```python
sensor_4_ma5
```

represents the moving average of sensor 4 over the previous 5 cycles.

---

## 3. Train/Test Strategy

Instead of random row splitting, the project uses:

* Engine-based splitting

This prevents data leakage and creates a more realistic predictive maintenance evaluation setup.

---

## 4. Feature Scaling

StandardScaler is used to normalize sensor values before training.

---

## 5. Models

Implemented models:

* RandomForestRegressor
* XGBoost Regressor

---

## 6. Hyperparameter Tuning

GridSearchCV is used for tuning:

* learning_rate
* max_depth
* n_estimators

---

# Evaluation Metrics

The model is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Example results:

```text
MAE: ~23
RMSE: ~30
R² Score: ~0.45
```

---

# Visualization

The project includes:

* Feature importance analysis
* Actual vs Predicted RUL plots

These help analyze degradation behavior and model performance.

---

# Example Workflow

```text
Raw Sensor Data
        ↓
Preprocessing
        ↓
RUL Generation
        ↓
Feature Engineering
        ↓
Scaling
        ↓
XGBoost Training
        ↓
Evaluation
        ↓
Prediction
```

---

# Installation

## Clone Repository

```bash
git clone <your-repo-url>
cd predictive-maintenance
```

---

## Create Virtual Environment

### Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Train Model

```bash
python src/train.py
```

---

## Evaluate Model

```bash
python src/evaluate.py
```

---

## Run Prediction

```bash
python src/predict.py
```

---

# Future Improvements

* LSTM / GRU based sequence models
* Real-time sensor streaming
* REST API deployment
* Interactive dashboard
* Docker deployment
* Advanced anomaly detection

---

# Key Learning Outcomes

* Time-series machine learning
* Predictive maintenance workflows
* Feature engineering for degradation modeling
* ML pipeline architecture
* Model evaluation and explainability
* End-to-end inference systems

---

# Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Joblib

---

# Status

Current Status:

* Core ML pipeline completed
* Model training and evaluation operational
* Prediction pipeline functional
* Additional deployment improvements planned

```
```
