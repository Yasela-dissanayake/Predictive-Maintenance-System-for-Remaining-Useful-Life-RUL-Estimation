import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

from preprocess import load_data, create_rul, get_sensor_columns, clip_rul
from features import add_rolling_features

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load dataset
df = load_data("/mnt/sda3/Other/Work/New research/predictive-maintenance/data/raw/train_FD001.txt")

# Create RUL Column
df = create_rul(df)
df = clip_rul(df)

# Select features
sensor_columns = get_sensor_columns(df)
df = add_rolling_features(df,sensor_columns)
df = df.dropna() 

# Split data
units = df['unit'].unique()

train_units = units[:80]
test_units = units[80:]

# create train/test df
train_df = df[df['unit'].isin(train_units)]
test_df = df[df['unit'].isin(test_units)]

feature_columns = [col for col in df.columns if 'sensor' in col]

# create features
X_train = train_df[feature_columns]
y_train = train_df['RUL']   

X_test = test_df[feature_columns]
y_test = test_df['RUL']

# adding scaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# model = RandomForestRegressor()
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=RANDOM_STATE
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

model = grid_search.best_estimator_

print("Best Prameters:")
print(grid_search.best_params_)

# model.fit(X_train_scaled, y_train)


joblib.dump(model, "/mnt/sda3/Other/Work/New research/predictive-maintenance/models/rf_model.pkl")
joblib.dump(scaler, "/mnt/sda3/Other/Work/New research/predictive-maintenance/models/scaler.pkl")
joblib.dump(feature_columns, "/mnt/sda3/Other/Work/New research/predictive-maintenance/models/features.pkl")

print("Training complete.")
print("Model saved.")