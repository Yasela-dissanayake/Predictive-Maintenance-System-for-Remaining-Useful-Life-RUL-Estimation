import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import load_data, create_rul, get_sensor_columns, clip_rul
from features import add_rolling_features

# Load dataset
df = load_data("/mnt/sda3/Other/Work/New research/predictive-maintenance/data/raw/train_FD001.txt")

# Create RUL Column
df = create_rul(df)
df = clip_rul(df)

# Select features
sensor_columns = get_sensor_columns(df)
df = add_rolling_features(df,sensor_columns)
df = df.dropna() 


X = df[sensor_columns]

y = df['RUL']

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

model = joblib.load("/mnt/sda3/Other/Work/New research/predictive-maintenance/models/rf_model.pkl")
scaler = joblib.load("/mnt/sda3/Other/Work/New research/predictive-maintenance/models/scaler.pkl")

X_test_scaled = scaler.transform(X_test)

# Make predictons
predictions = model.predict(X_test_scaled)

# Calc metrics
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

# Results
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.figure(figsize=(8,8))

plt.scatter(y_test, predictions, alpha=0.5)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")

plt.title("Actual vs Predicted RUL")

min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())

plt.plot([min_val, max_val], [min_val, max_val])

plt.tight_layout()
plt.show()

#########################

importance = model.feature_importances_

# create feature importance data
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print(feature_importance.head(10))

# plot importanec

plt.figure(figsize=(10, 6))

plt.bar(feature_importance['Feature'][:10], feature_importance['Importance'][:10])

plt.xticks(rotation=45)

plt.xlabel('Sensor Features')
plt.ylabel('Importance')

plt.title('Top 10 Feature Importances')

plt.tight_layout()

plt.show()


