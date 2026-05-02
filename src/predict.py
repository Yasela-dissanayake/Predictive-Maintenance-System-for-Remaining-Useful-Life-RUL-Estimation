import joblib
import pandas as pd

model = joblib.load("/mnt/sda3/Other/Work/New research/predictive-maintenance/models/rf_model.pkl")
scaler = joblib.load("/mnt/sda3/Other/Work/New research/predictive-maintenance/models/scaler.pkl")
feature_columns = joblib.load("/mnt/sda3/Other/Work/New research/predictive-maintenance/models/features.pkl")

input_data ={}

for col in feature_columns:
    input_data[col] = [0]

input_data['sensor_4_ma5'] = [500]
input_data['sensor_11_ma5'] = [600]
input_data['sensor_9_ma5'] = [700]

input_df = pd.DataFrame(input_data)

input_df = input_df[feature_columns]

input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)

print("Predicted RUL:", prediction[0])