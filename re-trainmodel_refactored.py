import pandas as pd
import numpy as np
import os
import boto3
import joblib
import matplotlib.pyplot as plt
from helper import add_all_engineered_features, aggregate_pft_data, df_from_s3

# Read file merged from backend chunk files
bucket = 'arable-adse-dev'
key = 'Carbon Project/Stress Index/UCD_Almond/C007978_merged_data.csv'
df_merged = df_from_s3(key, bucket, format="csv")

# Normalize the RH file
RHCUTOFF = 2
if df_merged['rh'].max() > RHCUTOFF:
    df_merged['rh'] = df_merged['rh'] / 100

# Feature engineering part 1
res = add_all_engineered_features(df_merged)

# Feature engineering part 2 for pft_count, pft_mean
res1 = aggregate_pft_data(df_merged)

# Define feature names for model input
input_cols = [
    'weighted_dsd_sum', 'dsd_max', 'dsd_peak_count', 'dsd_skew',
    'dsd_kurtosis', 'fft_integral_low', 'fft_integral_high',
    'fft_peak_count', 'spectral_sensitivity_ratio_4to7',
    'rh_ROC', 'internal_rh_ratio_ROC', 'internal_rh_ratio',
    'dew_temp_difference', 'swdw', 'device', 'time'
]
pft_cols = ['pft_count', 'pft_mean']

# Concatenate dataframes
output = pd.concat([res[input_cols], res1[pft_cols]], axis=1)

# Select relevant model features
model_features = [x for x in input_cols if x not in ['device', 'time']] + ['pft_count', 'pft_mean']

# Drop rows with missing data
test_df = output[model_features].dropna()

# Load model
GBR = joblib.load('GBR1p2p2.joblib')

# Define what-if scenarios
test_df['dsd_peak_count'] = output['dsd_peak_count']

# Make predictions
y_pred = GBR.predict(test_df[model_features])

# Create a new dataframe with predictions
new_df = pd.concat([test_df.reset_index(), pd.DataFrame(y_pred, columns=['precip'])], axis=1)
new_df['device'] = output['device']
new_df['time'] = output['time']

# Plot time series
new_df.plot(x='time', y=['precip'], figsize=(20, 4))
plt.show()
new_df.plot(x='time', y=['weighted_dsd_sum'], figsize=(20, 4))
plt.show()

# Load training data and plot feature importances
key = 'Carbon Project/Stress Index/UCD_Almond/x_train.csv'
x_train = df_from_s3(key, bucket, format="csv")

importances = GBR.feature_importances_
feature_names = x_train[input_cols].columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

print('Finish')