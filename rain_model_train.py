import pandas as pd
import numpy as np
import os
import boto3
import joblib
from helper import *
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

def main():
    # read file mergede from backend chunk files
    bucket='arable-adse-dev'
    key='Carbon Project/Stress Index/UCD_Almond/merged_rain_check2.csv'
    df_merged=df_from_s3(key, bucket, format="csv")

    RHCUTOFF=2
    # normalize the rh file
    if df_merged['rh'].max() >RHCUTOFF:
        df_merged['rh']=df_merged['rh']/100

    #feature engineering part 1
    res=add_all_engineered_features(df_merged)
    #feature engineering part 2 for pft_count, pft_mean
    res1=aggregate_pft_data(df_merged)

    #define feature names for model input
    input_cols=['dsd_kurtosis', 'fft_integral_low', 'fft_integral_high',
       'fft_peak_count',
       'spectral_sensitivity_ratio_4to7', 'rh_ROC', 'internal_rh_ratio_ROC',
       'internal_rh_ratio', 'dew_temp_difference', 'swdw','device', 'time']#
    pft_cols=['pft_count', 'pft_mean']
    output=pd.concat([res[input_cols], res1[pft_cols]], axis=1)

    # prep re-train model using jacob's training data
    x_train=df_from_s3('Carbon Project/Stress Index/UCD_Almond/x_train.csv', bucket, format="csv")
    y_train=df_from_s3('Carbon Project/Stress Index/UCD_Almond/y_train.csv', bucket, format="csv")

    model_features=[x for x in input_cols if x not in ['device', 'time']]
    test_df=output[model_features]
    test_df=test_df.dropna()

    # retrain decision tree model
    model_params={
              'alpha': 0.9, 
              'criterion': 'friedman_mse', 
              'init': None, 
              'learning_rate': 0.1, 
              'loss': 'absolute_error', 'max_depth': 7, 
              'max_features': None, 'max_leaf_nodes': None, 
              'min_impurity_decrease': 5.37e-05, 
              'min_samples_leaf': 25, 'min_samples_split': 6, 'min_weight_fraction_leaf': 0.0, 
              'n_estimators': 350, 'n_iter_no_change': None, 
              'random_state': 42, 'subsample': 0.78, 'tol': 0.0001, 
              'validation_fraction': 0.1, 'verbose': 1, 'warm_start': False
              }
    GBR = GradientBoostingRegressor(**model_params)
    GBR.fit(x_train[model_features], y_train)

    # Save the trained model to a file
    joblib.dump(GBR, 'GBR1p2p2.joblib')

    y_pred = GBR.predict(test_df)


if __name__ == '__main__':
    main()
