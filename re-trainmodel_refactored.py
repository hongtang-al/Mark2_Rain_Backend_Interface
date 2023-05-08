import pandas as pd
import joblib
from helper import add_all_engineered_features, aggregate_pft_data, df_from_s3


def make_predictions(model_file, input_file, output_file):
    # read file merged from backend chunk files
    bucket = 'arable-adse-dev'
    key = 'Carbon Project/Stress Index/UCD_Almond/C007978_merged_data.csv'
    df_merged = df_from_s3(key, bucket, format="csv")

    RHCUTOFF = 2
    # normalize the rh file
    if df_merged['rh'].max() > RHCUTOFF:
        df_merged['rh'] = df_merged['rh'] / 100

    # feature engineering part 1
    res = add_all_engineered_features(df_merged)
    # feature engineering part 2 for pft_count, pft_mean
    res1 = aggregate_pft_data(df_merged)

    # define feature names for model input
    input_cols = [
        'weighted_dsd_sum', 'dsd_max', 'dsd_peak_count', 'dsd_skew',
        'dsd_kurtosis', 'fft_integral_low', 'fft_integral_high',
        'fft_peak_count',
        'spectral_sensitivity_ratio_4to7', 'rh_ROC', 'internal_rh_ratio_ROC',
        'internal_rh_ratio', 'dew_temp_difference', 'swdw', 'device', 'time'
    ]
    pft_cols = ['pft_count', 'pft_mean']

    # combine engineered features and pft data
    res = pd.merge(res[input_cols], res1[pft_cols], left_index=True, right_index=True)

    # define features for the model
    model_features = [x for x in input_cols if x not in ['device', 'time']] + ['pft_count', 'pft_mean']

    # create the input data frame
    df = res[model_features].dropna()

    # load the saved model
    GBR = joblib.load(model_file)

    # make predictions
    y_pred = GBR.predict(df[model_features])

    # combine predictions with input data
    df['device'] = res['device']
    df['time'] = res['time']
    new_df = pd.concat([df.reset_index(), pd.DataFrame(y_pred, columns=['precip'])], axis=1)
    # # prep re-train model using jacob's training data
    key='Carbon Project/Stress Index/UCD_Almond/x_train.csv'
    x_train=df_from_s3(key, bucket, format="csv")

    # plot time series for review
    new_df.plot(x='time', y=['precip'], figsize=(20, 4))
    new_df.plot(x='time', y=['weighted_dsd_sum'], figsize=(20, 4))

    # save the output to a file
    new_df.to_csv(output_file, index=False)

    # If you want to get the feature importances from the saved model
    importances = GBR.feature_importances_
    feature_names = df[input_cols].columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # plot feature importances sorted by importance
if main == '__main__':
    make_predictions(GBR1p2p2.joblib, input_file, output_file)