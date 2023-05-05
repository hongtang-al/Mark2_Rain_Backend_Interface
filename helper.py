import pandas as pd
import numpy as np
import audio_math 
import boto3
import io
# from tqdm.notebook import tqdm
from tqdm import tqdm
tqdm.pandas()


def df_to_s3(df, key, bucket, verbose=True, format="csv"):
    if format == "csv":
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
    elif format == "parquet":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
    else:
        raise Exception(f"format '{format}' not recognized")
    # write stream to S3
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    if verbose:
        print(f"Uploaded file to s3://{bucket}/{key}")


def df_from_s3(key, bucket, format="csv", **kwargs):
    """read csv from S3 as pandas df
    Arguments:
        key - key of file on S3
        bucket - bucket of file on S3
        **kwargs - additional keyword arguments to pass pd.read_ methods
    Returns:
        df - pandas df
    """
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]
    if format == "csv":
        csv_string = body.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_string), **kwargs)
    elif format == "parquet":
        bytes_obj = body.read()
        df = pd.read_parquet(io.BytesIO(bytes_obj), **kwargs)
    else:
        raise Exception(f"format '{format}' not recognized")

    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
    if 'ref_time' in df.columns:
        df['ref_time'] = pd.to_datetime(df['ref_time'], errors='coerce', utc=True)
    return df


def list_keys(prefix, bucket):
    """return list of files with specific S3 prefix"""
    bucket_resource = boto3.resource("s3").Bucket(bucket)
    key_list = [file.key for file in bucket_resource.objects.filter(Prefix=prefix)]
    return key_list

def get_roc(df, col_of_interest, periods_back=4, verbose=True, device_present=True):

    def calc_slope(y):
        x = pd.Series(range(len(y)))
        # handle NAS
        idx = np.isfinite(y.to_numpy())
        slope = np.polyfit(x[idx], y[idx], 1)[0]
        return slope
    min_periods = 2
    if device_present:
        if df["device"].isna().all():
            df[f"{col_of_interest}_ROC"] = np.nan
            return df
        new_dfs = []
        for device, device_df in (
            tqdm(df.groupby("device")) if verbose else df.groupby("device")
        ):
            device_df.sort_values("time", inplace=True)
            device_df[f"{col_of_interest}_ROC"] = (
                device_df[col_of_interest]
                .rolling(periods_back, min_periods=min_periods)
                .apply(calc_slope)
            )
            new_dfs.append(device_df)

        if len(new_dfs) > 0:
            return pd.concat(new_dfs)
        else:
            df[f"{col_of_interest}_ROC"] = np.nan
            return df
    else:
        df.sort_values("time", inplace=True)
        df[f"{col_of_interest}_ROC"] = (
            df[col_of_interest]
            .rolling(periods_back, min_periods=min_periods)
            .apply(calc_slope)
        )
        return df

RAIN_ENERGY_THRESHOLD = 0.6
RAIN_LOG_FACTOR = 0.6


def binning_func(energy, threshold=RAIN_ENERGY_THRESHOLD):
    return np.floor(np.log(1 + ((energy - threshold) * RAIN_LOG_FACTOR)) / np.log(1.13))


def reverse_binning_func(drop_bin, threshold=RAIN_ENERGY_THRESHOLD):
    return (((np.e ** (drop_bin * np.log(1.13))) - 1) / RAIN_LOG_FACTOR) + threshold


dsd_weights = {f"dsd{i}": reverse_binning_func(i) for i in range(32)}

def aggregate_pft_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    The 30 pft bins for each minutely disdrometer record represent a 60 second time series with
    2 second increments. This function summarizes that time series for each minutely
    timestamp before the minutely time series data are aggregated

    Args:
        df: data frame containing pft columns (fft0-fft29)
    """
    pft_cols = ["pft{}".format(i) for i in range(30)]
    new_df = df.copy()
    new_df["pft_max"] = new_df[pft_cols].max(axis=1)
    new_df["pft_mean"] = new_df[pft_cols].mean(axis=1)
    new_df["pft_min"] = new_df[pft_cols].min(axis=1)
    new_df["pft_std"] = new_df[pft_cols].std(axis=1)
    new_df["pft_count"] = new_df[pft_cols][new_df[pft_cols] > 0].count(axis=1)
    new_df.drop(columns=pft_cols, inplace=True)
    return new_df


def add_weighted_dsd_data(df, weights=dsd_weights.values(), add_to_df=True):
    dsd_columns = [f"dsd{i}" for i in range(32)]
    dsd_data = df[dsd_columns]
    weighted_dsd_data = (dsd_data * weights).add_suffix("_weighted")
    if add_to_df:
        return pd.concat([df, weighted_dsd_data], axis=1)
    else:
        return weighted_dsd_data
    
def standardize_rh_units(rh_data: pd.Series, rh_unit) -> pd.Series:
    """
    Perform sanity check on RH units, and return standardized units (percent)
    :param rh_data: pandas series of relative humidity data
    :param rh_unit: expecterh unit. 'percent' for rh from 0-100, 'unitless' for rh from 0-1
    :return: rh_data with standardized units
    """
    if rh_unit == "unitless":
        max_expected_rh_value = 1
    elif rh_unit == "percent":
        max_expected_rh_value = 100
    else:
        raise Exception(
            f"rh_unit must be 'unitless' or 'percent', did not recognize '{rh_unit}'"
        )

    max_rh = rh_data.max()
    if max_expected_rh_value == 1 and max_rh > max_expected_rh_value:
        raise Exception(
            f"Expected max RH value of {max_expected_rh_value}, but found value of {max_rh}"
        )
    if max_expected_rh_value == 100 and max_rh <= 1:
        raise Exception(
            f"Expected max RH value of {max_rh}, but found max value of {max_rh}"
        )

    if max_expected_rh_value == 1:
        return rh_data * 100
    if max_expected_rh_value == 100:
        return rh_data

def add_all_engineered_features(
    prepared_df,
    add_weighted_dsd_sum=True,
    add_slp=False,
    add_dew_point_features=True,
    add_rh_plausibility_curve=False,
    add_precipitable_water=False,
    add_rh_deltas=True,
    add_fft_stats=True,
    add_fft_integrals=True,
    add_spectral_sensitivity_ratio=True,
    add_dsd_stats=True,
    add_dsd_peaks=True,
    add_fft_peaks=True,
    add_roc_features=True,
    verbose=True,
    device_present=True,
):
    dsd_cols = [f"dsd{i}" for i in range(32)]
    fft_cols = [f"fft{i}" for i in range(38)]
    fft_low_end_cols = [f"fft{i}" for i in range(19)]
    fft_high_end_cols = [f"fft{i}" for i in range(19, 38)]

    # handle object type columns (all nans)
    for col in dsd_cols + fft_cols:
        prepared_df[col] = prepared_df[col].astype(float)

    # standardize RH units
    print("Standardizing RH Units")
    rh_cols_with_units = {"rh": "unitless", "rh_u21": "percent", "spec_rh_1": "percent"}
    for col, unit in rh_cols_with_units.items():
        if col in prepared_df.columns:
            prepared_df[col] = standardize_rh_units(prepared_df[col], rh_unit=unit)

    # add weighted dsd sums
    if add_weighted_dsd_sum:
        if verbose:
            print("Calculating weighted DSD sums")
        prepared_df = add_weighted_dsd_data(prepared_df)
        weighted_dsd_cols = [f"dsd{i}_weighted" for i in range(32)]

        prepared_df["weighted_dsd_sum"] = prepared_df[weighted_dsd_cols].sum(axis=1)
    if add_dew_point_features:
        if verbose:
            print("Adding dewpoint temperature features")
        # add dew point temperature relations
        prepared_df["dew_temp_difference"] = prepared_df["tair"] - prepared_df["tdew"]
        prepared_df["dew_temp_ratio"] = prepared_df["tair"] / prepared_df["tdew"]
    if add_rh_plausibility_curve:
        if verbose:
            print("Calculating RH/Rainfall Plausibility Curve")
        prepared_df["rh_plausibility_curve"] = prepared_df["rh"].apply(
            lambda x: reverse_plausible_rainfall_equation(x, rh_unit="percent")
        )
    if add_precipitable_water:
        if verbose:
            print("Estimating precipitable water content")
        # add precipitable water estimate
        prepared_df["precipitable_water"] = gueymard94_pw(
            prepared_df["tair"], prepared_df["rh"]
        )
    if add_rh_deltas:
        if verbose:
            print("Calculating internal RH Deltas")
        # add RH deltas
        if "rh_u21" in prepared_df.columns and "spec_rh_1" in prepared_df.columns:
            raise Exception(
                "Found BOTH 'rh_u21' (Mark2) and 'spec_rh_1' (Mark3) in the data. Expecting only one")
        if "rh_u21" in prepared_df.columns:
            internal_rh_col = "rh_u21"
        elif "spec_rh_1" in prepared_df.columns:
            internal_rh_col = "spec_rh_1"
        else:
            raise Exception("Could not find internal RH col. Expecting 'rh_u21' for Mark2 or 'spec_rh_1' for Mark3")
        prepared_df["internal_rh_delta"] = prepared_df["rh"] - prepared_df[internal_rh_col]
        prepared_df["internal_rh_ratio"] = prepared_df["rh"] / prepared_df[internal_rh_col]
    if add_fft_stats:
        if verbose:
            print("Generating FFT stats")
        # prepare features from old model
        prepared_df["fft_skew"] = audio_math.get_skew(prepared_df, fft_cols)
        prepared_df["fft_kurtosis"] = audio_math.get_kurtosis(prepared_df, fft_cols)
        prepared_df["fft_max"] = audio_math.get_max(prepared_df, fft_cols)
    if add_fft_integrals:
        if verbose:
            print("Generating FFT integrals")
        prepared_df["fft_integral_low"] = audio_math.get_integral(
            prepared_df, fft_low_end_cols
        )
        prepared_df["fft_integral_high"] = audio_math.get_integral(
            prepared_df, fft_high_end_cols
        )
    if add_spectral_sensitivity_ratio:
        if verbose:
            print("Generating spectral sensitivity ratios")
        prepared_df[
            "spectral_sensitivity_ratio_4to7"
        ] = audio_math.calculate_spectral_selectivity_ratio(prepared_df, method="4to7")
        prepared_df[
            "spectral_sensitivity_ratio_integral_range"
        ] = audio_math.calculate_spectral_selectivity_ratio(
            prepared_df, method="integral_ratio"
        )
    if add_dsd_stats:
        prepared_df["dsd_skew"] = audio_math.get_skew(prepared_df, dsd_cols)
        prepared_df["dsd_kurtosis"] = audio_math.get_kurtosis(prepared_df, dsd_cols)
        prepared_df["dsd_max"] = audio_math.get_max(prepared_df, dsd_cols)

    # Add some of my own features in the spirit of old ones (characterizing distributions)
    if add_dsd_peaks:
        if verbose:
            print("Calculating DSD Peaks")

            prepared_df["dsd_peak_count"] = prepared_df[dsd_cols].progress_apply(
                audio_math.get_peak_count, axis=1
            )
        else:
            prepared_df["dsd_peak_count"] = prepared_df[dsd_cols].apply(
                audio_math.get_peak_count, axis=1
            )
    if add_fft_peaks:
        if verbose:
            print("Calculating FFT Peaks")
            prepared_df["fft_peak_count"] = prepared_df[fft_cols].progress_apply(
                audio_math.get_peak_count, axis=1
            )
        else:
            prepared_df["fft_peak_count"] = prepared_df[fft_cols].apply(
                audio_math.get_peak_count, axis=1
            )

    if add_roc_features:
        if verbose:
            print("Adding ROC Features")
        if add_slp:
            prepared_df = get_roc(
                prepared_df,
                "slp",
                periods_back=4,
                verbose=verbose,
                device_present=device_present,
            )
        prepared_df = get_roc(
            prepared_df,
            "swdw",
            periods_back=4,
            verbose=verbose,
            device_present=device_present,
        )
        if add_rh_deltas:
            prepared_df = get_roc(
                prepared_df,
                "internal_rh_ratio",
                periods_back=4,
                verbose=verbose,
                device_present=device_present,
            )
        prepared_df = get_roc(
            prepared_df,
            "rh",
            periods_back=4,
            verbose=verbose,
            device_present=device_present,
        )
        # prepared_df = get_roc(prepared_df, "p", periods_back=4, verbose=verbose)
    if verbose:
        print("Done!")
    return prepared_df


