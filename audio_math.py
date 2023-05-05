import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks


def get_skew(df, cols):
    """
    computes the skew
    df: the entire df
    cols: the names of either the dsd, pft, or fft columns
    returns: the skew of the curve created by the above columns for each row
    """
    return skew(df[cols], axis=1)


def get_kurtosis(df, cols):
    """
    computes the kurtosis (when all values are the same, it returns -3, which for now we switch to 0)
    df: the entire df
    cols: the names of either the dsd, pft, or fft columns
    returns: the kurtosis of the curve created by the above columns for each row
    """
    return kurtosis(df[cols], axis=1)


def get_max(df, cols):
    """
    grabs the maximum value
    df: the entire df
    cols: the names of either the dsd, pft, or fft columns
    returns: the maximum of the above columns for each row
    """
    return np.max(df[cols], axis=1)


def get_mean(df, cols):
    """
    grabs the mean value
    df: the entire df
    cols: the names of either the dsd, pft, or fft columns
    returns: the maximum of the above columns for each row
    """
    return np.mean(df[cols], axis=1)


def get_min(df, cols):
    """
    grabs the min value
    df: the entire df
    cols: the names of either the dsd, pft, or fft columns
    returns: the maximum of the above columns for each row
    """
    return np.min(df[cols], axis=1)


def get_std(df, cols):
    """
    grabs the standard deviation of values
    df: the entire df
    cols: the names of either the dsd, pft, or fft columns
    returns: the maximum of the above columns for each row
    """
    return np.std(df[cols], axis=1)


def get_integral(df, cols):
    """
    uses trapezoid method to estimate the integral
    df: the entire df
    cols: the names of the columns that will be used to create the curve
    returns: the area under the curve of the above columns for each row
    """
    hertz_range_of_data = 400
    bins_in_range = 19
    hertz_per_bin = hertz_range_of_data / bins_in_range
    return np.trapz(df[cols], dx=hertz_per_bin, axis=1)


def calculate_spectral_selectivity_ratio(df, method="4to7"):
    """
    computes spectral_selectivity_ratio
    df: the entire df
    cols: the names of either the dsd, pft, or fft columns
    returns: the skew of the curve created by the above columns for each row
    """

    clean_df = df.copy()
    # weird case where there are 570k + records with -1.72 as value in fft37 bin. wtf
    # replacing with 0
    clean_df.loc[clean_df["fft37"] < 0, "fft37"] = 0

    all_lower_frequencies = [f"fft{i}" for i in range(19)]
    all_upper_frequencies = [f"fft{i}" for i in range(19, 38)]

    if method == "integral_ratio":
        rain_response_frequencies = [
            "fft5",
            "fft6",
            "fft7",
        ]  # corresponds to 426Hz-468Hz

        upper_integral = get_integral(clean_df, all_upper_frequencies)
        lower_integral = get_integral(clean_df, all_lower_frequencies)
        rain_response_region_integral = get_integral(
            clean_df, rain_response_frequencies
        )
        lower_non_rain_response_integral = (
                lower_integral - rain_response_region_integral
        )

        spectral_selectivity_ratio = rain_response_region_integral / (
                upper_integral + lower_non_rain_response_integral
        )

        # Don't want infinite values here, so replacing with arbitrarily high value
        # max value in real world training data was 106k
        spectral_selectivity_ratio[spectral_selectivity_ratio == np.inf] = 1000000

        # Dont want nans when numerator and denomenator are both 0
        spectral_selectivity_ratio[rain_response_region_integral == 0] = 0

    elif method == "4to7":
        spectral_selectivity_ratio = clean_df["fft4"] / clean_df["fft7"]

        # Don't want very high values values here as numerator approaches 0,
        # so set reasonable max around 1.5x max values seen for denomenator
        spectral_selectivity_ratio[spectral_selectivity_ratio > 10000] = 10000
        # Dont want nans when numerator and denomenator are both 0
        spectral_selectivity_ratio[clean_df["fft4"] == 0] = 0
    else:
        raise Exception(f"unfamiliar method: {method}")

    return spectral_selectivity_ratio


def get_peak_count(array):
    # add buffer to the edges so peaks can be found at first or last element
    array = np.concatenate(([min(array)], array, [min(array)]))
    peaks = find_peaks(array)
    number_of_peaks = len(peaks[0])
    return number_of_peaks
