import pandas as pd
import numpy as np

def load_and_preprocess(filepath, species="Adelie Penguin", colony="King George Island"):
    """
    Load the penguin data, filtering for the given species and colony.
    Adds date features and sorts by date.
    """
    df = pd.read_csv(filepath)
    df = df[(df['common_name'] == species) & (df['colony_name'] == colony)].copy()
    df['date_gmt'] = pd.to_datetime(df['date_gmt'])
    df = df.sort_values(by='date_gmt')
    
    # Create additional date features.
    df['day_of_year'] = df['date_gmt'].dt.dayofyear
    df['month'] = df['date_gmt'].dt.month
    df['date'] = df['date_gmt'].dt.date
    return df

def aggregate_daily(df):
    """
    Aggregate the data by day and compute daily averages.
    Also adds the count of records per day.
    """
    daily_df = df.groupby('date').agg({
        'km_to_colony_mean': 'mean',
        't2m': 'mean',
        'siconc': 'mean',
        'sst': 'mean',
        'tp': 'mean',
        'sd': 'mean', 
        'rsn': 'mean', 
        'avg_smr': 'mean',
        'month': 'first',
        'day_of_year': 'first'
    }).reset_index()
    daily_df['n_records'] = df.groupby('date').size().values
    return daily_df

def filter_daily(daily_df, min_records=1):
    """
    Filter out days with fewer than min_records.
    """
    return daily_df[daily_df['n_records'] >= min_records]

def add_lag_features(daily_df, features, lag=1):
    """
    Add one-day lagged versions of the specified features.
    """
    for feature in features:
        daily_df[f'{feature}_lag{lag}'] = daily_df[feature].shift(lag)
    return daily_df

def drop_missing_rows(daily_df):
    """
    Drop rows with any missing values.
    """
    return daily_df.dropna()

def add_cyclical_features(daily_df):
    """
    Add cyclical (sine/cosine) transformations for day_of_year and month.
    """
    daily_df['day_of_year_sin'] = np.sin(2 * np.pi * daily_df['day_of_year'] / 365.0)
    daily_df['day_of_year_cos'] = np.cos(2 * np.pi * daily_df['day_of_year'] / 365.0)
    daily_df['month_sin'] = np.sin(2 * np.pi * daily_df['month'] / 12.0)
    daily_df['month_cos'] = np.cos(2 * np.pi * daily_df['month'] / 12.0)
    return daily_df

def smooth_features(daily_df, vars_list, window=7):
    """
    Apply a rolling mean with the given window to each variable in vars_list.
    Stores the result as {var}_smoothed.
    """
    for var in vars_list:
        daily_df[f'{var}_smoothed'] = daily_df[var].rolling(window=window, center=True).mean()
    return daily_df

def add_shifted_features(daily_df, vars_list, suffix="_smoothed_shift1", shift=1):
    """
    Create one-day lagged (shifted) versions of the smoothed variables.
    """
    for var in vars_list:
        daily_df[f'{var}_smoothed{suffix}'] = daily_df[f'{var}_smoothed'].shift(shift)
    return daily_df