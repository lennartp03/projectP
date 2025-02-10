import zipfile
import os
import pandas as pd
import numpy as np
import pathlib
import xarray as xr
import netCDF4 as nc
import dask
import json
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import zscore
from statsmodels.tsa.seasonal import STL


def monthly_data_prep(folder_path, extension):
    # target_files = [f for f in files if f.endswith(extension)]
    target_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)]

    flattened_frames = []

    for file in target_files: 
        ds = xr.open_dataset(file, engine='netcdf4')
        df = ds.to_dataframe()
        flattened = df.groupby(['valid_time', 'latitude', 'longitude']).sum().reset_index()
        flattened['valid_time'] = flattened['valid_time'].dt.date
        flattened = flattened.drop(columns=['number'])
        flattened_frames.append(flattened)
        
    flattened_0, flattened_1 = flattened_frames 

    if flattened_0["valid_time"].equals(flattened_1["valid_time"]):
        combined = flattened_0.merge(flattened_1, left_index=True, right_index=True)
        combined = combined.drop(columns=['valid_time_y', 'latitude_y', 'longitude_y'])
        combined = combined.rename(columns={'valid_time_x': 'valid_time', 'latitude_x': 'latitude', 'longitude_x': 'longitude'})
        return combined
    else:
        print('Dataframes do not have the same valid time.')
        return None
    
def aggregate_data(combined):
    grouped = combined.groupby('valid_time').agg('mean').reset_index()
    grouped = grouped[(grouped['valid_time'] >= pd.Timestamp('1996-01-01')) & (grouped['valid_time'] <= pd.Timestamp('2017-12-31'))].reset_index(drop=True)
    grouped['valid_time'] = pd.to_datetime(grouped['valid_time'])
    grouped['year'] = grouped['valid_time'].dt.year
    grouped['month'] = grouped['valid_time'].dt.month
    return grouped


def calculate_anomalies(df, reference_period_df):
    """
    devide columns by the mean of the reference period and devide by the standard deviation of the reference period
    """

    rp_grouped = reference_period_df.groupby('valid_time').agg('mean').reset_index()

    # only use needed columns
    columns = [col for col in rp_grouped.columns if col not in ['valid_time', 'latitude', 'longitude', 'year', 'month']]

    # calculate climate anomalies
    grouped_anomaly = df.copy()
    for column in columns:
        grouped_anomaly[colum