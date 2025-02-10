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
        grouped_anomaly[column] = (df[column] - rp_grouped[column].mean()) / rp_grouped[column].std()
    return grouped_anomaly




def plot_monthly(df, columns, titles, deg):
    # only keep values from years 1996, 2000, 2004, 2008, 2012, 2016
    df = df[df['year'].isin([1996,  2003,  2010, 2017])]


    for column in columns:
        plt.figure(figsize=(12, 3))
        
        # for each column, use only rows that dont have zero values
        df = df[df[column] != 0]

        # get title from titles dictionary
        title = titles[column]

        #scatter = plt.scatter(df['month'], df[column], c = 'black')
        scatter = plt.scatter(df['month'], df[column], 
                          c=df['year'], cmap='viridis')
    
        # Add colorbar
        plt.colorbar(scatter, label='Year')
        plt.xlabel('month')
        plt.title(title)
        plt.tight_layout()
        plt.show()


def run_temperature_regression(df, columns):
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year'] = df['year'] - 1996
    df['year'] = df['year'] + 1

    for column in columns:
        X = df[['year', 'sin_month', 'cos_month']]
        y = df[column]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        print(model.summary())

def monthly_stats(df, columns):
    """
    Mean and standard deviation grouped by month
    """
    results = {"Month": [], "Mean": [], "Std": []}
    for column in columns:
        monthly = df.groupby("month")[column].agg(["mean", "std"])
        results["Month"] = monthly.index
        results["Mean"] = monthly["mean"]
        results["Std"] = monthly["std"]
    results_df = pd.DataFrame(results)
    return results_df

def monthly_stats_each_year(df, columns):
    results = {"Year": [], "Month": [], "Mean": []}
    for column in columns:
        for year in df["year"].unique():
            monthly = df[df["year"] == year].groupby("month")[column].agg(["mean"])
            results["Year"].extend([year] * 12)
            results["Month"].extend(monthly.index)
            results["Mean"].extend(monthly["mean"])
    results_df = pd.DataFrame(results)
    return results_df


def anomaly_detection(df, columns, threshold):
    results = {"Variable": [], "Year": [], "Month": [], "Z-Score": []}
    for column in columns:
        df['Z-Score'] = zscore(df[column])

        # Define extreme anomalies (e.g., beyond ±2 std deviations)
        anomalies = df[np.abs(df['Z-Score']) > threshold].reset_index(drop=True)
        results["Variable"].extend([column] * len(anomalies))
        results["Year"].extend(anomalies["year"])
        results["Month"].extend(anomalies["month"])
        results["Z-Score"].extend(anomalies["Z-Score"])
    results_df = pd.DataFrame(results)

    # print number of anomalies
    print(f'Number of anomalies: {len(results_df)}')

def rolling_mean(df, columns):
    results = {"Variable": [], "Year": [], "Month": [], "Rolling_Mean": []}
    for column in columns:
        df["Rolling_Mean"] = df[column].rolling(window=12, center=True).mean()
        #df.plot(x="time", y=[column, "Rolling_Mean"], figsize=(10, 5), alpha=0.7)
        results["Variable"].extend([column] * len(df))
        results["Year"].extend(df["year"])
        results["Month"].extend(df["month"])
        results["Rolling_Mean"].extend(df["Rolling_Mean"])
    results_df = pd.DataFrame(results)
    return results_df

def describe_data(df, columns):
    results = {"Variable": [], "Mean": [], "Std": [], "Min": [], "25%": [], "50%": [], "75%": [], "Max": []}
    for column in columns:
        results["Variable"].append(column)
        results["Mean"].append(df[column].mean())
        results["Std"].append(df[column].std())
        results["Min"].append(df[column].min())
        results["25%"].append(df[column].quantile(0.25))
        results["50%"].append(df[column].median())
        results["75%"].append(df[column].quantile(0.75))
        results["Max"].append(df[column].max())
    results_df = pd.DataFrame(results)
    return results_df

def interquartile_range(df, columns):
    results = {"Variable": [], "IQR": []}
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        results["Variable"].append(column)
        results["IQR"].append(iqr)
    results_df = pd.DataFrame(results)
    return results_df

def percentile_analysis(df, columns):
    results = {"Variable": [], "5th Percentile": [], "95th Percentile": [], "Count Below": [], "Count Above": []}
    for col in columns:
        lower_bound = grouped_anomaly[col].quantile(0.05)
        upper_bound = grouped_anomaly[col].quantile(0.95)
        count_lower = (grouped_anomaly[col] < lower_bound).sum()
        count_upper = (grouped_anomaly[col] > upper_bound).sum()
        results["Variable"].append(col)
        results["5th Percentile"].append(lower_bound)
        results["95th Percentile"].append(upper_bound)
        results["Count Below"].append(count_lower)
        results["Count Above"].append(count_upper)
        print(f"Variable: {col}")
        print(f"5th Percentile: {lower_bound}")
        print(f"95th Percentile: {upper_bound}")
        print(f"Count Below: {count_lower}")
        print(f"Count Above: {count_upper}")
        
    results_df = pd.DataFrame(results) 
    return results_df

def decompose_time_series(df, columns):
    # Ensure you have a proper DateTime index:
    result_dict = {"Variable": [], "Trend": [], "Seasonal": [], "Residual": []}
    for column in columns:
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(Day=1))
        df.set_index('date', inplace=True)

        # Choose a variable, for example, sea surface temperature:

        # Decompose with an assumed period of 12 months (adjust if needed)
        stl = STL(df[column].dropna(), period=12)
        result = stl.fit()
        # Plot the decomposed components:
        result.plot()
        # plt.suptitle(f"STL Decomposition of {column}", fontsize=14)
        plt.show()
        # Store the results in a dictionary:
        result_dict["Variable"].append(column)
        result_dict["Trend"].append(result.trend)
        result_dict["Seasonal"].append(result.seasonal)
        result_dict["Residual"].append(result.resid)

    result_df = pd.DataFrame(result_dict)
    return result_df




def plot_yearly(df, columns, titles, deg):
    for column in columns:   
        plt.figure(figsize=(12, 3))
        
        # for each column, use only rows that dont have zero values
        df = df[df[column] != 0]

        title = titles[column]

        plt.scatter(df['year'], df[column], c = 'black')

        # add regression line
        z = np.polyfit(df['year'], df[column], deg)   # Hier 1, 2, 3, oder 4 degs rein?
        p = np.poly1d(z)
        plt.plot(df['year'],p(df['year']), linestyle = "-", c = 'red', linewidth = 0.3)


        plt.xlabel('Year')
        plt.title(title)
        plt.tight_layout()
        plt.show()