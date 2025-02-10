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
##


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
        grouped_anomaly[column] = (df[column] - rp_grouped[column].mean()) #/ rp_grouped[column].std()
    return grouped_anomaly

### ========================= ###

def calculate_anomalies_monthly(df, reference_period_df):
    """
    Calculate monthly climate anomalies based on reference period.
    """
    # Identify the variable columns (all except metadata)
    variable_columns = [col for col in df.columns if col not in ['valid_time', 'latitude', 'longitude', 'year', 'month']]
    
    df_anom = df.copy()
    
    # Get year and month from the valid_time column
    df_anom['year'] = df['valid_time'].dt.year
    df_anom['month'] = df['valid_time'].dt.month


    # Process month-by-month
    for month in range(1, 13):
        # Subset the reference period for the current month
        ref_subset = reference_period_df[reference_period_df['month'] == month]
        monthly_mean = ref_subset[variable_columns].mean()
        monthly_std = ref_subset[variable_columns].std()
        
        # Create a mask for rows corresponding to the current month
        mask = df['month'] == month
        
        # Apply the anomaly calculation for the current month
        df_anom.loc[mask, variable_columns] = (df.loc[mask, variable_columns] - monthly_mean) #/ monthly_std
    
    return df_anom

def plot_two_variables_stacked(df, var1, var2, titles, selected_years=[1996, 2003, 2010, 2017]):
    """
    Create a single figure with two vertically stacked subplots: one for var1 and one for var2.
    Instead of a colorbar, each year is plotted in a distinct color with a shared legend.
    """
    # Filter the DataFrame for the selected years
    df_filtered = df[df['year'].isin(selected_years)].copy()
    
    # Define a color for each year (using tab10, which gives up to 10 distinct colors)
    unique_years = sorted(df_filtered['year'].unique())
    cmap = plt.cm.viridis
    # For a discrete set of colors, map each year index to a normalized value between 0 and 1:
    year_colors = {year: cmap(i / (len(unique_years)-1)) for i, year in enumerate(unique_years)}

    
    # Create subplots (2 rows, 1 column)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(17, 6), sharex=True)
    
    # --- Plot for var1 (e.g., Sea Ice Concentration) ---
    ax1 = axes[0]
    for year in selected_years:
        # Filter data for the current year and exclude rows where the value is zero
        df_year = df_filtered[(df_filtered['year'] == year) & (df_filtered[var1] != 0)]
        ax1.plot(df_year['month'], df_year[var1],
                    color=year_colors[year],
                    label=str(year), marker = 'o')  # use the year as label
    ax1.set_title(f'{titles.get(var1, var1)} by Month')
    ax1.set_ylabel('Difference (% of cover)')
    # ax1.set_xticks(range(1, 13))
    # ax1.set_xticks([])
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    
    # --- Plot for var2 (e.g., Sea Surface Temperature) ---
    ax2 = axes[1]
    for year in selected_years:
        df_year = df_filtered[(df_filtered['year'] == year) & (df_filtered[var2] != 0)]
        ax2.plot(df_year['month'], df_year[var2],
                    color=year_colors[year],
                    label=str(year), marker = 'o')  # label each year
    ax2.set_title(f'{titles.get(var2, var2)} by Month')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Difference (Kelvin)')
    ax2.set_xticks(range(1, 13))
    
    # Create a single legend for the entire figure.
    # We'll take the handles and labels from the first subplot (they are the same for both)
    handles, labels = ax1.get_legend_handles_labels()
    # Place the legend at the top center of the figure, outside the subplots.
    fig.legend(handles, labels, loc='upper center', ncol=len(selected_years), frameon=False)
    
    # Adjust the layout to leave space for the legend
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



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
        lower_bound = df[col].quantile(0.05)
        upper_bound = df[col].quantile(0.95)
        count_lower = (df[col] < lower_bound).sum()
        count_upper = (df[col] > upper_bound).sum()
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