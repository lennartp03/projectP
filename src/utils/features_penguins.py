import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from datetime import datetime
from geopy import distance
from sklearn.preprocessing import OneHotEncoder
import json
##

def load_penguins(path):
    """
    Load penguin datasets in one dataframe
    """
    files = os.listdir(path)
    df_original = pd.concat([pd.read_csv(path + '/' + file) for file in files])
    df = df_original.drop(columns=['equinox', 'original_track_id']).reset_index(drop=True)
    return df


def date_formatting(df):
    """
    Replace dd-mm-yyyy or yyyy-mm-dd with dd/mm/yyyy  
    """
    date_target_format = '%d/%m/%Y'

    for i in range(len(df)):
        # check if format is YYYY-MM-DD
        if '-' in df.at[i, 'date_gmt'] and datetime.strptime(df.at[i, 'date_gmt'], '%Y-%m-%d'): 
            df.at[i, 'date_gmt'] = datetime.strptime(df.at[i, 'date_gmt'], '%Y-%m-%d').strftime(date_target_format)
        # check if format is DD-MM-YYYY
        elif '-' in df.at[i, 'date_gmt'] and datetime.strptime(df.at[i, 'date_gmt'], '%d-%m-%Y'):
            df.at[i, 'date_gmt'] = datetime.strptime(df.at[i, 'date_gmt'], '%d-%m-%Y').strftime(date_target_format)
    
    # Create datetime column
    df['datetime'] = df['date_gmt'].astype(str) + ' ' + df['time_gmt'].astype(str)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')

    return df

def geodesic_distance(df):
    """
    Calculate geodesic distance between bird and colony
    Elliposid used: WGS-84  
    """
    df['pos_bird'] = list(zip(df['latitude'], df['longitude']))
    df['pos_colony'] = list(zip(df['lat_colony'], df['lon_colony']))
    df['km_to_colony'] = df.apply(lambda row: distance.distance(row['pos_bird'], row['pos_colony']).km, axis=1)

    return df

def mean_distance(df):
    # Calculate mean distance and std per penguin
    mean_distance_per_penguin = df.groupby('track_id')['km_to_colony'].mean()
    std_distance_per_penguin = df.groupby('track_id')['km_to_colony'].std()

    # Average over mean distances
    mean_distance_overall = mean_distance_per_penguin.mean()
    std_distance_overall = mean_distance_per_penguin.std()
    return mean_distance_overall, std_distance_overall

def calculate_relative_position(df):
    """
    Position relative to colony
    """
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']

    for i in range(len(df)):
        delta_lat = df.at[i,'latitude'] - df.at[i, 'lat_colony']
        delta_lon = df.at[i, 'longitude'] - df.at[i, 'lon_colony']
        degrees_temp = math.atan2(delta_lat, delta_lon)/math.pi*180
        if degrees_temp < 0:
            degrees_final = degrees_temp + 360
        else:
            degrees_final = degrees_temp

        lookup = round(degrees_final/45) # because of 8 possible directions

        df.at[i, 'relative_position'] = directions[lookup]
        
    return df

def direction_of_travel(df):
    """
    Direction of travel
    If bird_id changes: relative position to colony
    If bird_id stays the same: calculate direction of travel
    """
    
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
    df.at[0, 'direction_of_travel'] = df.at[0, 'relative_position']

    for i in range(1, len(df)):
        if df.at[i, 'track_id'] != df.at[i-1,'track_id']:
            df.at[i, 'direction_of_travel'] = df.at[i, 'relative_position']
        else:
            delta_lat = delta_lat = df.iloc[i]['latitude'] - df.iloc[i-1]['latitude']
            delta_lon = df.iloc[i]['longitude'] - df.iloc[i-1]['longitude']
            degrees_temp = math.atan2(delta_lat, delta_lon)/math.pi*180

            if degrees_temp < 0:
                degrees_final = degrees_temp + 360
            else:
                degrees_final = degrees_temp

            lookup = round(degrees_final/45)

            df.at[i, 'direction_of_travel'] = directions[lookup]
    return df


def distance_since_last_measure(df):
    """
    Distance travelled between each measure 
    """
    df['km_since_last_measure'] = 0
    df['km_since_last_measure'] = df['km_since_last_measure'].astype(float)
    for i in range(1, len(df)):
        if df.at[i, 'track_id'] == df.at[i-1, 'track_id']:
            df.at[i, 'km_since_last_measure'] = distance.distance(df.at[i,'pos_bird'], df.at[i-1,'pos_bird']).km
        else:
            df.at[i, 'km_since_last_measure'] = 0
    return df




def distance_direction(df):
    """
    Distace North, East, South, West
    """
    df['delta_km_north'] = 0
    df['delta_km_north'] = df['delta_km_north'].astype(float)
    df['delta_km_south'] = 0
    df['delta_km_south'] = df['delta_km_south'].astype(float)

    for i in range(1, len(df)):
        start = (0, df.at[i-1, 'longitude'])
        finish = (0, df.at[i, 'longitude'])
        delta = distance.distance(start, finish).km
        if df.at[i, 'track_id'] == df.at[i-1, 'track_id']:
            if delta > 0:
                df.at[i, 'delta_km_north'] = delta
            if delta < 0:
                df.at[i, 'delta_km_south'] = delta
        else:
            df.at[i, 'delta_km_north'] = 0
            


    df['delta_km_east'] = 0
    df['delta_km_east'] = df['delta_km_east'].astype(float)
    df['delta_km_west'] = 0
    df['delta_km_west'] = df['delta_km_west'].astype(float)

    for i in range(1, len(df)):
        start = (df.at[i-1, 'latitude'], 0)
        finish = (df.at[i, 'latitude'], 0)
        delta = distance.distance(start, finish).km
        if df.at[i, 'track_id'] == df.at[i-1, 'track_id']:
            if delta > 0:
                df.at[i, 'delta_km_east'] = delta
            elif delta < 0:
                df.at[i, 'delta_km_west'] = delta
        else:
            df.at[i, 'delta_km_east'] = 0

    return df


def time_difference(df):
    df['minutes_since_last_measure'] = 0
    df['minutes_since_last_measure'] = df['minutes_since_last_measure'].astype(float)
    for i in range(1, len(df)):
        if df.at[i, 'track_id'] == df.at[i-1, 'track_id']:
            timestamp_1 = df.at[i-1, 'datetime']
            timestamp_2 = df.at[i, 'datetime']
            df.at[i,'minutes_since_last_measure'] = (timestamp_2 - timestamp_1).total_seconds()/60
        else:
            df.at[i, 'minutes_since_last_measure'] = 0
    return df


def save_df(df, output_path, name=str):
    df.to_csv(output_path + '/' +'penguins_cleaned.zip', compression={'method': 'zip', 'archive_name': name}, index=False)


def average_per_day(df):
    """
    Average time stamps to match weather: One per day
    Try to get a distribution for those accumulated time stamps --> one for each day per unique_id
    """
    df['date_gmt'] = pd.to_datetime(df['date_gmt'], format='%d/%m/%Y')

    aggregate_cols = ['latitude', 'longitude', 'lat_colony', 'lon_colony', 'km_to_colony', 'km_since_last_measure', 
                        'delta_km_north', 'delta_km_south', 'delta_km_east', 
                        'delta_km_west', 'minutes_since_last_measure']

    # Calculate mean
    df_mean = df.groupby(['track_id', 'date_gmt'])[aggregate_cols].mean().reset_index()
    df_mean.columns = ['track_id', 'date_gmt'] + [f'{col}_mean' for col in aggregate_cols]

    # Calculate standard deviation
    df_std = df.groupby(['track_id', 'date_gmt'])[aggregate_cols].std(ddof=0).reset_index()
    df_std.columns = ['track_id', 'date_gmt'] + [f'{col}_std' for col in aggregate_cols]

    #df_min
    df_min = df.groupby(['track_id', 'date_gmt'])[aggregate_cols].min().reset_index()
    df_min.columns = ['track_id', 'date_gmt'] + [f'{col}_min' for col in aggregate_cols]

    #df_max
    df_max = df.groupby(['track_id', 'date_gmt'])[aggregate_cols].max().reset_index()
    df_max.columns = ['track_id', 'date_gmt'] + [f'{col}_max' for col in aggregate_cols]

    # Merge dataframes
    df_penguins_pt1 = pd.merge(df_mean, df_std, on=['track_id', 'date_gmt'])

    df_penguins_pt2 = pd.merge(df_min, df_max, on=['track_id', 'date_gmt'])

    df_penguins_final = pd.merge(df_penguins_pt1, df_penguins_pt2, on=['track_id', 'date_gmt'])

    # Lookup operation to add 'common_name', 'site_name','colony_name' back based on track id
    df_subset = df.drop_duplicates(subset='track_id')
    add_back = ['common_name', 'site_name','colony_name']
    for column in add_back:
        df_penguins_final[column] = df_penguins_final['track_id'].map(df_subset.set_index('track_id')[column])

    return df_penguins_final

### ALTERNATIVE APPROACH; MAYBE MORE ELEGANT (IN TESTING) ###
"""
def average_per_day(df):
    # Convert 'date_gmt' to datetime
    df['date_gmt'] = pd.to_datetime(df['date_gmt'], format='%d/%m/%Y')

    # List of columns to aggregate
    aggregate_cols = [
        'latitude', 'longitude', 'lat_colony', 'lon_colony', 'km_to_colony',
        'km_since_last_measure', 'delta_km_north', 'delta_km_south',
        'delta_km_east', 'delta_km_west', 'minutes_since_last_measure'
    ]

    # Create an aggregation dictionary that applies multiple functions to each column
    agg_funcs = {col: ['mean', 'std', 'min', 'max'] for col in aggregate_cols}

    # Group the DataFrame by 'track_id' and 'date_gmt' and apply the aggregations
    df_agg = df.groupby(['track_id', 'date_gmt']).agg(agg_funcs)

    # Flatten the multi-level column index
    df_agg.columns = [f"{col}_{func}" for col, func in df_agg.columns]
    df_agg = df_agg.reset_index()

    # Create a lookup DataFrame that contains one row per track_id
    additional_cols = ['common_name', 'site_name', 'colony_name']
    df_lookup = df.drop_duplicates(subset='track_id').set_index('track_id')

    for col in additional_cols:
        df_agg[col] = df_agg['track_id'].map(df_lookup[col])

    return df_agg

"""



def save_to_json(df, path, name=str):
    """
    Save unique dates to JSON file to access with API (climate data)
    """
    df['date_gmt'] = df['date_gmt'].astype(str)

    # extract each unique combination of year, month and day in a dictionary
    unique_dates = df['date_gmt'].unique()
    unique_dates_dict = {}

    # suppose preprocessing of date already happened
    for date in unique_dates:
        year = date.split('-')[0]
        month = date.split('-')[1]
        day = date.split('-')[2]
        unique_dates_dict[date] = {'year': year, 'month': month, 'day': day}


    #df_encoded.to_csv(path_json + '/' + 'penguins_cleaned.csv', index=False, compression='zip')
    json.dump(unique_dates_dict, open(path + '/' + name, 'w'))
    print('JSON file saved to ' + path)




### ======= FUNCTIONS UNSURE IF NEEDED ======= ###

def mean_deviation_colony(df, df_penguins_final):
    species = df['common_name'].unique()
    years = df_penguins_final['year'].unique()
    results_species_colony = []
    for year in years:
        for specie in species:
            for colony in df[df['common_name'] == specie]['colony_name'].unique():
                df_specie = df_penguins_final[(df_penguins_final['common_name'] == specie) & (df_penguins_final['colony_name'] == colony) & (df_penguins_final['year'] == year)]
                # Calculate mean deviation manually
                mean_value = df_specie['km_to_colony_mean'].mean()
                # mean_deviation = (df_specie['km_to_colony_mean'] - mean_value).abs().mean()
                results_species_colony.append({'year': year, 'species': specie, 'colony': colony, 'mean distance': mean_value})
    results_species_colony = pd.DataFrame(results_species_colony)
    return results_species_colony



def plot_yearly_species(df, deg, species):
    df['date_gmt'] = pd.to_datetime(df['date_gmt'])
    df['year'] = df['date_gmt'].dt.year
    df['month'] = df['date_gmt'].dt.month
    for specie in species:
        plt.figure(figsize=(12, 3))
        
        # Filter the dataframe for the current species
        df_species = df[df['common_name'] == specie].copy()  # Use .copy() to avoid modifying df
        # Group by year and compute mean of km_to_colony
        df_species = df_species.groupby('year', as_index=False)['km_to_colony'].mean()
        title = specie

        plt.scatter(df_species['year'], df_species['km_to_colony'], c='black')


        if len(df_species) > deg:  # Ensure enough points for regression
            z = np.polyfit(df_species['year'], df_species['km_to_colony'], deg)
            p = np.poly1d(z)
            plt.plot(df_species['year'], p(df_species['year']), linestyle="-", c='red', linewidth=0.8)


        plt.xlabel('Year')
        plt.title(title)
        plt.tight_layout()
        plt.show()

def spread_species(df, species):
    for specie in species:
        print(f'\n========{specie}========')
        for colony in df[df['common_name'] == specie]['colony_name'].unique():
            df_specie = df[(df['common_name'] == specie) & (df['colony_name'] == colony)]
            spread = df_specie['km_to_colony_mean'].std(ddof=1)
            print(colony, spread)

def calculte_mean_deviation(df):
    species = df['common_name'].unique()
    for specie in species:
        print(f'\n========{specie}========')
        for colony in df[df['common_name'] == specie]['colony_name'].unique():
            df_specie = df[(df['common_name'] == specie) & (df['colony_name'] == colony)]
            
            # Calculate mean deviation manually
            mean_value = df_specie['km_to_colony_mean'].mean()
            mean_deviation = (df_specie['km_to_colony_mean'] - mean_value).abs().mean()
            print(colony, mean_deviation)


def check_availability(df, threshold):
    available_years = [str(x) for x in range(1996, 2018)]
    available_years = np.array(available_years)

    # Ensure the 'year' column in df_penguins_final is of type string
    df['year'] = df['year'].astype(str)

    # For each species and colony, see if there is data for more than 10 years
    species = df['common_name'].unique()
    results_species_colony = []

    for specie in species:
        for colony in df[df['common_name'] == specie]['colony_name'].unique():
            years_with_data = df[(df['common_name'] == specie) & (df['colony_name'] == colony)]['year'].unique()
            if len(years_with_data) >= threshold:
                print(f'{specie} , {colony} has data for more than {threshold} years')
            else:
                print('--')