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
            delta_lat 