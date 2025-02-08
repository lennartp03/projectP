import cdsapi
import json
import numpy as np
from collections import defaultdict

path_json = '/Users/philipp/Documents/02_Master_Uni/Uni_Tübingen/Semester_1/06 Data Literacy/02 Project/projectP/era5_datasets'
file = open(path_json + '/' + 'unique_dates_dict.json', 'r')
unique_dates_dict = json.load(file)

unique_dates = unique_dates_dict.keys()
# convert to np array
np_array = np.array([unique_dates_dict[date] for date in unique_dates]) 

grouped_days = defaultdict(set)
for entry in np_array:
    year_month = f"{entry['year']}-{entry['month']}"
    grouped_days[year_month].add(entry['day'])

# Convert the defaultdict to a regular dictionary for easier use
unique_combs = {key: sorted(value) for key, value in grouped_days.items()}

for item in unique_combs:
    year = str(item.split('-')[0])
    month = str(item.split('-')[1])
    days = [f'{day}' for day in unique_combs[item]]


    dataset = "derived-era5-single-levels-daily-statistics"
    request = {
        "product_type": "ensemble_mean",
        "variable": [
        "2m_temperature",
        "total_precipitation"
        "sea_surface_temperature",
        "snow_density",
        "snow_depth",
        "sea_ice_cover",
        "mean_snowmelt_rate"
    ],
        "year": str(year),
        "month": [
            str(month)
        ],
        "day": days,
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": [-38, -162, -86, -12]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()