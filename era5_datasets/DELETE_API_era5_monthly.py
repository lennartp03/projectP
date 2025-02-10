import cdsapi
import os

def get_monthly_era5_data(period,  target_directory):
    if period == 'reference':
        years = [
            "1961", "1962", "1963",
            "1964", "1965", "1966",
            "1967", "1968", "1969",
            "1970", "1971", "1972",
            "1973", "1974", "1975",
            "1976", "1977", "1978",
            "1979", "1980", "1981",
            "1982", "1983", "1984",
            "1985", "1986", "1987",
            "1988", "1989", "1990"
            ]
    elif period == 'normal':
        years = [
            "1991", "1992",
            "1993", "1994", "1995",
            "1996", "1997", "1998",
            "1999", "2000", "2001",
            "2002", "2003", "2004",
            "2005", "2006", "2007",
            "2008", "2009", "2010",
            "2011", "2012", "2013",
            "2014", "2015", "2016",
            "2017", "2018", "2019",
            "2020"
        ]
    dataset = "reanalysis-era5-single-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "2m_temperature",
            "sea_surface_temperature",
            "total_precipitation",
            "sea_ice_cover",
            "snow_density",
            "snow_depth",
            "mean_snowmelt_rate"
        ],
        "year": years,
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [-38, -162, -86, -12]
    }
    target_file = f"era5_{period}.zip"
    target_directory = target_directory
    target_path = os.path.join(target_directory, target_file)
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(target_path)