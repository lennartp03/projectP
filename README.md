# Project P

## Overview

This repository contains the materials for the practical project in **Data Literacy**, which aims to analyze and predict the movement of penguin colonies in the polar regions based on short-term weather forecasts. This repository includes all relevant as well as additional code that was used for conducting the analysis. It therefore serves as a practical foundation to the project report.

## Research Question

Can we analyze and predict the movement of penguin colonies based on short-term weather forecasts? How do the effects of climate change affect these movement patterns?

### Motivation

The polar regions have been severely impacted by global warming, leading to coastal shrinkage due to rising sea levels and ice sheet melting. The movements of seabird colonies, such as penguins, are likely influenced by these environmental changes. This project investigates whether climate measurements, such as temperature and inclement weather, can be used to predict their movements.

## Dataset

The project uses the following datasets (see project report for details):

- **Copernicus ERA5 Weather Data** (https://cds.climate.copernicus.eu)
- **Seabird Tracking Database** (https://www.seabirdtracking.org/)

The repository does not contain any data in the respective `/data` folder, as both - the penguin and the ERA5 data - must be requested from the respective sources.

## Methods and Analysis

1. **Descriptive and Exploratory Data Analysis**
2. **Species-Level Prediction of Mean Distance to Colony**
3. **Colony-Level Time Series Prediction of Mean Distance to Colony**

## Project Structure

```
projectp/
├── data/
│   ├── era5_daily/
│   ├── era5_monthly/
│   ├── era5_reference_period/
│   ├── era5/
│   └── penguin/
├── notebooks/
│   ├── api_retrieve_weather_data.ipynb
│   ├── merge_penguin_era5.ipynb
│   ├── penguin_feature_engineering.ipynb
│   ├── prediction_penguin_adelie.ipynb
│   ├── prediction_time_series.ipynb
│   └── additional/
│       └── *.ipynb
├── src/
│   ├── data/
│   │   └── api_functions.py
│   ├── model/
│   │   ├── interpretation.py
│   │   ├── non_time_series_model.py
│   │   └── regression.py
│   ├── utils/
│   │   ├── features_monthly.py
│   │   ├── features_penguins.py
│   │   └── preprocessing_lr.py
│   └── visualization/
│       └── plots_lr.py
├── README.md
├── LICENSE
└── requirements.txt
```

Context and description of the main directories:

**/data**

- Under the data directory there are respective placeholders that were used for the daily (and monthly) aggregated ERA5 weather data, the baseline timespan and also the penguin datasets. The final processed penguin dataset is name `penguins_final.csv`, the merged version with the weather covariates used for model training is `penguins_final_with_era5.csv`. As stated before, the data is not included in this repository due to restrictions.

**/notebooks**

- **api_retrieve_weather_data.ipynb** – Retrieves the weather data from the Copernicus ERA5 API.
- **merge_penguin_era5.ipynb** – Merges and interpolates the ERA5 weather data on the penguin tracking data.
- **penguin_feature_engineering.ipynb** – Includes steps for feature engineering on the penguin dataset.
- **prediction_penguin_adelie.ipynb** – Focuses on the general (time-series-independent) prediction of the daily mean distance to the colony for Adelie Penguins based on weather covariates.
- **prediction_time_series.ipynb** – Implements time-series forecasting (using Ridge and Lasso regression) for predicting colony movement.
- **additional/\*** – Contains supplementary notebooks that were used for exploratory analyses and contain further analyses and experiments - that were not part of the final reported results.

**/src**

- **src/data/api_functions.py** – Contains functions to interact with ERA5 APIs for retrieving weather data.
- **src/model/interpretation.py** – Includes script for computing SHAP values.
- **src/model/non_time_series_model.py** – Implements methods used in the `prediction_penguin_adelie.ipynb` notebook.
- **src/model/regression.py** – Implements methods used in the `prediction_time_series.ipynb` notebook.
- **src/utils/features_monthly.py** – Contains utility functions for extracting and processing monthly features from the data.
- **src/utils/features_penguins.py** – Provides functions to process and extract features specifically from the penguin tracking dataset.
- **src/utils/preprocessing_lr.py** – Includes data preprocessing routines tailored for linear regression analyses used in `prediction_time_series.ipynb`.
- **src/visualization/plots_lr.py** – Contains plotting functions for visualizing results.

## Requirements

To run this project, create a `venv` and install the required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

NOTE: This repository contains barely no datasets due to the restrictions of the Copernicus Climate Data Store and the Seabird Tracking Database. The data must be requested from the respective sources. Other than that, the logic cannot be executed without the data.

## Results and Reporting

The final report follows a structured format and was sent out seperately. It includes the following sections:

- **Abstract**
- **Introduction**
- **Methods**
- **Results**
- **Discussion & Limitations**

## Acknowledgments

- Copernicus Climate Data Store
- Seabird Tracking Database

## License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.
