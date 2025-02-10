# Project P

## Overview

This repository contains the materials for the practical project in **Data Literacy**, which aims to analyze and predict the movement of pneguin colonies in the polar regions based on short-term weather forecasts. This repository includes all relevant as well as additional code that was used for conducting the analysis. It therefore serves as a practical foundation to the project report.

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

## Project Structure (Tentative)

```
projectp/
├── data/
│   ├── era5_daily/
│   ├── era5_monthly/
│   ├── era5/
│   └── penguin/
├── notebooks/
│   ├── *.ipynb
│   └── additional/
├── src/
│   ├── model/
│   │   ├── interpretation.py
│   │   └── regression.py
│   ├── utils/
│   │   └── preprocessing_lr.py
│   └── visualization/
│       └── plots_lr.py
├── README.md
├── LICENSE
└── requirements.txt
```

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
