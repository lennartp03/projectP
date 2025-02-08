# Project P

## Overview

This repository contains the materials for the practical project in **Data Literacy**, which aims to analyze and predict the movement of seabird colonies in the polar regions based on short-term weather forecasts and the impact of ice cap shrinkage.

## Research Question

Can we analyze and predict the movement of seabird colonies based on short-term weather forecasts? How does the shrinkage of ice caps and coastal regions affect these movement patterns?

### Motivation

The polar regions have been severely impacted by global warming, leading to coastal shrinkage due to rising sea levels and ice sheet melting. The movements of seabird colonies, such as penguins, are likely influenced by these environmental changes. This project investigates whether climate measurements, such as temperature and inclement weather, can be used to predict seabird movements.

## Dataset

The project uses datasets from the **Copernicus Climate Data Store** and other sources:

- **Copernicus ERA5 Weather Data**
- **Seabird Tracking Database**

## Methods and Analysis

1. **Descriptive and Exploratory Data Analysis**
2. **Correlation Analysis** between movement tracks and influencing factors (e.g., temperature, ice cap shrinkage, coastal erosion)
3. **Movement Extraction & Modeling**
   - Extract movement coordinates of seabird colonies from the Seabird Tracking Database.
   - Train statistical models to relate movement data to short-term weather forecasts.

## Project Structure (Tentative)

```
📂 projectp
├── 📂 data                  # Raw and processed datasets
├── 📂 notebooks             # Jupyter Notebooks for analysis
├── 📂 src                   # Source code for data processing and modeling
├── 📂 reports               # Final project reports and documentation
├── README.md                # Project overview (this file)
└── requirements.txt         # Python dependencies
```

## Requirements

To run this project, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/lennartp03/projectp.git
   cd projectp
   ```
2. Run the Jupyter Notebooks in `notebooks/` to explore and analyze the data.
3. Execute scripts in `src/` to preprocess data and train models.

## Results and Reporting

The final report follows a structured format:

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
