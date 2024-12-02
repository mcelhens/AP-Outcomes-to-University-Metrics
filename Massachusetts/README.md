# AP Outcomes Data Processing and Modeling for Massachusetts

This repository contains the codes and resources used for machine learning models to analyze and predict Advanced Placement (AP) outcomes for Massachusetts. The datasets are located in the `../data/Massachusetts` directory within the main repository.

---

## Project Overview

This project aims to model and analyze AP outcomes in Massachusetts using various statistical and machine learning techniques. The primary target variable is the **AP pass rate**, defined as the percentage of exams scoring 3 or higher.

---

## File Descriptions

1. `MA_model_selection.ipynb`<br>
   Develops and evaluates various machine learning models to determine that XGBoost is the best model for the Massachusetts dataset.

2. `MA_XGBoost_SHAP.ipynb`<br>
   Trains the Massachusetts AP dataset using XGBoost, analyzes feature importance, and explores feature correlations with AP outcomes using SHAP.

3. `MA_plotly.ipynb`<br>
   Uses Plotly for interactive visualizations showing the linear relationships between AP outcomes and the top five significant features.

---

## Features and Target Variables

### Target Variable
- **PERCENT_3_OR_ABOVE**: The percentage of exams scoring 3 or higher.

### Features
The model utilizes 17 features, categorized as follows:

1. **Demographic Features**:
   - **Population**
   - **Per capita income**

2. **University-Related Features** (for the five nearest universities per county, categorized by type):
   - Average distance
   - Annual enrollment
   - Number of dormitory beds

   University types:
   - R1/R2 universities
   - Public universities
   - Private not-for-profit universities
   - Land-grant universities
   - STEM universities

### Data Sources and Computations
- **County Locations**: Derived from [SimpleMaps US Counties dataset](https://simplemaps.com/data/us-counties) and saved in `../data/uscounties.csv`.
- **University Locations**: Primarily acquired using Google Geocoders.
- **Distance Calculation**: Computed using the `geopy.distance` function.
- **Training Data**: Combines Massachusetts AP data from the 2019–2023 academic years.

### Top 5 Features Identified by SHAP
1. Per capita income
2. Population
3. Average distance to the five closest public universities
4. Average distance to the five closest private not-for-profit universities
5. Average annual enrollment for the five closest land-grant universities

Features like total AP exam count and student count were excluded due to potential multicollinearity with population and inconsistent availability across datasets.

---

## Statistical and Machine Learning Models

### Statistical Analysis (Using `statsmodels`)
- **Linear Regression Models**:
  - **Full Model**: Includes all 17 features.
  - **Non-University Model**: Includes only population and per capita income.
  - **University Model**: Includes only university-related variables.

- Significant p-values in the full model comparisons confirmed the importance of both university-related and non-university-related variables.

### Machine Learning Models (Using `sklearn` and `xgboost`)
We evaluated the following models using 5-fold cross-validation:

| Model                 | RMSE  | RMSE STD | R-squared |
|-----------------------|--------|----------|-----------|
| **Baseline**          | 17.390 | 0.473    | 0.000     |
| **Full Model**        | 12.496 | 0.265    | 0.483     |
| **Ridge Regression**  | 12.514 | 0.271    | 0.482     |
| **PCA (0.95)**        | 12.607 | 0.297    | 0.474     |
| **Adaboost**          | 10.710 | 0.030    | 0.620     |
| **Random Forest**     | 9.169  | 0.027    | 0.721     |
| **XGBoost**           | 9.078  | 0.040    | 0.725     |

- **Best Model**: XGBoost, chosen for its superior performance on RMSE and R-squared.

---

## Conclusion

XGBoost was identified as the best-performing model for predicting AP outcomes in Massachusetts. The repository includes the codes for model training, feature importance analysis, and interactive visualizations. This framework can be extended to other states or datasets for similar analyses.