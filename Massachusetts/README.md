# AP Outcomes data processing and modelling for Massachusetts
This folder contains our codes for machine learning models for Massachusetts. The datasets are in ../data/Massachusetts folder of the main directory.

### Description of the files
1. In `MA_model_selection.ipynb`, we create machine learning models and identify XGBoost as the best model for MA dataset.
2. In `MA_XGBoost_SHAP.ipynb`, we show how we use the XGBoost to train the MA AP performance. We further show the feature importance and their correlation with the AP outcome.
3. In `MA_plotly.ipynb`, we use the interactive plotly plots to show the linear relationship between AP outcomes and each of the top five features.

### Description of features and target variables
- Our target variable is the AP pass rate, which is defined as the percentage of exams scoring 3 or higher. The corresponding columns are named 'PERCENT_3_OR_ABOVE'. There are altogether 17 features that we use. Two are population and per capita income. The other 15 are related to nearby universities. We look at universities in five categories: R1/R2 (defined as having very high or high research activity in Carnegie classification), public, private not for profit, landgrant and STEM (defined as having at least one doctoral/research degree offered in STEM field). For each category, we consider five closest universities to the given county (or rather, its rough geographical center) and compute average distance, annual enrollment (in academic year 2019/20) and number of dorm rooms to/at those universities.
- The distance is computed by using `distance` function in `geopy`. The county location is obtained from https://simplemaps.com/data/us-counties; the corresponding dataset is saved as ../data/uscounties.csv in the main directory. The locations for the universities are mostly acquired via Google geocoders.
- We have combined the MA AP data from 2018 academic year to 2022 academic year. We use this five-year-data for data training.
- In `MA_XGBoost_SHAP.ipynb`, we have found that the top five important features for MA AP performance are (1) Per capita income, (2) Population, (3) Average distance to five closest public universities, (4) Average distances to five closest private not-for-profit universities, and (5) Average annual enrollment for the five closest land grant universities.
- While we have more columns like total count of AP exams, total students count, etc. in our dataframe, we decided not to use them as features for two reasons: one, they are likely highly correlated with population, and two, not all states' datasets have all of those features.

### Description of the models
- We first did some statistical analysis with statsmodels. We created linear regression models one with all 17 features (called full model), another with only population and per capita income as features (called nonuni model) and another with only the 15 university related variables as features (called uni model). We then looked at the p-values of full model compared to both uni model and nonuni model. Both of the p-values were extremely small.
- Next, we used sklearn and xgboost to make several predictive regression models: OLS linear regression (called full model), Ridge regression model, Adaboost regressor model, Random Forest regressor model and xgboost model. The average root mean squared errors (RMSE),  RMSEstandard deviations, and R-squared coefficients across the 5-fold cross validation and comparison with the baseline model is summarized as follows:

| Model               | RMSE          | RMSE STD | R-squared | 
| ------------------- | ------------- | -------- | --------- |
| Baseline model      | 17.390        | 0.473    | 0.000     |
| Full model          | 12.496        | 0.265    | 0.483     |
| Ridge model         | 12.514        | 0.271    | 0.482     |
| PCA (0.95) model    | 12.607        | 0.297    | 0.474     |
| Adaboost model      | 10.710        | 0.030    | 0.620     |
| Random Forest model | 9.169         | 0.027    | 0.721     |
| XGBoost model       | 9.078         | 0.040    | 0.725     |

- XBGoost model performed the best, so we choose it to be our model for Massachusetts to use on the testing data.
