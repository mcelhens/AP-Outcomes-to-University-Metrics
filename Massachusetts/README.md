# AP Outcomes data processing and modelling for Massachusetts
This folder contains our codes for machine learning models for Massachusetts. The datasets are in ../data/Massachusetts folder of the main directory.

### Description of the files
1. In `MA_model_selection.ipynb`, we trained the MA data using the RandomForest, AdaBoost, and XGBoost models. We have found that XGBoost performs the best in terms of getting the smallest RMSE valuses. As a result, we use XGBoost for the data analysis of the MA AP performance.
2. In `MA_XGBoost_SHAP.ipynb`, we show how we use the XGBoost to train the MA AP performance. We further show the feature importance and their correlation with the AP outcome.
3. In `MA_plotly.ipynb`, we use the interactive plotly plots to show the linear relationship between AP outcomes and each of the top five features.

### Description of features and target variables
- Our target variable is the AP pass rate, which is defined as the percentage of exams scoring 3 or higher. The corresponding columns are named 'PERCENT_3_OR_ABOVE'. There are altogether 17 features that we use. Two are population and per capita income. The other 15 are related to nearby universities. We look at universities in five categories: R1/R2 (defined as having very high or high research activity in Carnegie classification), public, private not for profit, landgrant and STEM (defined as having at least one doctoral/research degree offered in STEM field). For each category, we consider five closest universities to the given county (or rather, its rough geographical center) and compute average distance, annual enrollment (in academic year 2019/20) and number of dorm rooms to/at those universities.
- The distance is computed by using `distance` function in `geopy`. The county location is obtained from https://simplemaps.com/data/us-counties; the corresponding dataset is saved as ../data/uscounties.csv in the main directory. The locations for the universities are mostly acquired via Google geocoders.
- We have combined the MA AP data from 2018 academic year to 2022 academic year. We use this five-year-data for data training.
- In `MA_XGBoost_SHAP.ipynb`, we have found that the top five important features for MA AP performance are (1) Per capita income, (2) Population, (3) Average distance to five closest public universities, (4) Average distances to five closest private not-for-profit universities, and (5) Average annual enrollment for the five closest land grant universities.
- While we have more columns like total count of AP exams, total students count, etc. in our dataframe, we decided not to use them as features for two reasons: one, they are likely highly correlated with population, and two, not all states' datasets have all of those features.

### Description of the models
- We used Adaboost regressor model, Random Forest regressor model and xgboost model. The average root mean squared errors is summarized as follows:

| Model               | RMSE          |
| ------------------- | ------------- |
| Adaboost model      | 9.329         |
| Random Forest model | 7.927         |
| XGBoost model       | 7.813         |

- XBGoost model performed the best, so we choose it to be our model for Massachusetts to use on the testing data.
