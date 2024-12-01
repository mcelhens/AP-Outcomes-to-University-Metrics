# AP-Outcomes-to-University-Metrics

Authors (alphabetically): Prabhat Devkota, Shrabana Hazra, Jung-Tsung Li, Shannon J. McElhenney, Raymond Tana.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ap-outcomes.streamlit.app/)

## Project Description
                    
This project was designed to investigate the potential relationship between **[AP exam](https://apstudents.collegeboard.org/what-is-ap) performance** and the **presence of nearby universities**. It was initially hypothesized that local (especially R1/R2 or public) universities would contribute to better pass rates for AP exams in their vicinities as a result of their various outreach, dual-enrollment, tutoring, and similar programs for high schoolers. We produce a predictive model that uses a few features related to university presence, personal income, and population to predict AP exam performance.
            
## Background

AP exams are standardized tests widely available at high schools across the United States. During the 2022-2023 school year, [79\%](https://arc.net/l/quote/ewvgnupe) of all public high school students attended schools offering at least five AP courses. These exams are popular for their potential to earn college credits during high school by achieving high scores. In fact, high scores in most AP exams are eligible to receive college credits at roughly [2000](https://apcentral.collegeboard.org/media/pdf/program-summary-report-2024.pdf) higher-education institutions. 

AP exams are scored on a whole number scale between 1 (lowest) and 5 (highest). A student is said to *pass* their AP exam if they score a 3 or higher on the exam. The *pass rate* of a locality would be the proportion of AP exams passed out of all exams taken by its students during a single year. AP outcomes are often correlated to measures of socioeconomic factors: a [recent study](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4574500) confirmed that negative socioeconomic factors have a strong negative influence on exam scores; as well as being a non-native English language speaker. 

Beyond these socioeconomic factors, we wished to measure the strength of the effect of universities on AP outcomes. Without a clear source of data on all high school outreach programs offered by US universities, we made use of the various classifications offered by the [Carnegie Classifications of Institutions of Higher Education](https://carnegieclassifications.acenet.edu/). Of particular interest included R1 and R2 (i.e., doctoral with very high or high research activity, respectively), public, or private institutions. Other minority-serving aspects were also considered, such as historically Black, Hispanic-serving, and tribal colleges.

## Description of the files
- `broader_carnegie_classificatins.ipynb` contains exploration of the Carnegie classification data file `data/CCIHE2021-PublicData.xlsx`.
- `data_fixing.ipynb` and `Poorly_formatted_data_fixer.ipynb` contain some codes we used to process the badly presented AP datasets we obtained from various sources (esp. College Board and Department of Education of various states).
- `data_loaders.py` contains some functions that we routinely use to load various datasets in clean format. `distance_dorm_descriptors.py` contains the functions that we used to compute the distance, enrollment and dormrooms related features.
- `features.ipynb` contains some exploration of relation between various classification categories in Carnegie data file `data/CCIHE2021-PublicData.xlsx`. `features.py` is a supporting file to that notebook.
- `four_states_combined_model.ipynb` contains our modelling for the combined county-by-county AP data of the four states GA, MA, NC and WI.
- `Initial_analysis_national_level.ipynb` contains exploratory data analysis, exploring relation of AP pass rate with socio-economic and university count related features, done on state-by-state level, part of which led us to conclude why state-by-state was not a right resolution for our purposes.
- `models.py` contains some functions that allows one to explore our Machine Learning models.
- `National_initial_investigations.ipynb` contains more national level exploratory data analysis, more geared towards minority considerations, again part of which led us to believe state-by-state data was not a right resolution to look at.

## Methodology

We first started with the national level state by state data on AP scores obtained from College Board's website as well as the data from Carnegie on classification of universities. The data available in College Board was quite limited -- it was only for four years: 2012, 2017, 2021 and 2022. And from the exploratory data analysis, it became clearer to us that state-by-state is probably not the resolution that we want for our investigations. For more details, we refer to the jupyter notebooks `Initial_analysis_national_level.ipynb` and `National_initial_investigations.ipynb`. As such, we opted to search for county-by-county data in the Department of Education websites of various states. The four states -- Georgia, Massachusetts, North Carolina and Wisconsin -- had fairly comprehensive datasets.

Because different states have different properties, we decided it would be better to run our analysis on each state separately first. The data processing, analysis and modeling for different states are present in their corresponding directory (the folder `Wisconsin` contains the work for Wisconsin state, for example). *So we refer to the corresponding directory for detailed data processing, exploratory data analysis and modeling for each state.*

Once we were done with our work on each state, we combined the datasets from the four states to one dataframe and create a combined model for the four states. Our rationale was that while separate modeling would be preferable for different state, we would probably get more mileage out of our models on the states other than the four for which we had the data, if we combined them together. And the states that we have are quite diverse in characteristics. So, our hope is that while not perfect, the combined datasets will have better chance of encompassing all the counties in the 50 states in predictive modeling. The modeling is performed in the jupyter notebook `four_states_combined_models.ipynb` in the main directory.

## Description of features and target variables
The features and target variables for combined states model are same as that for the separate models for the individual states. Our target variable is the AP pass rate, which is defined as the percentage of exams scoring 3 or higher. The corresponding columns are named `PERCENT_3_OR_ABOVE`. There are altogether 17 numerical features that we use. Two are population and per capita income. 15 of the others are related to nearby universities. We look at universities in five categories: R1/R2 (defined as having very high or high research activity in Carnegie classification), public, private not for profit, landgrant and STEM (defined as having at least one doctoral/research degree offered in STEM field). For each category, we consider five closest universities to the given county (or rather, its rough geographical center) and compute average distance, annual enrollment (in academic year 2019/20) and number of dorm rooms to/at those universities. Additionally, we have one categorical feature `State`, which is encoded by three one hot encoders `Is_GA`, `Is_MA` and `Is_WI` to distinguish the states the corresponding data point belongs to.

## Summary of the models
We used `sklearn` and `xgboost` to make several predictive regression models: OLS Linear Regression (called full model), AdaBoost Regressor Model, Random Forest Regressor Model, and XGBoost Model. From these, we observed XGBoost performed the best. So, to further refine the XGBoost model, we performed some hyperparameter tuning. And to make the model more efficient, we also did some Principal Component Analysis (setting `n_components = 0.95`). From the hyperparameter tuning, we found the best hyperparameters to be: `n_estimators = 950`, `max_depth = 4`, and `learning_rate = 0.03`. And the PCA reduced the 20 features to 10 principal components.

The performance of various models (as compared to the baseline "take the average and call it a day" model) is summarized as follows:

| Model                                 | RMSE          | R-squared    |
| ------------------------------------- | ------------- | ------------ |
| Baseline                              | 19.423        | 0            |
| OLS Liner Regression                  | 14.816        | 0.436        |
| XGBoost (w/o hyperparameter tuning)   | 10.994        | 0.679        |
| AdaBoost Model                        | 13.455        | 0.520        |
| Random Forest Model                   | 11.343        | 0.657        |
| XGBoost (w/ hyperparameter tuning)    | 10.546        | 0.702        |
| XGBoost (PCA + hyperparameter tuning) | 11.152        | 0.669        |

As we can see, the hyperparameter-tuned XGBoost model performs the best, but the hyperparameter-tuned XGBoost with PCA was not far behind. Because of the reduction of the features, we expect the model with PCA to be more efficient and potentially also minimize overfitting issues. So, we decided to choose it as our final model to test on the testing data. The performance is as follows:

| Model                                 | RMSE          | R-squared    |
| ------------------------------------- | ------------- | ------------ |
| XGBoost (PCA + hyperparameter tuning) | 9.227         | 0.773        |

Surprisingly, the performance is even better than on cross-validation!

## Data references and required packages

### Data references in no particular style

1. Indiana University Center for Postsecondary Research (n.d.). The Carnegie Classification of Institutions of Higher Education, 2021 edition, Bloomington, IN.
2. United States Census Bureau -- the United States Office of Management and Budget. Metropolitan and Micropolitan. Core based statistical areas (CBSAs), metropolitan divisions, and combined statistical areas (CSAs). July, 2023.
3. GA Governor's Office of Student Achievement, Data Dashboards, Advanced Placement (AP) Scores, Years 2019, 2020, 2021, 2022, 2023
4. GA Department of Community Affairs Office of Research, Municipalities by County
5. MA Department of Elementary and Secondary Education, Statewide Reports -- Advanced Placement Performance. Academic year: 2019-2023.
6. Wisconsin Department of Public Instruction, Advanced Placement (AP) Scores, 2019-2023: https://dpi.wi.gov/wisedash/download-files
7. Federal Reserve Bank of St. Louis
8. The Bureau of Economic Analysis of the United States Department of Commerce
9. Advanced Placement (AP) -- College Board

### Python packages featured in this project

- `streamlit`
- `sklearn`
- `pandas`
- `statsmodels`
- `xgboost`
- `plotly`
- `matplotlib`
- `seaborn`
- `folium`
- `streamlit_folium`
- `shapely`
- `geopy`
- `geopandas`
- `numpy`
- `scipy`


