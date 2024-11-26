# AP-Outcomes-to-University-Metrics

Authors (alphabetically): Prabhat Devkota, Shrabana Hazra, Jung-Tsung Li, Shannon J. McElhenney, Raymond Tana.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ap-outcomes.streamlit.app/)

## Project Description
                    
This project was designed to investigate the potential relationship between **[AP exam](https://apstudents.collegeboard.org/what-is-ap) performance** and the **presence of nearby universities**. It was initially hypothesized that local (especially R1/R2 or public) universities would contribute to better pass rates for AP exams in their vicinities as a result of their various outreach, dual-enrollment, tutoring, and similar programs for high schoolers. We produce a predictive model that uses a few features related to university presence, personal income, and population to predict AP exam performance.
            
## Background

AP exams are standardized tests widely available at high schools across the United States. During the 2022-2023 school year, [79\%](https://arc.net/l/quote/ewvgnupe) of all public high school students attended schools offering at least five AP courses. These exams are popular for their potential to earn college credits during high school by achieving high scores. In fact, high scores in most AP exams are eligible to receive college credits at roughly [2000](https://apcentral.collegeboard.org/media/pdf/program-summary-report-2024.pdf) higher-education institutions. 

AP exams are scored on a whole number scale between 1 (lowest) and 5 (highest). A student is said to *pass* their AP exam if they score a 3 or higher on the exam. The *pass rate* of a locality would be the proportion of AP exams passed out of all exams taken by its students during a single year. AP outcomes are often correlated to measures of socioeconomic factors: a [recent study](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4574500) confirmed that negative socioeconomic factors have a strong negative influence on exam scores; as well as being a non-native English language speaker. 

Beyond these socioeconomic factors, we wished to measure the strength of the effect of universities on AP outcomes. Without a clear source of data on all high school outreach programs offered by US universities, we made use of the various classifications offered by the [Carnegie Classifications of Institutions of Higher Education](https://carnegieclassifications.acenet.edu/). Of particular interest included R1 and R2 (i.e., doctoral with very high or high research activity, respectively), public, or private institutions. Other minority-serving aspects were also considered, such as historically Black, Hispanic-serving, and tribal colleges.

## Data references and required packages

### Data references in no particular style

1. Indiana University Center for Postsecondary Research (n.d.). The Carnegie Classification of Institutions of Higher Education, 2021 edition, Bloomington, IN.
2. United States Census Bureau -- the United States Office of Management and Budget. Metropolitan and Micropolitan. Core based statistical areas (CBSAs), metropolitan divisions, and combined statistical areas (CSAs). July, 2023.
3. GA Governor's Office of Student Achievement, Data Dashboards, Advanced Placement (AP) Scores, Years 2019, 2020, 2021, 2022, 2023
4. GA Department of Community Affairs Office of Research, Municipalities by County
5. MA Department of Elementary and Secondary Education, Statewide Reports: 2022-23 Advanced Placement Performance
6. Wisconsin Department of Public Instruction, Advanced Placement (AP) Scores, 2019-2023: https://dpi.wi.gov/wisedash/download-files
7. Federal Reserve Bank of St. Louis

### Python packages featured in this project

- `streamlit`
- `sklearn`
- `pandas`
- `statsmodels`
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


