import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import streamlit as st
import matplotlib
from streamlit_folium import st_folium
from pathlib import Path
import geopandas as gpd
import folium
from shapely import wkt
import pickle
import joblib
N_CORES = joblib.cpu_count(only_physical_cores = True)

here_prefix = str(Path(__file__).parent) + '/'
data_prefix = str(Path(__file__).parent) + '/../data/'
html_prefix = data_prefix + 'html/'
parent_prefix = str(Path(__file__).parent) + '/../'

############################# ▲▲▲▲▲▲ IMPORTS ▲▲▲▲▲▲ #############################
############################# ▼▼▼▼▼▼ GLOBALS ▼▼▼▼▼▼ #############################

states_of_interest = ['GA', 'WI', 'MA', 'NC']
MA_neighbors = ['MA', 'NY', 'CT', 'NH', 'RI', 'ME', 'VT', 'NH']
WI_neighbors = ['WI', 'MI', 'MN', 'IA', 'IL']
GA_neighbors = ['GA', 'NC', 'SC', 'FL', 'AL', 'TN']

features_dict = {
    'AP Pass Rate (3 or higher)': 'PassRate',
    'Per capita Personal Income': 'Income',
    'Population': 'Population'
}
explore_features_dict = {
    'Population': { 'unit': 'people', 'label': 'population', 'step': 1000, 'min': 0, 'max': 10000000},
    'Per capita Personal Income': { 'unit': 'USD ($)', 'label': 'per_capita_income', 'step': 1000, 'min': 0, 'max': 5000000},
    'Distance to Closest Five R1/R2': { 'unit': 'miles', 'label': 'closest_five_r1r2_avg', 'step': 1, 'min': 0, 'max': 200},
    'Distance to Closest Five Public': { 'unit': 'miles', 'label': 'closest_five_public_avg', 'step': 1, 'min': 0, 'max': 200},
    'Distance to Closest Five Private': { 'unit': 'miles', 'label': 'closest_five_private_nfp_avg', 'step': 1, 'min': 0, 'max': 200},
    'Distance to Closest Five Land Grant': { 'unit': 'miles', 'label': 'closest_five_landgrnt_avg', 'step': 1, 'min': 0, 'max': 200},
    'Distance to Closest Five STEM': { 'unit': 'miles', 'label': 'closest_five_stem_avg', 'step': 1, 'min': 0, 'max': 200},
    'Enrollment of Closest Five R1/R2': { 'unit': 'students', 'label': 'closest_five_avg_enrollment_r1r2', 'step': 100, 'min': 0, 'max': 100000},
    'Enrollment of Closest Five Public': { 'unit': 'students', 'label': 'closest_five_avg_enrollment_public', 'step':100 , 'min': 0, 'max': 100000},
    'Enrollment of Closest Five Private': { 'unit': 'students', 'label': 'closest_five_avg_enrollment_private_nfp', 'step': 100, 'min': 0, 'max': 100000},
    'Enrollment of Closest Five Land Grant': { 'unit': 'students', 'label': 'closest_five_avg_enrollment_landgrnt', 'step': 100, 'min': 0, 'max': 100000},
    'Enrollment of Closest Five STEM': { 'unit': 'students', 'label': 'closest_five_avg_enrollment_stem', 'step': 100, 'min': 0, 'max': 100000},
    'No. Dorm Rooms of Closest Five R1/R2': { 'unit': 'rooms', 'label': 'closest_five_avg_dormrooms_r1r2', 'step': 100, 'min': 0, 'max': 50000},
    'No. Dorm Rooms of Closest Five Public': { 'unit': 'rooms', 'label': 'closest_five_avg_dormrooms_public', 'step': 100, 'min': 0, 'max': 50000},
    'No. Dorm Rooms of Closest Five Private': { 'unit': 'rooms', 'label': 'closest_five_avg_dormrooms_private_nfp', 'step': 100, 'min': 0, 'max': 50000},
    'No. Dorm Rooms of Closest Five Land Grant': { 'unit': 'rooms', 'label': 'closest_five_avg_dormrooms_landgrant', 'step': 100, 'min': 0, 'max': 50000},
    'No. Dorm Rooms of Closest Five STEM': { 'unit': 'rooms', 'label': 'closest_five_avg_dormrooms_stem', 'step': 100, 'min': 0, 'max': 50000}
}
national_features_dict = {
    'AP Pass Rate (3 or higher)': 'PassRate',
    'AP Score Mean (out of 5)': 'Mean',
    'Total No. AP Exams': 'Total',
    'Offer 5+ Exams (%)': '5+Exams%',
    'Asian Participation (%)': '%Asian',
    'Hispanic or Latino Participation (%)': '%HispanicOrLatino',
    'White Participation (%)': '%White',
    'Black or African American Participation (%)': '%BlackOrAfricanAmerican',
    'Native American or Alaska Native Participation (%)': '%NativeAmericanOrAlaskaNative',
    'Native Hawaiian or other Pacific Islander Participation (%)': '%NativeHawaiianOrOtherPacificIslander',
    'Two or More Races Participation (%)': '%TwoOrMoreRaces',
}

############################# ▲▲▲▲▲▲ GLOBALS ▲▲▲▲▲▲ #############################
############################# ▼▼▼▼▼▼ CACHING ▼▼▼▼▼▼ #############################

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    layout = 'wide',
    page_title = 'University Influence on AP Exam Outcomes',
    page_icon = ':material/school:',
)

# Carnegie broader categories data  
@st.cache_data
def load_broader_categories():
    return pd.read_csv(data_prefix + 'broader_categories_counts.csv')

# Summary stats for three states
@st.cache_data
def get_state_summaries():
    MA_stats = pd.read_csv(here_prefix + 'MA_summary_stats.csv')
    WI_stats = pd.read_csv(here_prefix + 'WI_summary_stats.csv')
    GA_stats = pd.read_csv(here_prefix + 'GA_summary_stats.csv')
    return MA_stats, WI_stats, GA_stats

# AP performance/availability/participation data for three states
@st.cache_data
def get_state_AP_tables():
    MA_AP_table = pd.read_csv(here_prefix + 'MA_AP_table.csv')
    WI_AP_table = pd.read_csv(here_prefix + 'WI_AP_table.csv')
    GA_AP_table = pd.read_csv(here_prefix + 'GA_AP_table.csv')
    return MA_AP_table, WI_AP_table, GA_AP_table

# Wisconsin's predictive model
@st.cache_data
def get_WI_data_and_model():
    WI_full_df = pd.read_csv(data_prefix + 'Wisconsin/train_test_split/Wisconsin_closest_five_method.csv')
    WI_full_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
    WI_model = joblib.load(data_prefix + 'WI_pickled/WI_random_forest_model.pkl')
    return WI_full_df, WI_model

# Three states' counties data
def load_county_choropleth_data():
    counties_map_data = pd.read_csv(here_prefix + 'States_Counties_Map_Data.csv')
    counties_map_data['Year'] = counties_map_data['Year'].astype(str)
    return counties_map_data[counties_map_data['Year'] == '2022']

############################# ▲▲▲▲▲▲ CACHING ▲▲▲▲▲▲ #############################
############################# ▼▼▼▼▼▼ METHODS ▼▼▼▼▼▼ #############################

# Produce geo_dataframe from shape data
def reconstruct_geo(pre_geo_data):
    pre_geo_data['geometry'] = pre_geo_data['geometry'].apply(wkt.loads)
    geo_data = gpd.GeoDataFrame(pre_geo_data, geometry = 'geometry')
    geo_data.set_crs(epsg = 4326, inplace = True)
    return geo_data

def predict_changed_row_entry(model, row, feature, changed_value):
    changed_row = row.copy(deep = True)
    changed_row[feature] = max(changed_value, 0)
    return round(float(model.predict(changed_row)[0]), 1)

def WI_predict_perturb(df, model, county, feature_name, feature_change, feature_dict):
    try:
        feature = feature_dict[feature_name]['label']
        unit = feature_dict[feature_name]['unit']
        step = feature_dict[feature_name]['step']
        
        row = pd.DataFrame(df[(df['COUNTY'] == county) & (df['Year'] == 2022)][df.columns[8:]].iloc[0]).T
        original_feature_value = row[feature].iloc[0]
        changed_feature_value = max(original_feature_value + feature_change, 0)
        prediction_at_original = float(model.predict(row)[0])
        prediction_at_changed = predict_changed_row_entry(model, row, feature, changed_feature_value)
        prediction_change = round(prediction_at_changed - prediction_at_original, 1)
        change_direction = 'increase' if prediction_change > 0 else 'decrease' if prediction_change < 0 else 'not change'

        X = [changed_feature_value + step * n  for n in range(-10, 11, 2)]
        y = [predict_changed_row_entry(model, row, feature, x) for x in X]
        fixed_point = [changed_feature_value, prediction_at_changed]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = X, 
            y = y, 
            mode = 'lines+markers', 
            line = dict(
                color = 'rgb(255, 102, 0)',
                width = 4
            ),
            showlegend = False,
            name = "Pred. Rate"
        ))

        fig.add_trace(go.Scatter(
            x = [fixed_point[0]], 
            y = [fixed_point[1]], 
            mode = 'markers', 
            marker = dict(
                color = 'rgb(255, 102, 0)',
                size = 20
            ),
            showlegend = False,
            hoverinfo = 'skip',
        ))

        fig.update_layout(
            title = f'Nearby Values of {feature_name}',
            xaxis_title = f'{feature_name} ({unit})',
            yaxis_title = 'Predicted AP Pass Rate (%)',
            height = 400,
            template = 'plotly_white',
            showlegend = False
        )

        fig.update_layout(hovermode='x')

        return change_direction, abs(prediction_change), fig
    except Exception as e:
        return None, None, None
    
def display_pickled_plot(filepath, prefix = ''): 
    try:
        with open(prefix + filepath, 'rb') as f:
            st.plotly_chart(pickle.load(f))
    except Exception as e:
        print(f"Failed to load pickled asset with filepath {prefix + filepath}")
        print(f"Error encountered: {e}")

def display_html_plot(filepath, height = 500):
    try:
        with open(html_prefix + filepath, 'r') as f:
            st.components.v1.html(f.read(), height = height)
    except Exception as e:
        print(f"Failed to load html asset with filepath {html_prefix + filepath}")
        print(f"Error encountered: {e}")


############################# ▲▲▲▲▲▲   METHODS  ▲▲▲▲▲▲ #############################
############################# ▼▼▼▼▼▼ APP LAYOUT ▼▼▼▼▼▼ #############################

def main():

    ############################# ▼▼▼▼▼▼ CACHED ▼▼▼▼▼▼ #############################

    # Load in cached data and models
    pre_county_geo_data = load_county_choropleth_data()
    county_geo_data = reconstruct_geo(pre_county_geo_data)
    MA_stats, WI_stats, GA_stats = get_state_summaries()
    MA_AP_table, WI_AP_table, GA_AP_table = get_state_AP_tables()
    WI_full_df, WI_model = get_WI_data_and_model()
    
    ############################# ▲▲▲▲▲▲ CACHED ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼ STYLES ▼▼▼▼▼▼ #############################
    
    # Change some CSS styling in the page for various components, mostly for centering in the page
    style = """
    <style>
    /* Center iFrames */
    .stElementContainer iframe {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    /* Center Plotly plots */
    div.user-select-none.svg-container {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    /* Center images */
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    /* Exclude the GitHub badge */
    img[src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github"] {
        display: inline;
        margin-left: 0;
        margin-right: 0;
    }
    /* Center the fullscreen container */
    div[data-testid="stFullScreenFrame"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    /* Ensure tables retain their internal layout */
    div[data-testid="stDataFrame"] {
        width: auto;
        margin: 0;
    }
    </style>
    """
    # Apply the CSS style
    st.markdown(style, unsafe_allow_html = True)

    ############################# ▲▲▲▲▲▲ STYLES ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼ HEADER ▼▼▼▼▼▼ #############################

    st.markdown('# The Relationship between AP Test Outcomes and Nearby Universities')
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/mcelhens/AP-Outcomes-to-University-Metrics)")

    ############################# ▲▲▲▲▲▲ HEADER ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼  TABS  ▼▼▼▼▼▼ #############################

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Home", "Explore the Model", "Modeling Methods", "The Model", "Massachusetts", "Wisconsin", "Georgia",  "References"])

    ############################# ▼▼▼▼▼▼ HOME TAB ▼▼▼▼▼▼ #############################

    with tab1:

        ############################# ▼▼▼▼▼▼ INTRODUCTION ▼▼▼▼▼▼ #############################

        st.markdown("## Home")

        st.markdown('''
        ### Project Description
                    
        This project was designed to investigate the potential relationship between **[AP exam](https://apstudents.collegeboard.org/what-is-ap) performance** and the **presence of nearby universities**. It was initially hypothesized that local (especially R1/R2 or public) universities would contribute to better pass rates for AP exams in their vicinities as a result of their various outreach, dual-enrollment, tutoring, and similar programs for high schoolers. We produce a predictive model that uses a few features related to university presence, personal income, and population to predict AP exam performance.
                    
        You may interact with our main predictive model in the tab labeled `Explore the Model`. 
        
        You may further see our modeling methodology in the tab labeled `Modeling Method`, as well individual analyses for National (state-level) data, three states: Massachusetts, Wisconsin, and Georgia, as well as a combined analysis for those three states in their respective tabs. 
                    
        ### Background

        AP exams are standardized tests widely available at high schools across the United States. During the 2022-2023 school year, [79\%](https://arc.net/l/quote/ewvgnupe) of all public high school students attended schools offering at least five AP courses. These exams are popular for their potential to earn college credits during high school by achieving high scores. In fact, high scores in most AP exams are eligible to receive college credits at roughly [2000](https://apcentral.collegeboard.org/media/pdf/program-summary-report-2024.pdf) higher-education institutions. 

        AP exams are scored on a whole number scale between 1 (lowest) and 5 (highest). A student is said to *pass* their AP exam if they score a 3 or higher on the exam. The *pass rate* of a locality would be the proportion of AP exams passed out of all exams taken by its students during a single year. AP outcomes are often correlated to measures of socioeconomic factors: a [recent study](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4574500) confirmed that negative socioeconomic factors have a strong negative influence on exam scores; as well as being a non-native English language speaker. 

        Beyond these socioeconomic factors, we would like to measure the strength of the effect of universities on AP outcomes. Without a clear source of data on all high school outreach programs offered by US universities, we make use of the various classifications offered by the [Carnegie Classifications of Institutions of Higher Education](https://carnegieclassifications.acenet.edu/). Of particular interest include R1 and R2 (i.e., doctoral with very high or high research activity, respectively), public, private, or land-grant institutions. Other minority-serving aspects are also considered, such as historically Black, Hispanic-serving, and tribal colleges.

        **Authors (alphabetical)**: *Prabhat Devkota, Shrabana Hazra, Jung-Tsung Li, Shannon J. McElhenney, Raymond Tana*
                    ''')

        ############################# ▲▲▲▲▲▲    INTRODUCTION     ▲▲▲▲▲▲ #############################
        ############################# ▼▼▼▼▼▼ NATIONAL CHOROPLETH ▼▼▼▼▼▼ #############################

        st.markdown("### National AP Performance, Availability, and Participation Data")
        st.markdown("Below you may explore the state-specific data provided publically by the [CollegeBoard on AP performance, availability, and participation](https://apcentral.collegeboard.org/about-ap/ap-data-research/national-state-data) across the United States from the academic year 2022-2023. Simply select a feature you would like to display and watch the map update. Hover over states to get a fuller summary. Universities are drawn as dots on the map.")
        national_selected_feature = st.selectbox("Select Feature to Display", national_features_dict.keys(), key = 'select a feature national choropleth')
        display_html_plot(f'National {national_selected_feature} Choropleth.html')

        ############################# ▲▲▲▲▲▲ NATIONAL CHOROPLETH ▲▲▲▲▲▲ #############################

    ############################# ▲▲▲▲▲▲       HOME TAB       ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼ DATA EXPLORATION TAB ▼▼▼▼▼▼ #############################

    with tab2:
        st.markdown("## Explore the Model")

        ############################# ▼▼▼▼▼▼   PERTURBATIONS   ▼▼▼▼▼▼ #############################

        # Interactive sentence with dropdowns and inputs
        st.markdown("### Explore Model Predictions: Wisconsin")

        st.markdown("""
            What lies below is a tool for exploring how changes in certain features (such as per-capita income, population, or distance to a certain type of university) would be expected to alter AP performance in a county, all according to our predicted model. We focus on the state of Wisconsin for this demonstration. Wisconsin officials may find benefit in this tool when strategically planning appropriate measures for improving educational outcomes; whereas Wisconsin residents (especially parents) may wish to use this tool when comparing the educational outcomes expected in one locality versus another. 
        """)
            
        with st.container(border = True):
            st.markdown("## Wisconsin AP Pass Rate Prediction Model Exploration Tool")
            left_co, right_co = st.columns(2)
            with left_co:
                st.markdown("""
                    #### On the left
                            
                    1. **Select a feature** to manually change using the dropdown.
                    2. **Input a desired numerical change** to that feature using the number input. 
                    3. **Select the county** in Wisconsin whose feature you'll change using the dropdown.
                """)
            with right_co:
                st.markdown("""
                    #### On the right
                            
                    4. **Observe the predicted change** to AP pass rates. 
                    5. **View nearby changes** to the same feature and their associated changes to AP pass rate in the plot below.
                """)

            left_co, right_co = st.columns(2)
            with left_co.container(border = True):
                c1, c2 = st.columns([0.2, 0.8])
                with c1:
                    st.write("If")
                with c2:
                    # Feature selection dropdown
                    model_features = list(explore_features_dict.keys())
                    selected_model_feature = st.selectbox(model_features[0], model_features, label_visibility = 'collapsed', key = 'Select Feature to Perturb')

                c3, c4, c5 = st.columns([0.2, 0.5, 0.3])
                with c3:
                    st.write("changed by")
                with c4: 
                    # Number input for value change
                    value_change = st.number_input('Change',
                                            value = 0,
                                            min_value = - explore_features_dict[selected_model_feature]['max'],
                                            step = explore_features_dict[selected_model_feature]['step'], 
                                            max_value = explore_features_dict[selected_model_feature]['max'],
                                            label_visibility = 'collapsed')
                with c5:
                    # Units (adjust based on feature)
                    st.write(explore_features_dict[selected_model_feature]['unit'])
                
                c6, c7, c8 = st.columns([0.2, 0.5, 0.3])
                with c6:
                    st.write("in")
                with c7:
                    # County selection dropdown
                    county_options = np.sort(county_geo_data[(county_geo_data['PassRate'].notna()) & (county_geo_data['State_Abbreviation'] == 'WI')]['County'].unique())
                    selected_county = st.selectbox(county_options[0], county_options, label_visibility = 'collapsed', key = 'select WI county to perturb')
                with c8:
                    st.write("county, Wisconsin,")
                display_html_plot(f'Wisconsin AP Pass Rate (3 or higher) Choropleth short.html', height = 300)

            with right_co.container(border = True):
                c8, c9 = st.columns([0.2, 0.8])
                with c8:
                    st.write("then")
                with c9:

                    # Get the prediction
                    change_direction, prediction_change, fig = WI_predict_perturb(
                        df = WI_full_df, 
                        model = WI_model, 
                        county = selected_county, 
                        feature_name = selected_model_feature,
                        feature_change = value_change,
                        feature_dict = explore_features_dict
                    )

                    if change_direction == None or prediction_change == None:
                        st.write(f'the model is not sure how the AP pass rate would change (it threw an error).')
                    elif value_change == 0 or prediction_change == 0:
                        st.write("we would expect **no significant change** in the county's AP pass rate.")
                    else:
                        st.write(f"we would expect this county's AP pass rate to **{change_direction}** by **{prediction_change} percentage points**.")
                    st.plotly_chart(fig)
    #     ############################ ▲▲▲▲▲▲   PERTURBATIONS    ▲▲▲▲▲▲ #############################
    #     ############################ ▼▼▼▼▼▼ COUNTY CHOROPLETH ▼▼▼▼▼▼ #############################
    #
    #     st.markdown("### County-level Choropleth Map")
    #
    #     st.markdown("Below you may explore AP performance data within our three main states of interest: Massachusetts, Wisconsin, and Georgia at the level of their counties. You may select a feature to display in the dropdown above the map, and hover over localities for a fuller view. Universities are drawn as dots on the map.")
    # 
    #     county_selected_feature = st.selectbox("Select Feature to Display", features_dict.keys(), key = 'select a feature main choropleth')
    #     display_html_plot(f'County {county_selected_feature} Choropleth.html')
    #
    #     ############################ ▲▲▲▲▲▲ COUNTY CHOROPLETH ▲▲▲▲▲▲ #############################
    # 
    # ############################ ▲▲▲▲▲▲ DATA EXPLORATION TAB ▲▲▲▲▲▲ #############################
    # ############################ ▼▼▼▼▼▼   MODEL METHODS TAB    ▼▼▼▼▼▼ #############################
    # 
    with tab3:

        st.markdown("## Modeling Problem")

        st.markdown("""
            Our target variable is the AP pass rate of a locality (be that a state, county, or school district), which is defined as the percentage of exams scoring 3 or higher. 
                    
            The success of our model is largely driven by the features we select. The features we use are determined by a few criteria: **relevant**, **easy to obtain**, **easy to interpret**, and **easy to compute**.

            We tested models using a number of features, some of which were directly sourced from the datasets we use, and some of which were constructed from other features. In this tab, we present our general procedure for engineering and selecting features for the final model, as well as present the features we use in our final model.
                    
            ### Engineering Distance-Based Features
            
            Certain features used for prediction of AP scores depend on contributions from nearby universities. We attempted to train on many such features, including some related to proximity to:
            - Public Universities
            - Private Not-for-Profit Universities
            - Land-Grant Universities
            - Womens' Colleges/Universities
            - Historically-Black Colleges/Universities
            - Hispanic-Serving Universities
            - Tribal Universities
            - Minority-Serving Universities
                
            Various numerical variables might depend on such categories of universities, such as
            - Distance to the University
            - Total Enrollment of the University
            - Number of Dorm Beds at the University
                    
            We attempted two main schemes for engineering such features:
            1. Take weighted averages using a weighting function that smoothly decreases with distance. 
            2. Combine the contributions from the closet few universities.
                    
            Below we outline these two methods, and it turned out that features based on the second method usually out-performed those designed in the first method.
                    
            #### 1. Weighted Average
                    
            Every university, county, and school district in our datasets is assigned a set of coordinates. Whenever we wish to average a variable $X$ that depends on the university, we take the following approach (which will also apply to the situation of measuring distance to universities of a certain type by considering $X \equiv 1$). We take a weighted average of the variable $X$ across all universities according to some function that shrinks with distance. If university $i$ is at distance $d_i$ from a given school district and has value $X = X_i$, then we estimate the feature value of variable $X$ about this school district to be:
                    
            $$
                \widetilde{X}[\\varepsilon] = \sum_i w(d_i) \cdot X_i \quad \\text{where} \quad w(d) = \\frac{1}{1 + \\frac{d}{\\varepsilon}}.
            $$
                    
            In this model, $\\varepsilon > 0$ serves as a smoothing factor which we set to 10 miles. This choice comes with the interpretation of 10 miles being a good scale for what kinds of distances over which are reasonable to expect universities to have a consistant impacts on the education of nearby high schoolers. Universities within 10 miles of a school district will contribute much more to the sum than schools beyond that. 
                    
            #### 2. Combining Contributions from Closest Few
                    
            In order to measure variable $X$ about a fixed location, add the value of $X$ at the closest, say, $N$ universities. If university $i$ is at distance $d_i$ from a given school district and has variable value $X_i$, then we estimate the feature value of variable $X$ about the school district to be:
                    
            $$
                \widetilde{X}[N] = \\frac{1}{N} \sum_{n = 1}^N \{X_{i_n} : \\text{$(d_{i_n})$ are the smallest $N$ distances}\}.
            $$    
                    
            We compared some values of $N$ such as nearest 2 or nearest 5 universities.

            ### Feature Selection

            Feature selection (i.e., feature elimination) is important for a few reasons:
            1. Some regression models depend on features being functionally independent.
            2. Feature selection can reduce overfitting.
            3. Feature selection can reduce the number of features used in the model, which can make the model more interpretable.

            Other models are more robust to having high correlations between features. Regardless, we employed the popular feature selection method of Principal Component Analysis (PCA). We used the `scikit-learn` library to perform PCA. 
                    
            Various feature importance metrics are available for feature selection as well. We made use of [Shapley (SHAP) values](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) for this. SHAP values are a measure of how much a feature contributes to the model's output, and they come from the study of game theory where the output of a cooperative game depends on whether a player has joined the game or not. Higher SHAP values indicate that the feature -- when included in the model -- is more influential in the model's predictions compared to when the feature is excluded. Various tabs include the SHAP plots for each resolution, and can be read from top to bottom: the highest features being the most important. 

            ### Final Model Features

            Final model features were determined first by hypothesizing what features might be most influential in the model through data exploration on a subset of our combined dataset across the three states of interest (Massachusetts, Wisconsin, and Georgia), and then by performing a combination of PCA and comparison of SHAP values. We describe the full set of selected features below, listed in order of general importance:
            - `Per-Capita Income`: the average per capita income in a given locality.
            - `Population`: the population of the entire locality (state, county, or school district).
            - `Distance to Closets Five Universities`: really, this feature was included many times but for a number of types of universities (as determined by the Carnegie Classification): R1/R2, Public, Private Non-Profit, Land Grant, and STEM-focused.
            - `Average Enrollment in Closest Five Universities`: again, split by university type.
            - `Average Number of Dorm Rooms in Closest Five Universities`: again, split by university type.
                    
            There are altogether twenty features that we use. Two are population and per-capita income. Fifteen of the others are related to nearby universities. For each of the five university categories, we consider the five closest universities to the given locality and compute the average distance to, the annual enrollment (in academic year 2019-2020) at, and the number of dorm rooms at those universities. The last three are one-hot encoders to distinguish the state in which a locality lies.

            ### Model Selection

            Using $5$-fold cross validation on our training dataset with our selected features, we compared various models' performance on individual states' and the combined dataset (MA + WI + GA). Those models included:
            - Naive Average
            - Ordinary Least Squares Regression
            - Ridge Regression
            - Principal Component Analysis ($0.95$)
            - Random Forest
            - Adaboost
            - XGBoost

            XGBoost and Random Forest were the top two performers across the states, but XGBoost was the best model in most contexts. 
        """)
        
        st.markdown("### Model Training")

        st.markdown("""
        One limitation of producing a model to predict AP performance involves the poor resolution of data across the country: some states have very few counties, or very few reporting school districts, to usefully train model on a single year's worth of data. Instead, we wished to train on a combination of the past few years' worth of data. For this to have any hope of producing a powerful model, one would hope that university influences remain mostly consistent across those few years. One way to see this is the case is by observing the stasis evidenced through Carnegie Basic classifications.
        
        Carnegie [Basic Classifications](https://carnegieclassifications.acenet.edu/carnegie-classification/classification-methodology/basic-classification/) are rather detailed: some [33](https://carnegieclassifications.acenet.edu/wp-content/uploads/2023/03/CCIHE2021-FlowCharts.pdf) different classifications were used in the 2021 Basic Classifications scheme. Moreover, the definitions of these classifications have not been consistent across the years. In order to get a picture of how the number of universities in certain classifications has changed over the past few decades, we manually define some broader classifications that can be compared across the various classification schemes employed by Carnegie, and present their frequencies over time below. 
        """)

        broader_categories_counts_df = load_broader_categories()
        fig = px.line(broader_categories_counts_df, x = 'year', y = broader_categories_counts_df.columns[1:], markers = True)
        fig.update_layout(
            hovermode = 'closest',
            title = 'Broader Carnegie Categories over the Years',
            xaxis_title = "Year",
            yaxis_title = "Counts",
            legend_title = "Broad Carnegie Category"
        )
        st.plotly_chart(fig)

        st.markdown(""" 
            Most broad classifications have remained steady, with the exception of many universities that were previously not classified now being considered *special focus institutions*, i.e., institutions confering degrees in one main field. 
                    
            This provides some justification to assume not too much drift occurs in the country's university make-up over the past few years, permitting us to train a model on AP performance data in each state by combining the past five years' worth of AP data (that is, 2018-2019 until 2022-2023).
        """)            

        st.markdown("""
            ### Minority-Serving Features
            
            Despite our hypothesis that minority student performance on AP exams would be affected by the presence of minority-serving institutions, it appears as though this proposed effect is not important. 
                    
            We illustrate this on the national level. On the left, we see how the quantity of minority-serving institutions in a state does not correlate well with AP performance across racial groups. On the right, we see how minority students' performances do not correlate strongly with the presence of various classes of universities which are minority-serving. 
        """)

        left_co, right_co = st.columns(2)
        image_path = data_prefix + 'state_by_state_pickled/'
        left_co, right_co = st.columns(2)
        with left_co:
            st.image(image_path + 'national_outcome_vs_msi.png', caption = '')
        with right_co: 
            st.image(image_path + 'national_msi_correlations.png', caption = '')
            
                    
    ############################# ▲▲▲▲▲▲  MODEL METHODS TAB   ▲▲▲▲▲▲ #############################


    ############################# ▼▼▼▼▼▼ THE MODEL TAB ▼▼▼▼▼▼ #############################
    with tab4: 
        image_path = data_prefix + 'Combined/'
        st.markdown("## The Model")

        st.markdown("### Comparing Models")

        st.markdown("""
            Using the data we collected for AP performance (i.e., pass rates) for counties and/or school districts in four US states: Massachusetts, Wisconsin, Georgia, and North Carolina, we produced a model which we refer to as the *combined model*. Following our method of comparing various model architectures, we found that XGBoost performed best for prediction purposes even after PCA, and further performed hyperparameter tuning on XGBoost to improve its performance. We summarize the results in the following table:
        """)

        st.dataframe(
            data = {
                'Model': ['Baseline', 'OLS Linear Regression', 'XGBoost (w/o hyperparameter tuning)', 'AdaBoost', 'Random Forest', 'XGBoost (w/ hyperparameter tuning)', 'XGBoost (PCA + hyperparameter tuning)'],
                'RMSE': [19.423, 14.816, 10.994, 13.455, 11.343, 10.546, 11.152],
                'R²': [0, 0.436, 0.679, 0.520, 0.657, 0.702, 0.669]
            },
            on_select = 'ignore',
            hide_index = True,
        )

        st.markdown("""
                    As we can see, the hyperparameter-tuned XGBoost model performs the best, but the hyperparameter-tuned XGBoost with PCA was not far behind. Because of the reduction of the features, we expect the model with PCA to be more efficient and potentially also minimize overfitting issues. So, we decided to choose it as our final model to test on the testing data. The performance is as follows:""")
        
        st.dataframe(
            data = {
                'Model': ['XGBoost (PCA + hyperparameter tuning)'],
                'RMSE': [9.227],
                'R²': [0.773]
            },
            on_select = 'ignore',
            hide_index = True,
        )

        st.markdown("That is, over $77\%$ of the variation in AP pass rate is explained via this XGBoost model by the variance in the various features. And the model is typically within 9.3 percentage points of the true AP pass rate.")

        st.markdown("### Modeling Choices")

        st.markdown("""            
            The hyperparameters, feature selection, and evaluation choices made in this modeling process were as follows: 

            - Number of estimators in the XGBoost model: $950$.
            - Maximum depth in the XGBoost model: $4$. 
            - Learning rate of the XGBoost model: $0.03$. 
            - Number of components in the Principal Component Analysis (PCA): $95\%$. 
            - Number of features considered in PCA: $20$.
            - Number of principal components after PCA: $10$. 
            - Number of folds in cross-validation when comparing models: $5$. 
                    
            ### Model Evaluation

            Our final, combined model achieved a coefficient of determination $R^2 = 70.2\%$, meaning the combined model may use the top nine principal components to explain about $70\%$ of the variance in AP passing rates. And the combined model achieved root mean squared error of $\\text{RMSE} = 10.58$ percentage points on AP passing rates. 
                    
            ### Model Features
                    
            The five features which were assigned highest importance according to their SHAP values were as follows:
                    
            1. `Per capita income`
            2. `Average distance to the five closest land grant universities`
            3. `Population`
            4. `Average distance to the five closest public universities`
            5. `Average distance to five closest private not-for-profit universities`
                    
            We summarize these importances in the following SHAP plots.
        """)
        
        left_co, right_co = st.columns(2)
        with left_co:
            st.image(image_path + 'shap_summary_plot_bar.png', caption = 'SHAP values for XGBoost on combined (four states) AP exam performance data over 2018-2022')
        with right_co: 
            st.image(image_path + 'shap_summary_plot_density_scatter.png', caption = 'SHAP densities for XGBoost on combined (four states) AP exam performance data over 2018-2022')

        st.markdown("""
           Despite our suspicions that R1/R2 universities would have a stronger influence on AP exam performance than other university features, it turns out that the *control* (i.e., the governance and funding structure of the institution), as well as the *institutional designation* (i.e., whether it has land-grant status), displayed stronger predictive powers for the counties and school districts' performances on AP tests in these states.          
        """)

        st.markdown("### Results: Wisconsin, 2017-2018")

        st.markdown("""
           We now test how well the combined model performs on historical data from Wisconsin. We ask the model to predict AP pass rates for all counties in Wisconsin, and compare those predictions to reality. We map the results below.               
        """)

        left_co, right_co = st.columns(2)
        with left_co:
            st.markdown("#### Predicted")
            display_html_plot(f'Wisconsin Prediction Choropleth.html')
            st.markdown("""
                The accuracy of the predictions by the combined model on this year's worth of Wisconsin county data are as shown on the right. 
                        
                In context, this means that our combined model can explain over $56\%$ of the variance in AP pass rates in Wisconsin during 2017-2018 from the variance in the various features feeding its principal component features. Moreover, the model is only off in its predictions by about $8.4$ percentage points on average. 
            """)
        with right_co: 
            st.markdown("#### Truth")
            display_html_plot(f'Wisconsin True Choropleth.html')
            st.dataframe(
                data = {
                    'Model': ['XGBoost (PCA + hyperparameter tuning)'],
                    'RMSE': [8.392],
                    'R²': [0.562]
                },
                on_select = 'ignore',
                hide_index = True,
            )

    ############################# ▲▲▲▲▲▲  THE MODEL TAB   ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼ MASSACHUSETTS TAB ▼▼▼▼▼▼ #############################

    with tab5: 
        pickled_path = data_prefix + 'MA_pickled/'
        image_path = data_prefix + 'Massachusetts/plot/'

        st.markdown("## Massachusetts")
        st.markdown('''
            We present some of our exploratory results based on the data available for AP performance in Massachusetts. Analysis on this state was particularly fruitful for the abundance of data on a school district level.
            
            Massachusetts is **densely-populated**, being the 16th most populated state yet 44th in terms of land area. Massachusetts offers iconic institutions such as Harvard University and Massachusetts Institute of Technology, as well as a strong public university system (UMass), placing the state as a **firm leader in higher education** within the US. Massachusetts's main industries are mainly **professional or technical **services, real estate, or healthcare**.
                    
            ### Summary
                    ''')
        
        # Summary and Choropleth
        
        left_co, right_co = st.columns(2)
        with left_co:
            st.markdown('''
                #### AP Performance, Availability, Participation
                        
                Below we summarize the AP performance, availability, and participation in Massachusetts in 2022-2023. 
                        ''')

            ############################# ▼▼▼▼▼▼ MASSACHUSETTS AP TABLE ▼▼▼▼▼▼ #############################
            st.dataframe(data = MA_AP_table, 
                         hide_index = True,
                         key = 'MA_AP_table', 
                         on_select = "ignore",
                         selection_mode = ["multi-row", "multi-column"],
                         use_container_width = True,
                         height=int(35.2*(len(MA_AP_table)+1))
                         )
            ############################# ▲▲▲▲▲▲ MASSACHUSETTS AP TABLE ▲▲▲▲▲▲ #############################
            
            st.markdown('''
                Massachusetts performs slightly above average to most states in AP exams, reflecting its students having strong preparation and support on average. Massachusetts ranked 4th amongst all states in percent of students earning the top score (5) in their AP exams. And a large majority of its school districts offer at least five exams, meaning the state prioritizes college readiness for its high schoolers. However, it is clear that are some distinct disparaties in participation and access to AP exams between racial groups, despite having the second highest Black participation in the country.
                        ''')

            # Scores
            display_pickled_plot('MA_score_distribution.pkl', prefix = pickled_path)

        with right_co:
            ##----------CHOROPLETH MAP OF MASSACHUSETTS
            
            st.markdown('''
                #### Model Features and University Statistics
                        
                Below we present AP performance, per-capita income, population, and university data from the year 2022 in and around Massachusetts (we include nearby states New York, Vermont, New Hampshire, Maine, Rhode Island, and Connecticut).
                        ''')

            ############################# ▼▼▼▼▼▼ MASSACHUSETTS CHOROPLETH ▼▼▼▼▼▼ #############################  
            MA_selected_feature = st.selectbox("Select Feature to Display", features_dict.keys(), key = 'select a feature MA choropleth')
            display_html_plot(f'Massachusetts {MA_selected_feature} Choropleth.html')
            ############################# ▲▲▲▲▲▲ MASSACHUSETTS CHOROPLETH ▲▲▲▲▲▲ #############################  

            st.markdown('''
                        Below we tally the universities which are either high research doctoral (R1/R2), historically Black, Hispanic serving, tribal, women's, public, or private not-for-profit. Most of Massachusetts' universities are concentrated around Boston in the east of the state.                         
                        ''')
            
            ############################# ▼▼▼▼▼▼ MASSACHUSETTS UNIVERSITIES TABLE ▼▼▼▼▼▼ #############################
            st.dataframe(data = MA_stats, 
                         hide_index = True,
                         key = 'MA_summary_data', 
                         on_select = "ignore",
                         selection_mode = ["multi-row", "multi-column"],
                         use_container_width = True
                         )
            ############################# ▲▲▲▲▲▲ MASSACHUSETTS UNIVERSITIES TABLE ▲▲▲▲▲▲ #############################

        # Plot some trends with AP Performance, various features, etc.
        st.markdown("""
            ### Trends with AP Performance
            
            We analyze AP performance in Massachusetts using a detailed, school district-level dataset from the Massachusetts Department of Elementary and Secondary Education. The state serves as an ideal focus for this study due to its dense population and world-class educational institutions, including some of the most renowned universities globally.
                    
            Although modest in size -- ranking 44th in land area -- Massachusetts is the 16th most populous state and is renowned for its academic and intellectual achievements. It is home to Harvard University and the Massachusetts Institute of Technology (MIT), both consistently ranked among the world's top universities. Other esteemed colleges, such as Boston University, Tufts University, and the University of Massachusetts system, further contribute to the state's leadership in higher education and innovation. This strong educational foundation aligns with a thriving economy driven by industries like healthcare, biotechnology, and professional and technical services. By exploring the impact of proximity to universities on AP performance, this study examines how access to world-class higher education resources enhances high school AP outcomes and prepares students for success in a competitive, knowledge-driven economy. 
                    
            ### SHAP Values for Feature Selection
            
            We modeled AP performance in Massachusetts with XGBoost. We further made use of SHAP values for feature selection. First, we use the SHAP summary bar plot to show the average impact of each feature on the model's predictions, as measured by their mean absolute SHAP values. The top five features identified as the most influential and hence selected for further analysis included: 

            1. `Per capita income`
            2. `Population`
            3. `Average distance to five closest public universities`
            4. `Average distance to five closest private not-for-profit universities`
            5. `Average annual enrollment for the five closest land grant universities`
                    
            Next, we present a both a bar plot and density scatter plot of SHAP values to illustrate how each feature influences the model's predictions for individual validation samples. In the density plot on the right, each point represents a sample: its position along the $x$-axis indicates the feature's positive or negative impact on the model's output, and its color reflects the feature value (:red[red] for high values, :blue[blue] for low values). High-density areas highlight overlapping SHAP values, emphasizing the variability in feature impacts across samples.
        """)

        left_co, right_co = st.columns(2)
        with left_co:
            st.image(image_path + 'shap_summary_plot_bar.png', caption = 'SHAP values for XGBoost on Massachusetts AP exam performance data over 2018-2022')
        with right_co:
            st.image(image_path + 'shap_summary_plot_density_scatter.png', caption = 'SHAP densities for XGBoost on Massachusetts AP exam performance data over 2018-2022')        

        st.markdown("""            
            In both plots above, we find that per-capita income is the most significant factor influencing AP outcomes in Massachusetts, outweighing the combined impact of the next four features. Wealthier school districts likely benefit from several advantages, including better funding that enables more AP course offerings, improved materials, and access to highly qualified teachers. Students in these financially advantaged areas also gain additional support through resources such as tutoring, test preparation, and enrichment programs, further enhancing their academic success.
                    
            ### Pass Rate against Important Features
                    
            After identifying the top five features in our model, we analyze the linear relationship between AP outcomes and each of these key features using Ordinary Least Squares (OLS) linear regression, implemented through the statistical framework provided by `statsmodels`. The per-capita income of a school district shows a strong positive correlation with the percentage of AP exams scoring 3 to 5 (the AP pass rate), as indicated by the fitted :green[green] line. For example, districts with a per-capita income of \$200,000 have an AP pass rate of approximately 90\%, compared to a pass rate of only 50\% for districts with a per capita income of \$75,000.
        """)

        display_pickled_plot('MA_pass_vs_school_district_income.pkl', prefix = pickled_path)

        st.markdown("""            
            The district population show a weak negative correlation with the AP pass rate. Larger districts with higher populations generally exhibit lower AP pass rates compared to smaller, less populous districts. However, this trend is not particularly strong, especially among districts with low populations. The reason for this weak negative correlation is not apparent from the plot. We hypothesize that family per capita income may play an implicit yet significant role, as high-income families often reside in suburbs (areas with lower population density) rather than urban centers (areas with higher population density).
        """)

        display_pickled_plot('MA_pass_vs_school_district_population.pkl', prefix = pickled_path)

        st.markdown("""            
            The average distance to the five closest public universities plays an important role in the AP pass rate. This plot below shows that living closer to public universities (e.g., the UMass system) is associated with higher AP pass rates. For instance, reducing the distance from 30 miles to 10 miles increases the AP pass rate from 56\% to 65\%, a significant 9\% improvement.
        """)

        display_pickled_plot('MA_pass_vs_closest_five_public_avg.pkl', prefix = pickled_path)

        st.markdown("""            
            The average distance to the five closest private universities significantly impacts the AP pass rate. This plot illustrates that living closer to private universities (e.g., Harvard and MIT) correlates with higher AP pass rates. For example, residing within 20 miles of a prestigious private university is associated with a fitted AP pass rate exceeding 60\%.
        """)

        display_pickled_plot('MA_pass_vs_closest_five_private_nfp_avg.pkl', prefix = pickled_path)

    ############################# ▲▲▲▲▲▲ MASSACHUSETTS TAB ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼   WISCONSIN TAB   ▼▼▼▼▼▼ #############################

    with tab6: 
        pickled_path = data_prefix + 'WI_pickled/'
        image_path = data_prefix + 'Wisconsin/plot/'

        st.markdown("## Wisconsin")
        st.markdown('''
            We present some of our exploratory results based on the data available for AP performance in Wisconsin. 
                    
            Wisconsin is a **mid-sized** state (23rd largest by land area) with a similarly middling population density (20th highest). Wisconsin is well known for its public **University of Wisconsin System** with its flagship campus in Madison: the system is a significant driver of higher education and research in the state. The state's main industries include **agriculture and manufacturing**.
                    
            ### Summary
                    ''')
        
        # Summary and Choropleth
        
        left_co, right_co = st.columns(2)
        with left_co:
            st.markdown('''
                #### AP Performance, Availability, Participation
                        
                Below we summarize the AP performance, availability, and participation in Wisconsin in 2022-2023. 
                        ''')

            ############################# ▼▼▼▼▼▼ WISCONSIN AP TABLE ▼▼▼▼▼▼ #############################
            st.dataframe(data = WI_AP_table, 
                         hide_index = True,
                         key = 'WI_AP_table', 
                         on_select = "ignore",
                         selection_mode = ["multi-row", "multi-column"],
                         use_container_width = True,
                         height=int(35.2*(len(WI_AP_table)+1))
                         )
            ############################# ▲▲▲▲▲▲ WISCONSIN AP TABLE ▲▲▲▲▲▲ #############################            
            
            st.markdown('''
                        Wisconsin 

                        Wisconsin follows the pattern of northern states outperforming the rest of the country in AP exam pass rates. In fact, Wisconsin is in the top 10 states in terms of pass rate for 2022. But Wisconsin has some of the worst participation by Black students in the country. 
                        ''')
            
            # Scores
            display_pickled_plot('WI_score_distribution.pkl', prefix = pickled_path)

        with right_co:
            ##----------CHOROPLETH MAP OF WISCONSIN
            
            st.markdown('''
                #### Model Features and University Statistics
                        
                Below we present AP performance, per-capita income, population, and university data from the year 2022 in and around Wisconsin (we include nearby states Minnesota, Michigan, Illinois, and Iowa).
                        ''')

            ############################# ▼▼▼▼▼▼ WISCONSIN CHOROPLETH ▼▼▼▼▼▼ #############################
            WI_selected_feature = st.selectbox("Select Feature to Display", features_dict.keys(), key = 'select a feature WI choropleth')
            display_html_plot(f'Wisconsin {WI_selected_feature} Choropleth.html')
            ############################# ▲▲▲▲▲▲ WISCONSIN CHOROPLETH ▲▲▲▲▲▲ #############################

            st.markdown('''
                        Below we tally the universities which are either high research doctoral (R1/R2), historically Black, Hispanic serving, tribal, women's, public, or private not-for-profit. Most of Wisconsin's universities are concentrated in the south and east of the state. But a significant number of universities exist in its neighboring states, including in Chicago and Minneapolis-St. Paul.      
                        ''')

            ############################# ▼▼▼▼▼▼ WISCONSIN UNIVERSITIES TABLE ▼▼▼▼▼▼ #############################
            st.dataframe(data = WI_stats, 
                         hide_index = True,
                         key = 'WI_summary_data', 
                         on_select = "ignore",
                         selection_mode = ["multi-row", "multi-column"],
                         use_container_width = True
                         )
            ############################# ▲▲▲▲▲▲ WISCONSIN UNIVERSITIES TABLE ▲▲▲▲▲▲ #############################

        # Some basic trends with AP Performance
        st.markdown("""
            ### Trends with AP Performance
            
            Wisconsin is a state in the Upper Midwest of the US, lying to the southwest of the Great Lakes. It is the 23rd largest state by area and 20th largest by population with a population of almost 6 million. It is the 28th richest state in terms of per capita income with a per capita income of \$61,992 in 2022 (according to Federal Reserve Bank of St. Louis). Owing to its fairly rural population, Wisconsin does not boast very many grand cities (with Milwaukee and Madison being the two most populous). And its economy is driven by manufacturing, agriculture and tourism. Wisconsin consists of 72 counties and more than 300 school districts. 
                
            Due to having a fairly rural population, the number of notable universities in Wisconsin is comparatively fewer, many of which are clustered around Milwaukee. The major public university system in the state is the University of Wisconsin System, which includes the flagship University of Wisconsin-Madison, and is one of the largest higher education systems in the country. Close to Wisconsin to the west is the Minneapolis-St. Paul metropolitan area, where one may find several colleges and universities such as the University of Minnesota Twin Cities, among others.

            The Wisconsin Department of Public Instruction is the state education management agency in Wisconsin. It keeps a comprehensive tab of [education-related data](https://dpi.wi.gov/wisedash/download-files) in the state, where one may find the pass rates (percentage of students scoring 3 or higher) of AP examinations by school districts and counties over five academic years: 2018/19 to 2022/23. Because of the limited availability of population and income related data for school districts, we opted to train on county-wise data for any machine learning models predicting AP pass rates in Wisconsin.
                    
            ### Model Selection and Performance
            
            We used various regression tools available via `sklearn` or `xgboost`. The performance of the models (as measured by root mean squared error and coefficients of determination) are summarized as follows.
                    
            We summarize the results of testing various models trained on the county-level AP performance data from Wisconsin between 2018-2019 and 2022-2023. 
        """)

        st.dataframe(
            data = {
                'Models': ['Baseline', 'OLS Linear Regression', 'Ridge', 'AdaBoost', 'Random Forest', 'XGBoost'],
                'RMSE': [13.895, 11.933, 11.897, 10.214, 9.864, 10.376],
                'R²': [-0.029, 0.236, 0.242, 0.444, 0.480, 0.425]
            },
            on_select = 'ignore',
            hide_index = True,
        )

        st.markdown("""            
            As seen, the Random Forest model performed the best in terms of both root mean squared error and $R^2$-coefficient. As such, we chose Random Forest as the model of our choice for Wisconsin.
                    
            ### SHAP Values for Feature Selection

            There were altogether 17 features present in our modeling. However, the features are not all equally important. We used Shapley Additive Explanations (SHAP for short) to interpret our Random Forest model and understand the important of various features on our random forest model. First, we use the SHAP summary bar plot to show the average impact of each feature on the model's predictions, as measured by their mean absolute SHAP values. The top five features identified as the most influential and hence selected for further analysis included: 
                    
            1. `Per capita income`
            2. `Population`
            3. `Average distance to five closest private not-for-profit universities`
            4. `Average distance to five closest public universities`
            5. `Average number of dorm rooms amongst the five closest STEM universities`
                    
            In particular, per-capita income and population dominate in importance towards explaining the variance in AP performance amongst the various features. Nonetheless, the contribution of the other 15 features -- when summed -- is quite significant.
                    
            The following SHAP bar and scatter density plots elucidate the feature importances more clearly: in the scatter plot, each dot represents a sample whose color indicates the feature value on the sample (:red[red] for high value and :blue[blue] for low value). The placement of a point along the $x$-axis represents the feature's positive or negative impact on the model's output during that sample. One may see that per-capita income and population have higher quantity of red dots scattered on the positive side of the $x$-axis, more confirmation of these two features' predictive power.
        """)

        left_co, right_co = st.columns(2)
        with left_co:
            st.image(pickled_path + 'shap_random_forest_bar_plot.png', caption = 'SHAP values for Random Forest model on Wisconsin AP exam performance data over 2018-2022')
        with right_co:
            st.image(pickled_path + 'shap_random_forest_scatter_plot.png', caption = 'SHAP densities for Random Forest model on Wisconsin AP exam performance data over 2018-2022')       

        st.markdown("""
            ### Pass Rate against Important Features
                    
            Let us summarize the relationship of pass rate with the five important features identified by the SHAP plots. Intuitively, one should expect the pass rate to be positively correlated with per-capita income: higher income counties have boast over 80\% pass rates whereas low income counties have pass rates even below 30\%, barring a few exceptions. In fact, Forest County, WI offers both the highest AP exam pass rate and the lowest per-capita income amongst all Wisconsin counties. We visualize this relationship below.
        """)

        display_pickled_plot('WI_AP_pass_rate_by_counties_vs_income.pkl', prefix = pickled_path)

        st.markdown("""            
            The relationship between population and pass rate, on the other hand, is not as straightforward. While the OLS regression line fitted to these variables has positive slope, the scatter plot does not follow a visually clear trend. It is to some surprise that population concludes as the second most important feature in the chosen model. 
        """)

        display_pickled_plot('WI_AP_pass_rate_by_counties_vs_population.pkl', prefix = pickled_path)

        st.markdown("""            
            Moreover, one would expect that counties located closer to (or containing) particular universities would have higher pass rates for two reasons: (1) that most universities may be more likely to be found in wealthier counties, while (2) many universities (especially those public) should be invested in coummunity outreach programs improving the educational quality of nearby high schools. With its comparatively stronger public university system, Wisconsin might be expected to exhibit a stronger relationship between pass rates and distance to public universities. We one may observe from the below plot, this intuition matches reality. Notice the the trendline for pass rate against average distance to the closest five universities is almost flat for private universities, but significantly more negatively sloped for public. A simple explanation for why might posit that public universities are more obliged to develop community outreach programs to strengthen the educational quality of their surrouding localities.
        """)

        left_co, right_co = st.columns(2)
        with left_co:
            display_pickled_plot('WI_AP_pass_rate_by_counties_vs_avg_five_private.pkl', prefix = pickled_path)
        with right_co:
            display_pickled_plot('WI_AP_pass_rate_by_counties_vs_avg._dist_public.pkl', prefix = pickled_path)

        st.markdown("""            
            Finally, consider pass rate against the average number of dorm rooms amongst the closest five private universities. Without a strong intuitive reason *a priori*, we may not explain with much certainty the apparent negatively sloped trend. Perhaps a presence of large private universities is a marker for an absence of large public universities sharing the same space. 
        """)

        display_pickled_plot('WI_AP_pass_rate_by_counties_vs_avg_five_private_dormrooms.pkl', prefix = pickled_path)

    ############################# ▲▲▲▲▲▲ WISCONSIN TAB ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼  GEORGIA TAB  ▼▼▼▼▼▼ #############################

    with tab7: 
        pickled_path = data_prefix + 'GA_pickled/'
        image_path = parent_prefix + 'Georgia/'

        st.markdown("## Georgia")
        st.markdown('''
            We present some of our exploratory results based on the data available for AP performance in Georgia. 
                    
            Georgia is a mid-sized state (24th largest by land area) with a relatively **high population** (8th largest by population). Over half (57.2%) of the state's population is **concentrated in the Atlanta metro area**, which also hosts some of the state's most influential universities like Georgia Institute of Technology and the Unviersity of Georgia. Moreover, Georgia is **33.2% Black or African American**, and offers 9 **historically Black colleges**, ranking third across all states in both respects. Moorehouse College is a notable example that has been home to celebrated African American graduates and recent investments in research on issues affecting Black men. Georgia's main industries are **areospace, automotive, and manufacturing**. 
                    
            ### Summary
                    ''')
        
        # Summary and Choropleth
        
        left_co, right_co = st.columns(2)
        with left_co:
            st.markdown('''
                #### AP Performance, Availability, Participation
                        
                Below we summarize the AP performance, availability, and participation in Georgia in 2022-2023. 
                        ''')
            
            ############################# ▼▼▼▼▼▼ GEORGIA AP TABLE ▼▼▼▼▼▼ #############################
            st.dataframe(data = GA_AP_table, 
                         hide_index = True,
                         key = 'GA_AP_table', 
                         on_select = "ignore",
                         selection_mode = ["multi-row", "multi-column"],
                         use_container_width = True,
                         height=int(35.2*(len(GA_AP_table)+1))
                         )
            ############################# ▲▲▲▲▲▲ GEORGIA AP TABLE ▲▲▲▲▲▲ #############################

            st.markdown('''
                Georgia has the lowest passing rate out of all three states considered in our state-by-state analysis, but not by much. Actually, Georgia's passing rates ranked very well in 2022-2023 in comparison to those of its southeastern counterparts, and were almost four percentage points above the national average. The state also experiences some of the worst disparities between Asian, White, and Black student participations.
                        ''')
            
            # Scores
            display_pickled_plot('GA_score_distribution.pkl', prefix = pickled_path)

        with right_co:
            ##----------CHOROPLETH MAP OF GEORGIA

            st.markdown('''
                #### Model Features and University Statistics
                        
                Below we present AP performance, per-capita income, population, and university data from the year 2022 in and around Georgia (we include nearby states North Carolina, South Carolina, Florida, Alabama, and Tennessee).
                        ''')

            ############################# ▼▼▼▼▼▼ GEORGIA CHOROPLETH ▼▼▼▼▼▼ #############################
            GA_selected_feature = st.selectbox("Select Feature to Display", features_dict.keys(), key = 'select a feature GA choropleth')
            display_html_plot(f'Georgia {GA_selected_feature} Choropleth.html')
            ############################# ▲▲▲▲▲▲ GEORGIA CHOROPLETH ▲▲▲▲▲▲ #############################

            st.markdown('''
                        Below we tally the universities which are either high research doctoral (R1/R2), historically Black, Hispanic serving, tribal, women's, public, or private not-for-profit. Many of Georgia's universities are concentrated around Atlanta in the north of the state.                     
                        ''')

            ############################# ▼▼▼▼▼▼ GEORGIA UNIVERSITIES TABLE ▼▼▼▼▼▼ #############################
            st.dataframe(data = GA_stats, 
                         hide_index = True,
                         key = 'GA_summary_data', 
                         on_select = "ignore",
                         selection_mode = ["multi-row", "multi-column"],
                         use_container_width = True,
                         )
            ############################# ▲▲▲▲▲▲ GEORGIA UNIVERSITIES TABLE ▲▲▲▲▲▲ #############################

        # Some basic trends with AP Performance
        st.markdown(""" 
                    
            ### Trends with AP Performance
                    
            First, we summarize the results of our analysis on Georgia: Georgia proves interesting in its modestly reduced dependence on income; however, it generally trends like the other states. Its largest university influence was land-grant university dorms and distance both with slight positive relationships. To our surprise, nearby universities of other types do not have a positive relationship. Land-grant universities had the highest average distance away. One has to ask if perhaps the distance is so far that it may actually be an erroneous conclusion. We selected XGBoost as the optimal model for Georgia. Overall, Georgia provides evidence to the variance of states across the US. Further research would be advantaged by minority considerations at least at the university level, training over more years of data, and the absorbtion of other city districts into their appropriate counties.
              
            We trained various regression tools available via `sklearn` or `xgboost`. The performance of the models (as measured by root mean squared error and coeffiients of determination) are summarized as follows.
        """)

        st.dataframe(
            data = {
                'Models': ['Baseline', 'OLS Linear Regression', 'Ridge', 'AdaBoost', 'Random Forest', 'XGBoost'],
                'RMSE': [20.05,  16.57, 16.56, 12.90, 13.52, 12.05],
                'R²': [-0.037, 0.283, 0.284, 0.562, 0.521, 0.609]
            },
            on_select = 'ignore',
            hide_index = True,
        )

        st.markdown(""" 
            As seen, the XGBoost model performed the best in terms of both root mean squared error and $R^2$-coefficient. As such, we chose XGBoost as the model of our choice for Georgia.
                    
            ### SHAP Values for Feature Selection

            There were altogether 17 features present in our modeling. However, the features are not all equally important. We used Shapley Additive Explanations (SHAP for short) to interpret our Random Forest model and understand the important of various features on our random forest model. First, we use the SHAP summary bar plot to show the average impact of each feature on the model's predictions, as measured by their mean absolute SHAP values. The top five features identified as the most influential and hence selected for further analysis included: 
                    
            1. `Per capita income`
            2. `Population`
            3. `Average number of dorm rooms amongst the five closest land-trant universities`
            4. `Average distance to five closest land-grant universities`
            5. `Average distance to five closest private, not-for-profit universities`
                    
            In particular, per-capita income and population dominate in importance towards explaining the variance in AP performance amongst the various features. Nonetheless, the contribution of the other 15 features -- when summed -- is quite significant.
                    
            The following SHAP bar and scatter density plots elucidate the feature importances more clearly: in the scatter plot, each dot represents a sample whose color indicates the feature value on the sample (:red[red] for high value and :blue[blue] for low value). The placement of a point along the $x$-axis represents the feature's positive or negative impact on the model's output during that sample. One may see that per-capita income and population have higher quantity of red dots scattered on the positive side of the $x$-axis, more confirmation of these two features' predictive power.
        """)

        left_co, right_co = st.columns(2)
        with left_co:
            st.image(image_path + 'shap_summary_xbg.png', caption = 'SHAP values for Random Forest model on Georgia AP exam performance data over 2018-2022')
        with right_co:
            st.image(image_path + 'shap_xgb.png', caption = 'SHAP densities for Random Forest model on Georgia AP exam performance data over 2018-2022')       

        st.markdown("""
            ### Pass Rate against Important Features
                    
            Let us summarize the relationship of pass rate with the five important features identified by the SHAP plots. Intuitively, one should expect the pass rate to be positively correlated with per-capita income and population. This is what we see:
        """)

        left_co, right_co = st.columns(2)
        with left_co:
            display_pickled_plot('GA_AP_pass_rate_by_counties_per_capita_income.pkl', prefix = pickled_path)
        with right_co:
            display_pickled_plot('GA_AP_pass_rate_by_counties_population.pkl', prefix = pickled_path)

        st.markdown("""                    
            Less important of relationships are visualized below. 
        """)

        display_pickled_plot('GA_AP_pass_rate_by_counties_closest_five_avg_dormrooms_landgrant.pkl', prefix = pickled_path)
        
        left_co, right_co = st.columns(2)
        with left_co:
            display_pickled_plot('GA_AP_pass_rate_by_counties_closest_five_landgrant_avg.pkl', prefix = pickled_path)
        with right_co:
            display_pickled_plot('GA_AP_pass_rate_by_counties_closest_five_private_nfp_avg.pkl', prefix = pickled_path)
    
    ############################# ▲▲▲▲▲▲   GEORGIA TAB   ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼  REFERENCES TAB ▼▼▼▼▼▼ #############################

    with tab8:
        st.markdown("## References")
        st.markdown('''
            1. Indiana University Center for Postsecondary Research (n.d.). The Carnegie Classification of Institutions of Higher Education, 2021 Edition. Bloomington, IN: Author. [Link](https://carnegieclassifications.acenet.edu/)
            2. United States Census Bureau, United States Office of Management and Budget (July, 2023). Metropolitan and Micropolitan Core-Based Statistical Areas (CBSAs), Metropolitan Divisions, and Combined Statistical Areas (CSAs). [Link](https://www.census.gov/geographies/reference-maps/2023/geo/cbsa.html)
            3. Georgia Governor's Office of Student Achievement. Data Dashboards: Advanced Placement (AP) Scores, Years 2019–2023. [Link](https://gosa.georgia.gov/dashboards-data-report-card/downloadable-data)
            4. Georgia Department of Community Affairs Office of Research. Municipalities by County. [Link](https://dca.georgia.gov/)
            5. Massachusetts Department of Elementary and Secondary Education. Statewide Reports: 2022–23 Advanced Placement Performance. [Link](https://profiles.doe.mass.edu/statereport/ap.aspx)
            6. Wisconsin Department of Public Instruction. Advanced Placement (AP) Scores, 2019–2023. [Link](https://dpi.wi.gov/wisedash/download-files)
            7. Federal Reserve Bank of St. Louis. [Link](https://fred.stlouisfed.org/)
            8. United States Census Bureau. Cartographic Boundary Fiels - Shapefile. 2018. [Link](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html)
            9. AP National and State Data. AP Central. College Board. 2024. [Link](](https://apcentral.collegeboard.org/about-ap/ap-data-research/national-state-data). 
                    ''')

    ############################# ▲▲▲▲▲▲ REFERENCES TAB ▲▲▲▲▲▲ #############################
    ############################# ▲▲▲▲▲▲  TABS  ▲▲▲▲▲▲ #############################
############################# ▲▲▲▲▲▲ APP LAYOUT ▲▲▲▲▲▲ #############################


# Automatically runs when executed
if __name__ == "__main__":
    main()