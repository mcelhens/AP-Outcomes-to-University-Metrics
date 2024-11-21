import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import streamlit as st
from streamlit_folium import st_folium, folium_static
from pathlib import Path
import geopandas as gpd
import folium
from shapely import wkt
import pickle
data_prefix = '../data/'

print("---------Raymond's printout--------------")
import os
print("Current working directory:", os.getcwd())
print("Could try accessing the US data with file path", Path(__file__).parent / "US_States_Map_Data.csv")
print("Now changing the path")
path = Path(__file__).parent / "streamlit_app"
print("Now `path` set equal to", path)
print("\tand `Path(__file__).parent ==", Path(__file__).parent)
print("Could try accessing the US data with file path", Path(__file__).parent / "US_States_Map_Data.csv")
print("Current working directory:", os.getcwd())
print("-----------------------------------------")

US_States_map_data_path = path + '/' + "US_States_Map_Data.csv"


############################# ▲▲▲▲▲▲ IMPORTS ▲▲▲▲▲▲ #############################
############################# ▼▼▼▼▼▼ GLOBALS ▼▼▼▼▼▼ #############################

# years = ['2019', '2020', '2021', '2022']
states_of_interest = ['GA', 'WI', 'MA']
MA_neighbors = ['MA', 'NY', 'CT', 'NH', 'RI', 'ME', 'VT', 'NH']
WI_neighbors = ['WI', 'MI', 'MN', 'IA', 'IL']
GA_neighbors = ['GA', 'NC', 'SC', 'FL', 'AL', 'TN']

features_dict = {
    'AP Pass Rate (3 or higher)': 'PassRate',
    'Per capita Personal Income': 'Income',
    'Population': 'Population'
}
national_features_dict = {
    'AP Pass Rate (3 or higher)': 'PassRate',
    'AP Score Mean (out of 5)': 'Mean',
    'Total No. AP Exams': 'Total',
    'Offer 5+ Exams (%)': '5+Exams%',
    'Asian Participation (%)': '%Asian',
    'Hispanic/Latino Participation (%)': '%HispanicOrLatino',
    'White Participation (%)': '%White',
    'Black or African American Participation (%)': '%BlackOrAfricanAmerican',
    'Native American or Alaska Native Participation (%)': '%NativeAmericanOrAlaskaNative',
    'Native Hawaiian or other Pacific Islander Participation (%)': '%NativeHawaiianOrOtherPacificIslander',
    'Two or More Races Participation (%)': '%TwoOrMoreRaces',
}

model_features_dict = {
    'Per capita Personal Income': {
        'units': 'dollars',
        'min': -100000,
        'max': 500000,
        'step': 10000,
    },
    'Average Distance to Nearby Universities': {
        'units': 'miles',
        'min': -30,
        'max': 100,
        'step': 1,
    },
}

############################# ▲▲▲▲▲▲ GLOBALS ▲▲▲▲▲▲ #############################
############################# ▼▼▼▼▼▼ CACHING ▼▼▼▼▼▼ #############################

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    layout = 'wide',
    page_title = 'AP Outcomes vs University Metrics',
    page_icon = 'https://preview.redd.it/uxmxgdgoqdz81.jpg?width=640&crop=smart&auto=webp&s=ca71011f6de31eb654b50a972b14806b19e98e52', # This is an emoji shortcode. Could be a URL too.
)

@st.cache_data
def load_universities_data():
    universities_data = pd.read_csv('../data/carnegie_with_location.csv')[['name', 'stabbr', 'latitude', 'longitude']]
    MA_nearby_universities = universities_data[universities_data['stabbr'].isin(MA_neighbors)]
    WI_nearby_universities = universities_data[universities_data['stabbr'].isin(WI_neighbors)]
    GA_nearby_universities = universities_data[universities_data['stabbr'].isin(GA_neighbors)]
    return universities_data, MA_nearby_universities, WI_nearby_universities, GA_nearby_universities

@st.cache_data
def load_national_choropleth_data():
    # return pd.read_csv('US_States_Map_Data.csv')
    return pd.read_csv(US_States_map_data_path)

@st.cache_data
def load_county_choropleth_data():
    counties_map_data = pd.read_csv('States_Counties_Map_Data.csv')
    counties_map_data['Year'] = counties_map_data['Year'].astype(str)
    return counties_map_data[counties_map_data['Year'] == '2022']
    
@st.cache_data
def load_broader_categories():
    return pd.read_csv('../data/broader_categories_counts.csv')

@st.cache_data
def get_state_summaries():
    MA_stats = pd.read_csv('MA_summary_stats.csv')
    WI_stats = pd.read_csv('WI_summary_stats.csv')
    GA_stats = pd.read_csv('GA_summary_stats.csv')
    return MA_stats, WI_stats, GA_stats

@st.cache_data
def get_state_AP_tables():
    MA_AP_table = pd.read_csv('MA_AP_table.csv')
    WI_AP_table = pd.read_csv('WI_AP_table.csv')
    GA_AP_table = pd.read_csv('GA_AP_table.csv')
    return MA_AP_table, WI_AP_table, GA_AP_table

############################# ▲▲▲▲▲▲ CACHING ▲▲▲▲▲▲ #############################
############################# ▼▼▼▼▼▼ METHODS ▼▼▼▼▼▼ #############################

# Model prediction (replace with actual model)
def predict_ap_pass_rate(county, year, feature, new_value):
    # Check that new_value is numerical!
    # Find the row in the data that matches this county and year
    # Then use all of its features besides whatever feature has a new value
    # The truth is filled in with PassRate from this row
    
    truth = 20
    prediction = 10 # Replace with actual model
    prediction_change = truth - prediction
    change_direction = 'increase' if prediction_change >= 0 else 'decrease'
    return change_direction, abs(prediction_change)

def choropleth(geo_data, 
               selected_feature, 
               university_data, 
               features_dict,
               title,
               fields,
               aliases,
               center,
               zoom):
    # Define the choropleth layer based on the selected feature and year
        choropleth_layer = folium.Choropleth(
            geo_data = geo_data,
            name = f'{title} choropleth',
            data = geo_data,
            columns = ['GEOID', features_dict[selected_feature]],
            key_on = 'feature.properties.GEOID',
            fill_color = 'YlOrRd',
            nan_fill_color = 'lightgrey',
            fill_opacity = 0.7,
            line_opacity = 0.2,
            legend_name = f'{selected_feature} {title}'
        )

        # Define tooltips with certain areas
        area_tooltips = folium.GeoJson(
            geo_data,
            name = f'{title} tooltips',
            control = False,
            style_function = lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
            tooltip = folium.features.GeoJsonTooltip(
                fields = fields,
                aliases = aliases,
                localize = True
            )
        )

        if university_data is not None:
            # Add a new layer for university markers
            university_layer = folium.FeatureGroup(name = f'{title} universities')
            # Add markers for each university in the DataFrame
            for _, row in university_data.iterrows():
                folium.Circle(
                    radius = 300,
                    fill = False,
                    color = "black",
                    fill_color = "orange",
                    opacity = 1,
                    fill_opacity = 0.2,
                    weight = 2,
                    location = [row['latitude'], row['longitude']],
                    popup = folium.Popup(f"{row['name']}", max_width = 300),
                    tooltip = row['name']
                ).add_to(university_layer)

        # Map center coordinates
        m = folium.Map(location = center, zoom_start = zoom)
        # Add choropleth layer to the map
        choropleth_layer.add_to(m)
        # Add the area tooltips to the map
        area_tooltips.add_to(m)
        # Add the university layer to the map
        university_layer.add_to(m)
        # Add a layer control to toggle layers
        folium.LayerControl().add_to(m)
        # Render the map in Streamlit using folium_static
        folium_static(m)

def reconstruct_geo(pre_geo_data):
    pre_geo_data['geometry'] = pre_geo_data['geometry'].apply(wkt.loads)
    geo_data = gpd.GeoDataFrame(pre_geo_data, geometry = 'geometry')
    geo_data.set_crs(epsg = 4326, inplace = True)
    return geo_data

def pickled_plot(filepath, prefix = ''): 
    try:
        with open(prefix + filepath, 'rb') as f:
            st.plotly_chart(pickle.load(f))
    except:
        pass

############################# ▲▲▲▲▲▲   METHODS  ▲▲▲▲▲▲ #############################
############################# ▼▼▼▼▼▼ APP LAYOUT ▼▼▼▼▼▼ #############################

def main():

    ############################# ▼▼▼▼▼▼ CACHED ▼▼▼▼▼▼ #############################

    # Load in cached data
    pre_national_geo_data = load_national_choropleth_data()
    pre_county_geo_data = load_county_choropleth_data()
    universities_data, MA_nearby_universities, WI_nearby_universities, GA_nearby_universities = load_universities_data()
    MA_stats, WI_stats, GA_stats = get_state_summaries()
    MA_AP_table, WI_AP_table, GA_AP_table = get_state_AP_tables()

    # Reconstruct geometries from WKT strings (not hashable so can't cache this part)
    national_geo_data = reconstruct_geo(pre_national_geo_data)
    county_geo_data = reconstruct_geo(pre_county_geo_data)
    MA_geo_data = county_geo_data[county_geo_data['State_Abbreviation'] == 'MA']
    WI_geo_data = county_geo_data[county_geo_data['State_Abbreviation'] == 'WI']
    GA_geo_data = county_geo_data[county_geo_data['State_Abbreviation'] == 'GA']
    
    ############################# ▲▲▲▲▲▲ CACHED ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼ STYLES ▼▼▼▼▼▼ #############################
    
    # Change some CSS styling in the page for iframes, helps reliably center all choropleth maps and similar
    style = """
    <style>
    .stElementContainer iframe {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    """
    # Apply the CSS style
    st.markdown(style, unsafe_allow_html = True)

    ############################# ▲▲▲▲▲▲ STYLES ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼ HEADER ▼▼▼▼▼▼ #############################

    st.markdown('# The Relationship between AP Test Outcomes and Nearby Universities')
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/mcelhens/AP-Outcomes-to-University-Metrics)")

    ############################# ▲▲▲▲▲▲ HEADER ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼  TABS  ▼▼▼▼▼▼ #############################

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Home", "Data Exploration", "Our Model", "Massachusetts", "Wisconsin", "Georgia", "References"])

    ############################# ▼▼▼▼▼▼ HOME TAB ▼▼▼▼▼▼ #############################

    with tab1:

        ############################# ▼▼▼▼▼▼ INTRODUCTION ▼▼▼▼▼▼ #############################

        st.markdown("## Home")

        st.markdown('''
        ### Project Description
                    
        This project was designed to investigate the potential relationship between **[AP exam](https://apstudents.collegeboard.org/what-is-ap) performance** and the **presence of nearby universities**. It was initially hypothesized that local (especially R1/R2 or public) universities would contribute to better pass rates for AP exams in their vicinities as a result of their various outreach, dual-enrollment, tutoring, and similar programs for high schoolers. We produce a predictive model that uses a few features related to university presence, personal income, and population to predict AP exam performance.
                    
        ### Background

        AP exams are standardized tests widely available at high schools across the United States. During the 2022-2023 school year, [79\%](https://arc.net/l/quote/ewvgnupe) of all public high school students attended schools offering at least five AP courses. These exams are popular for their potential to earn college credits during high school by achieving high scores. In fact, high scores in most AP exams are eligible to receive college credits at roughly [2000](https://apcentral.collegeboard.org/media/pdf/program-summary-report-2024.pdf) higher-education institutions. 

        AP exams are scored on a whole number scale between 1 (lowest) and 5 (highest). A student is said to *pass* their AP exam if they score a 3 or higher on the exam. The *pass rate* of a locality would be the proportion of AP exams passed out of all exams taken by its students during a single year. AP outcomes are often correlated to measures of socioeconomic factors: a [recent study](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4574500) confirmed that negative socioeconomic factors have a strong negative influence on exam scores; as well as being a non-native English language speaker. 

        Beyond these socioeconomic factors, we would like to measure the strength of the effect of universities on AP outcomes. Without a clear source of data on all high school outreach programs offered by US universities, we make use of the various classifications offered by the [Carnegie Classifications of Institutions of Higher Education](https://carnegieclassifications.acenet.edu/). Of particular interest include R1 and R2 (i.e., doctoral with very high or high research activity, respectively), public, or private institutions. Other minority-serving aspects are also considered, such as historically Black, Hispanic-serving, and tribal colleges.

        **Authors (alphabetical)**: *Prabhat Devkota, Shrabana Hazra, Jung-Tsung Li, Shannon J. McElhenney, Raymond Tana*
                    ''')

        ############################# ▲▲▲▲▲▲    INTRODUCTION     ▲▲▲▲▲▲ #############################
        ############################# ▼▼▼▼▼▼ NATIONAL CHOROPLETH ▼▼▼▼▼▼ #############################

        st.markdown("### National AP Performance, Availability, and Participation Data")

        national_selected_feature = st.selectbox("Select Feature to Display", national_features_dict.keys(), key = 'select a feature national choropleth')
        # Generate the Choropleth map of all US states
        choropleth(
            geo_data = national_geo_data, 
            selected_feature = national_selected_feature, 
            university_data = universities_data, 
            features_dict = national_features_dict,
            title = 'All States AP Performance and Demographics 2022',
            fields = ['State', 'PassRate', 'Mean', 'Total', '5+Exams%', '%Asian', '%HispanicOrLatino', '%White', '%BlackOrAfricanAmerican', '%NativeAmericanOrAlaskaNative', '%NativeHawaiianOrOtherPacificIslander', '%TwoOrMoreRaces'],
            aliases = ['State Name:', 'Pass Rate (%)', 'Mean AP Score', 'Total No. AP Exams', 'Offer 5+ Exams (%)', '% Asian:', '% Hispanic/Latino:', '% White:', '% Black or African American:', '% Native American or Alaska Native:', '% Native Hawaiian or other Pacific Islander:', '% Two or More Races:'],
            center = [40, -96],
            zoom = 4
        )        
        ############################# ▲▲▲▲▲▲ NATIONAL CHOROPLETH ▲▲▲▲▲▲ #############################

    ############################# ▲▲▲▲▲▲       HOME TAB       ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼ DATA EXPLORATION TAB ▼▼▼▼▼▼ #############################

    with tab2:
        st.markdown("## Data Exploration")

        ############################# ▼▼▼▼▼▼ COUNTY CHOROPLETH ▼▼▼▼▼▼ #############################

        ##----------CHOROPLETH MAP
        # Display select boxes
        st.markdown("### County-level Choropleth Map")

        st.markdown("#### Map Options")
        selected_feature = st.selectbox("Select Feature to Display", features_dict.keys(), key = 'select a feature main choropleth')

        # Generate the Choropleth map of counties in our states of interest
        choropleth(
            geo_data = county_geo_data, 
            selected_feature = selected_feature, 
            university_data = universities_data, 
            features_dict = features_dict,
            title = 'States of Interest by County 2022',
            fields = ['County_State', 'PassRate', 'Income', 'Population', 'Year'],
            aliases = ['County:', 'AP Pass Rate (%):', 'Per-capita Income: $', 'Population:', 'Year:'],
            center = [39.5, -82],
            zoom = 5
        )

        ############################# ▲▲▲▲▲▲ COUNTY CHOROPLETH ▲▲▲▲▲▲ #############################
        ############################# ▼▼▼▼▼▼   PERTURBATIONS   ▼▼▼▼▼▼ #############################

        ##--------------EXPLORE MODEL PREDICTIONS

        # Interactive sentence with dropdowns and inputs
        st.markdown("### Explore Model Predictions")

        # Create columns to arrange components inline
        c1, c2, c3, c4, c5, c6, c7 = st.columns([0.4, 3.5, 1, 1, 1, 2, 5])

        with c1:
            st.write("If")
        with c2:
            # Feature selection dropdown
            model_features = list(model_features_dict.keys())
            selected_model_feature = st.selectbox('', model_features, label_visibility = 'collapsed', key = 'select feature for perturbation')
        with c3:
            st.write("changed by")
        with c4:
            # Number input for value change
            value_change = st.number_input('',
                                        min_value = 0, 
                                        step = model_features_dict[selected_model_feature]['step'], 
                                        label_visibility = 'collapsed')
        with c5:
            # Units (adjust based on feature)
            st.write(model_features_dict[selected_model_feature]['units'] + " in")
        with c6:
            # County selection dropdown
            county_options = county_geo_data[county_geo_data['PassRate'].notna()]['County_State'].unique()
            selected_county = st.selectbox('', county_options, label_visibility = 'collapsed', key = 'select county for perturbation')
        with c7:
            # Get the prediction
            change_direction, prediction_change = predict_ap_pass_rate(selected_county, '2022', selected_model_feature, value_change)
            # Display the prediction result
            if value_change > 0:
                st.write(f"then AP passing rates would **{change_direction}** by **{prediction_change:.2f} percentage points**.")
            else:
                st.write("then no change.")

        ############################# ▲▲▲▲▲▲   PERTURBATIONS    ▲▲▲▲▲▲ #############################
        ############################# ▼▼▼▼▼▼ BROADER CATEGORIES ▼▼▼▼▼▼ #############################

        ##----------BROADER CATEGORIES
        # Interactive sentence with dropdowns and inputs
        st.markdown("### Broad Carnegie Categories over the Years")

        st.markdown('''
        Carnegie [Basic Classifications](https://carnegieclassifications.acenet.edu/carnegie-classification/classification-methodology/basic-classification/) are rather detailed: some [33](https://carnegieclassifications.acenet.edu/wp-content/uploads/2023/03/CCIHE2021-FlowCharts.pdf) were used in the 2021 Basic Classifications scheme. Moreover, the definitions of these classifications have not been consistent across the years. In order to get a picture of how the number of universities in certain classifications has changed over the past few decades, we manually define some broader classifications that can be compared across the various classification schemes employed by Carnegie, and present their frequencies over time below. 
                    
        Most broad classifications have remained steady, with the exception of many universities that were previously not classified now being considered "special focus institutions". That is, institutions confering degrees in one main field.
                    ''')

        broader_categories_counts_df = load_broader_categories()
        # st.dataframe(data = broader_categories_counts_df, width = None, height = None, use_container_width = False, hide_index = True, column_order = None, column_config = None, key = None, on_select = "ignore", selection_mode = "multi-row")
        fig = px.line(broader_categories_counts_df, x = 'year', y = broader_categories_counts_df.columns[1:], markers = True)
        # Hovering and Legend
        fig.update_layout(
            hovermode = 'closest',
            title = 'Broader Carnegie Categories over the Years',
            xaxis_title = "Year",
            yaxis_title = "Counts",
            legend_title = "Broad Carnegie Category"
        )
        st.plotly_chart(fig)

        ############################# ▲▲▲▲▲▲ BROADER CATEGORIES ▲▲▲▲▲▲ #############################
    
    ############################# ▲▲▲▲▲▲ DATA EXPLORATION TAB ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼     OUR MODEL TAB    ▼▼▼▼▼▼ #############################

    with tab3:
        st.markdown("## Our Model")

        st.markdown('''
            ### Feature Selection
                    ''')

        st.markdown('''
            ### Distance-Based Features
                    
            Our approach towards engineering features that aim to capture a tally of nearby universities or a local average of a variable is to take weighted averages using a weighting function that smoothly decreases with distance. 
                    
            Every university, county, and school district in our datasets is assigned a set of coordinates. Whenever we wish to average a variable $X$ (say, the number of dorm rooms on campus) that depends on university, we take the following approach (which will in particular apply to the situation of measuring distance to universities of a certain type by setting $X \equiv 1$). Rather than disregard universities beyond some cutoff distance or to gather only the closest few schools, we take a weighted average of the variable $X$ across all universities according to some function that shrinks with distance. If university $i$ is at distance $d_i$ from a given school district and has value $X = X_i$, then we estimate the feature value of variable $X$ about this school district to be:
                    
            $$
                \widetilde{X} = \sum_i w(d_i) \cdot X_i \quad \\text{where} \quad w(d) = \\frac{1}{1 + \\frac{d}{\\varepsilon}}.
            $$
                    
            In this model, $\\varepsilon > 0$ serves as a smoothing factor which we set to 10 miles. This choice comes with the interpretation of 10 miles being a good scale for what kinds of distances over which are reasonable to expect universities to have a consistant impacts on the education of nearby high schoolers. Universities within 10 miles of a school district will contribute much more to the sum than schools beyond that. 
                    ''')
                    


    ############################# ▲▲▲▲▲▲   OUR MODEL TAB   ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼ MASSACHUSETTS TAB ▼▼▼▼▼▼ #############################

    with tab4: 
        pickled_path = 'MA_pickled/'
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
                        
                Below we summarize the AP performance, availability, and participation in Massachusetts in 2022. 
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
            pickled_plot('MA_score_distribution.pkl', prefix = data_prefix + pickled_path)

        with right_co:
            ##----------CHOROPLETH MAP OF MASSACHUSETTS
            
            st.markdown('''
                #### Model Features and University Statistics
                        
                Below we present AP performance, per-capita income, population, and university data from the year 2022 in and around Massachusetts (we include nearby states New York, Vermont, New Hampshire, Maine, Rhode Island, and Connecticut).
                        ''')

            ############################# ▼▼▼▼▼▼ MASSACHUSETTS CHOROPLETH ▼▼▼▼▼▼ #############################  
            MA_selected_feature = st.selectbox("Select Feature to Display", features_dict.keys(), key = 'select a feature MA choropleth')

            # Generate the Choropleth map of MA counties and nearby universities
            choropleth(
                geo_data = MA_geo_data, 
                selected_feature = MA_selected_feature, 
                university_data = MA_nearby_universities,
                features_dict = features_dict, 
                title = 'Massachusetts by County 2022',
                fields = ['County_State', 'PassRate', 'Income', 'Population', 'Year'],
                aliases = ['County:', 'AP Pass Rate (%):', 'Per-capita Income: $', 'Population:', 'Year:'],
                center = [42.4, -71.7],
                zoom = 8
            )
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
        st.markdown("### Trends with AP Performance")
        

        MA_pickled_plots = ['MA_pass_vs_dorm_bed_land_grant_inverse_distance.pkl', 'MA_pass_vs_dorm_bed_private_inverse_distance.pkl', 'MA_pass_vs_dorm_bed_public_inverse_distance.pkl', 'MA_pass_vs_dorm_bed_R1R2_inverse_distance.pkl', 'MA_pass_vs_dorm_bed_STEM_inverse_distance.pkl', 'MA_pass_vs_enrollment_land_grant_inverse_distance.pkl', 'MA_pass_vs_enrollment_private_nfp_inverse_distance.pkl', 'MA_pass_vs_enrollment_Public_inverse_distance.pkl', 'MA_pass_vs_enrollment_R1R2_inverse_distance.pkl', 'MA_pass_vs_enrollment_STEM_inverse_distance.pkl', 'MA_pass_vs_Land_Grant_Inverse_Distance.pkl', 'MA_pass_vs_per_pupil_expenditures.pkl', 'MA_pass_vs_Private_nfp_Inverse_Distance.pkl', 'MA_pass_vs_Public_Inverse_Distance.pkl', 'MA_pass_vs_R1_R2_inverse_distance.pkl', 'MA_pass_vs_R1R2_Inverse_Distance.pkl', 'MA_pass_vs_school_district_income.pkl', 'MA_pass_vs_school_district_population.pkl', 'MA_pass_vs_STEM_Inverse_Distance.pkl']
        left_co, right_co = st.columns(2)
        with left_co:
            for plot_filepath in MA_pickled_plots[:int(len(MA_pickled_plots) / 2)]:
                print(plot_filepath)
                pickled_plot(plot_filepath, prefix = data_prefix + pickled_path)
                
        with right_co:
            for plot_filepath in MA_pickled_plots[int(len(MA_pickled_plots) / 2):]:
                pickled_plot(plot_filepath, prefix = data_prefix + pickled_path)

    ############################# ▲▲▲▲▲▲ MASSACHUSETTS TAB ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼   WISCONSIN TAB   ▼▼▼▼▼▼ #############################

    with tab5: 
        pickled_path = 'WI_pickled/'
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
                        
                Below we summarize the AP performance, availability, and participation in Wisconsin in 2022. 
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
            pickled_plot('WI_score_distribution.pkl', prefix = data_prefix + pickled_path)

        with right_co:
            ##----------CHOROPLETH MAP OF WISCONSIN
            
            st.markdown('''
                #### Model Features and University Statistics
                        
                Below we present AP performance, per-capita income, population, and university data from the year 2022 in and around Wisconsin (we include nearby states Minnesota, Michigan, Illinois, and Iowa).
                        ''')

            ############################# ▼▼▼▼▼▼ WISCONSIN CHOROPLETH ▼▼▼▼▼▼ #############################
            WI_selected_feature = st.selectbox("Select Feature to Display", features_dict.keys(), key = 'select a feature WI choropleth')

            # Generate the Choropleth map of WI counties and nearby universities
            choropleth(
                geo_data = WI_geo_data, 
                selected_feature = WI_selected_feature, 
                university_data = WI_nearby_universities, 
                features_dict = features_dict,
                title = 'Wisconsin by County 2022',
                fields = ['County_State', 'PassRate', 'Income', 'Population', 'Year'],
                aliases = ['County:', 'AP Pass Rate (%):', 'Per-capita Income: $', 'Population:', 'Year:'],
                center = [44.5, -88.8],
                zoom = 6
            )
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
        st.markdown("### Trends with AP Performance")

    ############################# ▲▲▲▲▲▲ WISCONSIN TAB ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼  GEORGIA TAB  ▼▼▼▼▼▼ #############################

    with tab6: 
        pickled_path = 'GA_pickled/'
        st.markdown("## Georgia")
        st.markdown('''
            We present some of our exploratory results based on the data available for AP performance in Georgia. 
                    
            Georgia is a mid-sized state (24th largest by land area) with a relatively **high population** (8th largest by population). Over half (57.2%) of the state's population is **concentrated in the Atlanta metro area**, which also hosts some of the state's most influential universities like Georgia Institute of Technology and the Unviersity of Georgia. Moreover, Georgia is **33.2% Black or African American**, and offers 9 **historically Black colleges**, ranking third across all states in both respects. Georgia's main industries are **areospace, automotive, and manufacturing**. 
                    
            ### Summary
                    ''')
        
        # Summary and Choropleth
        
        left_co, right_co = st.columns(2)
        with left_co:
            st.markdown('''
                #### AP Performance, Availability, Participation
                        
                Below we summarize the AP performance, availability, and participation in Georgia in 2022. 
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
                Georgia has the lowest passing rate out of all three states considered in our state-by-state analysis, but not by much. Actually, Georgia's passing rates ranked very well in 2022 in comparison to those of its southeastern counterparts, and were almost four percentage points above the national average. The state also experiences some of the worst disparities between Asian, White, and Black student participations.
                        ''')
            
            # Scores
            pickled_plot('GA_score_distribution.pkl', prefix = data_prefix + pickled_path)

        with right_co:
            ##----------CHOROPLETH MAP OF GEORGIA

            st.markdown('''
                #### Model Features and University Statistics
                        
                Below we present AP performance, per-capita income, population, and university data from the year 2022 in and around Georgia (we include nearby states North Carolina, South Carolina, Florida, Alabama, and Tennessee).
                        ''')

            ############################# ▼▼▼▼▼▼ GEORGIA CHOROPLETH ▼▼▼▼▼▼ #############################
            GA_selected_feature = st.selectbox("Select Feature to Display", features_dict.keys(), key = 'select a feature GA choropleth')
            
            # Generate the Choropleth map of GA counties and nearby universities
            choropleth(
                geo_data = GA_geo_data, 
                selected_feature = GA_selected_feature, 
                university_data = GA_nearby_universities, 
                features_dict = features_dict,
                title = 'Georgia by County 2022',
                fields = ['County_State', 'PassRate', 'Income', 'Population', 'Year'],
                aliases = ['County:', 'AP Pass Rate (%):', 'Per-capita Income: $', 'Population:', 'Year:'],
                center = [32.2, -82.9],
                zoom = 6
            )
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
        st.markdown("### Trends with AP Performance")
    
    ############################# ▲▲▲▲▲▲   GEORGIA TAB   ▲▲▲▲▲▲ #############################
    ############################# ▼▼▼▼▼▼  REFERENCES TAB ▼▼▼▼▼▼ #############################

    with tab7:

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
                    ''')

    ############################# ▲▲▲▲▲▲ REFERENCES TAB ▲▲▲▲▲▲ #############################
    ############################# ▲▲▲▲▲▲  TABS  ▲▲▲▲▲▲ #############################
############################# ▲▲▲▲▲▲ APP LAYOUT ▲▲▲▲▲▲ #############################


# Automatically runs when executed
if __name__ == "__main__":
    main()