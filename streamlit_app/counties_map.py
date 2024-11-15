import pandas as pd
import numpy as np
import plotly.express as px
import math
import streamlit as st
from streamlit_folium import st_folium, folium_static
from pathlib import Path
import geopandas as gpd
import folium
from shapely import wkt

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    layout = 'wide',
    page_title = 'AP Outcomes vs University Metrics',
    page_icon = 'https://preview.redd.it/uxmxgdgoqdz81.jpg?width=640&crop=smart&auto=webp&s=ca71011f6de31eb654b50a972b14806b19e98e52', # This is an emoji shortcode. Could be a URL too.
)

years = ['2019', '2020', '2021', '2022']
states_of_interest = ['GA', 'WI', 'MA']
features_dict = {
    'AP Pass Rate (3 or higher)': 'PassRate',
    'Per capita Personal Income ($)': 'Income',
    'Population (persons)': 'Population'
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

@st.cache_data
def load_choropleth_data():
    # Load universities data
    universities_df = pd.read_csv('../data/carnegie_with_location.csv')[['name', 'stabbr', 'latitude', 'longitude']]
    universities_df = universities_df[universities_df['stabbr'].isin(states_of_interest)]
    
    # Load counties data
    counties_map_data = pd.read_csv('States_Counties_Map_Data.csv')
    counties_map_data['Year'] = counties_map_data['Year'].astype(str)

    # Make hashable geometries for caching
    counties_map_data['geometry'] = counties_map_data['geometry'].apply(lambda p : wkt.loads(str(p)))
    geo_data = gpd.GeoDataFrame(counties_map_data, geometry = 'geometry')
    geo_data.set_crs(epsg = 4326, inplace = True)
    geo_data['geometry'] = geo_data['geometry'].apply(lambda geom: geom.wkt)

    # Pre-filter on Year
    geo_data_dict = {
        year : geo_data[geo_data['Year'] == year] for year in years
    }

    return geo_data_dict, universities_df

@st.cache_data
def load_broader_categories():
    return pd.read_csv('../data/broader_categories_counts.csv')

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

def main():

    st.markdown('# The Relationship between AP Test Outcomes and Nearby Universities')

    tab1, tab2, tab3 = st.tabs(["Home", "Data Exploration", "Our Model"])

    with tab1:
        st.markdown("## Home")
        st.markdown('''
        This project was designed to investigate the potential relationship between **[AP exam](https://apstudents.collegeboard.org/what-is-ap) performance** and the **presence of nearby universities**. It was initially hypothesized that local (especially R1/R2 or public) universities would contribute to better pass rates for AP exams in their vicinities as a result of their various outreach, dual-enrollment, tutoring, and similar programs for high schoolers. 

        AP exams are standardized tests widely available at high schools across the United States. During the 2022-2023 school year, [79\%](https://arc.net/l/quote/ewvgnupe) of all public high school students attended schools offering at least five AP courses. These exams are popular for their potential to earn college credits during high school by achieving high scores. In fact, high scores in most AP exams are eligible to receive college credits at roughly [2000](https://apcentral.collegeboard.org/media/pdf/program-summary-report-2024.pdf) higher-education institutions. 

        AP exams are scored on a whole number scale between 1 (lowest) and 5 (highest). A student is said to *pass* their AP exam if they score a 3 or higher on the exam. The *pass rate* of a locality would be the proportion of AP exams passed out of all exams taken by its students during a single year. AP outcomes are often correlated to measures of socioeconomic factors: a [recent study](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4574500) confirmed that negative socioeconomic factors have a strong negative influence on exam scores; as well as being a non-native English language speaker. 

        Beyond these socioeconomic factors, we would like to measure the strength of the effect of universities on AP outcomes. Without a clear source of data on all high school outreach programs offered by US universities, we make use of the various classifications offered by the [Carnegie Classifications of Institutions of Higher Education](https://carnegieclassifications.acenet.edu/). Of particular interest include R1 and R2 (i.e., doctoral with very high or high research activity, respectively), public, or private institutions. Other minority-serving aspects are also considered, such as historically Black, Hispanic-serving, and tribal colleges.

        **Authors (alphabetical)**: *Prabhat Devkota, Shrabana Hazra, Jung-Tsung Li, Shannon J. McElhenney, Raymond Tana*
                    ''')

    with tab2:
        st.markdown("## Data Exploration")

        ##----------CHOROPLETH MAP
        # Display select boxes
        st.markdown("### County-level Choropleth Map")

        st.markdown("#### Map Options")
        selected_feature = st.selectbox("Select Feature to Display", features_dict.keys())
        selected_year = st.selectbox("Select Year", years)

        # Load in cached data
        geo_data_dict, universities_df = load_choropleth_data()
        # Filter based on Year
        geo_data = geo_data_dict[selected_year]

        # Reconstruct geometries from WKT strings
        geo_data['geometry'] = geo_data['geometry'].apply(wkt.loads)
        geo_data = gpd.GeoDataFrame(geo_data, geometry = 'geometry')
        geo_data.set_crs(epsg = 4326, inplace = True)

        # Define the choropleth layer based on the selected feature and year
        choropleth_layer = folium.Choropleth(
            geo_data = geo_data,
            name = 'Choropleth Map',
            data = geo_data,
            columns = ['GEOID', features_dict[selected_feature]],
            key_on = 'feature.properties.GEOID',
            fill_color = 'YlOrRd',
            nan_fill_color = 'lightgrey',
            fill_opacity = 0.7,
            line_opacity = 0.2,
            legend_name = f'{selected_feature} ({selected_year}) by County'
        )

        # Define tooltips with county name (including state abbreviation) and selected feature value
        county_tooltips = folium.GeoJson(
            geo_data,
            name = 'tooltips',
            control = False,
            style_function = lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
            tooltip = folium.features.GeoJsonTooltip(
                fields = ['County_State', 'PassRate', 'Income', 'Population', 'Year'],
                aliases = ['County:', 'AP Pass Rate (%):', 'Per-capita Income: $', 'Population:', 'Year:'],
                localize = True
            )
        )

        # Add a new layer for university markers
        university_layer = folium.FeatureGroup(name = "Universities")
        # Add markers for each university in the DataFrame
        for _, row in universities_df.iterrows():
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
        m = folium.Map(location = [39.5, -82], zoom_start = 4.5)
        # m = folium.Map(location = [48, -102], zoom_start = 3)
        # Add choropleth layer to the map
        choropleth_layer.add_to(m)
        # Add the county tooltips to the map
        county_tooltips.add_to(m)
        # Add the university layer to the map
        university_layer.add_to(m)
        # Add a layer control to toggle layers
        folium.LayerControl().add_to(m)
        # Render the map in Streamlit using folium_static
        #   ...alternatively, can use st_folium rather than folium_static
        st_data = folium_static(m, width = 725)

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
            selected_model_feature = st.selectbox('', model_features, label_visibility = 'collapsed')
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
            county_options = geo_data[geo_data['PassRate'].notna()]['County_State'].unique()
            selected_county = st.selectbox('', county_options, label_visibility = 'collapsed')
        with c7:
            # Get the prediction
            change_direction, prediction_change = predict_ap_pass_rate(selected_county, selected_year, selected_model_feature, value_change)
            # Display the prediction result
            if value_change > 0:
                st.write(f"then AP passing rates would **{change_direction}** by **{prediction_change:.2f} percentage points**.")
            else:
                st.write("then no change.")

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
    with tab3:
        st.markdown("## Our Model")

# Automatically runs when executed
if __name__ == "__main__":
    main()