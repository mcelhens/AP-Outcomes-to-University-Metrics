import pandas as pd
import numpy as np
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
    page_title = 'Counties Choropleth Map',
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

    # Sidebar for feature and year selection
    st.sidebar.header("Map Options")
    selected_feature = st.sidebar.selectbox("Select Feature to Display", features_dict.keys())
    selected_year = st.sidebar.selectbox("Select Year", years)

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
            fields = ['County_State', 'PassRate', 'Income', 'Year'],
            aliases = ['County:', 'AP Pass Rate (%):', 'Per-capita Income: $', 'Year:'],
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
    # Display the map
    st.subheader("Choropleth Map Title")
    # Render the map in Streamlit using folium_static
    #   ...alternatively, can use st_folium rather than folium_static
    st_data = folium_static(m, width = 725)

    #####################################################

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

# Automatically runs when executed
if __name__ == "__main__":
    main()