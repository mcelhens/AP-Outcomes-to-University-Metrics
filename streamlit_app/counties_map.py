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

def main():
    # Example DataFrame for universities
    universities_df = pd.read_csv('carnegie_with_location.csv')[['name', 'latitude', 'longitude']]
    # Prepare your data
    counties_map_data = pd.read_csv('States_Counties_Map_Data.csv')
    # List of features to select from
    features_dict = {
        'AP Pass Rate (3 or higher)': 'PassRate',
        'Per capita Personal Income ($)': 'Income'
    }
    # List of years
    counties_map_data['Year'] = counties_map_data['Year'].astype(str)
    years = counties_map_data['Year'].unique()
    # Add dropdown menus for feature and year selection
    selected_feature = st.selectbox("Select Feature to Display", features_dict.keys())
    selected_year = st.selectbox("Select Year", years)
    # Filter by selected year
    selected_data = counties_map_data[counties_map_data['Year'] == selected_year]
    # Fix geometry type
    selected_data['geometry'] = selected_data['geometry'].apply(lambda p : wkt.loads(str(p)))
    # Instantiate geo_data
    geo_data = gpd.GeoDataFrame(selected_data, geometry = 'geometry')
    # Set the CRS
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
            aliases = ['County:', 'AP Pass Rate:', 'Per-capita Income:', 'Year:'],
            localize = True
        )
    )

    # Add a new layer for university markers
    university_layer = folium.FeatureGroup(name = "Universities")
    # Add markers for each university in the DataFrame
    for _, row in universities_df.iterrows():
        folium.Circle(
            radius = 1000,
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
    # m = folium.Map(location = [39.5, -82], zoom_start = 4.5)
    m = folium.Map(location = [48, -102], zoom_start = 3)
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

# Automatically runs when executed
if __name__ == "__main__":
    main()