import pandas as pd
import geopandas as gpd
import numpy as np
import math
import streamlit as st
import folium
from streamlit_folium import st_folium
from pathlib import Path

# st.set_page_config(layout="wide")

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Counties Choropleth Map',
    page_icon='https://preview.redd.it/uxmxgdgoqdz81.jpg?width=640&crop=smart&auto=webp&s=ca71011f6de31eb654b50a972b14806b19e98e52', # This is an emoji shortcode. Could be a URL too.
)

import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import streamlit as st
from streamlit_folium import folium_static

# Cache and load data, filtered to just the states of interest
@st.cache_data
def load_and_filter_shapefile(state_fips_of_interest):
    gdf = gpd.read_file('../data/county_shapes/cb_2018_us_county_5m.shp')
    gdf_states = gdf[gdf['STATEFP'].isin(state_fips_of_interest)]
    return gdf_states

# Prepare the data for demonstration purposes
@st.cache_data
def prepare_data(geoids):
    # Fix a random seed
    np.random.seed(5)
    years = ['2020', '2021', '2022', '2023']
    data_frames = []
    for year in years:
        data = pd.DataFrame({
            'GEOID': geoids.astype(str),
            'Year': year,
            'Population': np.random.randint(10000, 1000000, size=len(geoids)),
            'Income': np.random.randint(30000, 100000, size=len(geoids)),
            'Unemployment': np.random.uniform(2.0, 10.0, size=len(geoids))
        })
        data_frames.append(data)
    full_data = pd.concat(data_frames, ignore_index=True)
    return full_data

def main():
    # Example DataFrame for universities
    universities_df = pd.read_csv('carnegie_with_location.csv')[['name', 'latitude', 'longitude']]

    # Mapping of state abbreviations to FIPS codes
    state_fips_codes = {
        'WI': '55',
        'MA': '25',
        'GA': '13'
    }
    states_of_interest = ['WI', 'MA', 'GA']
    state_fips_of_interest = [state_fips_codes[state] for state in states_of_interest]

    # Load and filter the shapefile
    gdf_states = load_and_filter_shapefile(state_fips_of_interest)

    # Map 'STATEFP' to state abbreviation
    fips_to_state = {code: abbr for abbr, code in state_fips_codes.items()}
    gdf_states['STATE'] = gdf_states['STATEFP'].map(fips_to_state)
    gdf_states['GEOID'] = gdf_states['GEOID'].astype(str)

    # Prepare your data
    data = prepare_data(gdf_states['GEOID'])
    # Merge 'NAME' and 'STATE' into data to create 'County_State'
    # Select only the necessary columns to merge
    county_info = gdf_states[['GEOID', 'NAME', 'STATE']]
    # Merge with data
    data = data.merge(county_info, on='GEOID', how='left')
    # Create 'County_State' column
    data['County_State'] = data['NAME'] + ', ' + data['STATE']

    # List of features to select from
    features = ['Population', 'Income', 'Unemployment']
    # List of years
    years = ['2020', '2021', '2022', '2023']
    # Add dropdown menus for feature and year selection
    selected_feature = st.selectbox("Select Feature to Display", features)
    selected_year = st.selectbox("Select Year", years)

    # Filter data for the selected year
    data_selected = data[data['Year'] == selected_year]
    # Merge data with geospatial data
    gdf_merged = gdf_states.merge(
        data_selected[['GEOID', selected_feature, 'County_State']],
        on='GEOID',
        how='left'
    )
    # Add the selected year to the merged GeoDataFrame
    gdf_merged['Year'] = selected_year

    # Define the choropleth layer based on the selected feature and year
    choropleth_layer = folium.Choropleth(
        geo_data = gdf_merged,
        name = 'Choropleth Map',
        data  = data_selected,
        columns = ['GEOID', selected_feature],
        key_on = 'feature.properties.GEOID',
        fill_color = 'YlOrRd',
        fill_opacity = 0.7,
        line_opacity = 0.2,
        legend_name = f'{selected_feature} ({selected_year}) by County'
    )

    # Define the tooltip fields and aliases
    tooltip_fields = ['County_State', selected_feature, 'Year']
    tooltip_aliases = ['County:', f'{selected_feature}:', 'Year:']
    # Define tooltips with county name (including state abbreviation) and selected feature value
    county_tooltips = folium.GeoJson(
        gdf_merged,
        name = 'tooltips',
        control = False,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
        tooltip=folium.features.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True
        )
    )

    # Add a new layer for university markers
    university_layer = folium.FeatureGroup(name = "Universities")
    # Add markers for each university in the DataFrame
    for _, row in universities_df.iterrows():
        folium.Circle(
            radius = 1000,
            fill = False,
            color="black",
            fill_color="orange",
            opacity = 1,
            fill_opacity=0.2,
            weight=2,
            location = [row['latitude'], row['longitude']],
            popup = folium.Popup(f"{row['name']}", max_width = 300),
            tooltip = row['name']
        ).add_to(university_layer)


    # Map center coordinates
    # m = folium.Map(location = [39.5, -82], zoom_start = 4.5)
    m = folium.Map(location=[48, -102], zoom_start = 3)
    # Add choropleth layer to the map
    choropleth_layer.add_to(m)
    # Add the county tooltips to the map
    county_tooltips.add_to(m)
    # Add the university layer to the map
    university_layer.add_to(m)
    # Add a layer control to toggle layers
    folium.LayerControl().add_to(m)

    # Render the map in Streamlit using folium_static
    # Alternatively, can use st_folium rather than folium_static
    st_data = folium_static(m, width = 725)

# Automatically runs when executed
if __name__ == "__main__":
    main()