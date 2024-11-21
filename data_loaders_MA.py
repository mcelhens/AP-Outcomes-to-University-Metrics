# Data loaders

import pandas as pd

def gimmeMA(prefix = ''):
    '''
    Massachusetts AP score dataset.
    Return the AP scores 3-5, sorted by 13 counties.
    Data listed are from 2019 to 2022.
    '''   
    return pd.read_excel(prefix + 'data/MA_data/county_passrate_19_22.xlsx', sheet_name='2019-22')