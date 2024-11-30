# Data loaders

import pandas as pd

def gimmeCountyIncomes(prefix = ''): 
    """Returns the county level data for per-capita income"""
    incomes = pd.read_csv(prefix + '../data/CAINC_Incomes_Counties_2019_2022.csv')
    incomes = incomes[incomes['Description'] == 'Per capita personal income (dollars) 2/']
    incomes = incomes[incomes['GeoName'].str.contains(',')]
    def split_county_state(geoName, county = True):
        S = geoName.split(', ')
        return S[0] if county else S[len(S) - 1]
    incomes['County'] = incomes.apply(lambda row: split_county_state(row['GeoName'], county = True), axis = 1)
    incomes['State_Abbreviation'] = incomes.apply(lambda row: split_county_state(row['GeoName'], county = False), axis = 1)
    return incomes[['County', 'State_Abbreviation', '2018', '2019', '2020', '2021', '2022']]

def gimmeCountyPopulation(prefix = ''): 
    """Returns the county level data for population"""
    population = pd.read_csv(prefix + '../data/CAINC_Incomes_Counties_2019_2022.csv')
    population = population[population['Description'] == 'Population (persons) 1/']
    population = population[population['GeoName'].str.contains(',')]
    def split_county_state(geoName, county = True):
        S = geoName.split(', ')
        return S[0] if county else S[len(S) - 1]
    population['County'] = population.apply(lambda row: split_county_state(row['GeoName'], county = True), axis = 1)
    population['State_Abbreviation'] = population.apply(lambda row: split_county_state(row['GeoName'], county = False), axis = 1)
    return population[['County', 'State_Abbreviation', '2018', '2019', '2020', '2021', '2022']]


def gimmeCarnegieLimted(prefix = ''): # limitations need revision
    """ Returns the limited Carnegie Dataset
        County information should be applied post-loading where appropriate
    """
    carnegie=pd.read_csv(prefix + 'data/CCIHE2021-PublicData_limited.csv') # Carnegie university clasification as dataframe
    return carnegie

def gimmeCarnegieLimtedWithLocations(prefix = ''): # Added locations to the limited Carnegie set
    """ Returns the limited Carnegie Dataset with locations (lat and long, address)
        County information should be applied post-loading where appropriate
    """
    carnegie_with_locations=pd.read_csv(prefix + 'data/carnegie_with_location.csv') # Carnegie university clasification as dataframe
    return carnegie_with_locations

def gimmeCarnegieFull(prefix = ''):
    """ Returns the limited Carnegie Dataset
        County information should be applied post-loading where appropriate
    """
    carnegie=pd.read_excel(prefix + 'data/CCIHE2021-PublicData.xlsx',sheet_name='Data') # Carnegie university clasification as dataframe
    return carnegie

def gimmeOutcomes(year, prefix = ''):
    """Returns the AP outcomes data with state abbreviations along your choice of year"""

    outcomes=pd.read_excel(prefix + 'data/AP_data_fixed-outcome.xlsx',sheet_name=str(year))

    states_abbrv=pd.read_csv(prefix + 'data/State Abbreviation.csv')

    outcomes=pd.merge(outcomes,states_abbrv,on='State')

    return outcomes

def gimmeParticipation(year, prefix = ''):
    """Returns the AP participation data with state abbreviations along your choice of year"""

    participation=pd.read_excel(prefix + 'data/AP_data_fixed-participation.xlsx',sheet_name=str(year))

    states_abbrv=pd.read_csv(prefix + 'data/State Abbreviation.csv')
    
    participation.columns=participation.iloc[0]
    participation=participation[1:]

    participation=pd.merge(participation,states_abbrv,on='State')

    return participation

def gimmeAvailability(year, prefix = ''):
    """Returns the AP availability data with state abbreviations along your choice of year"""

    availability=pd.read_excel(prefix + 'data/AP_data_fixed-availability.xlsx',sheet_name=str(year))

    states_abbrv=pd.read_csv(prefix + 'data/State Abbreviation.csv')

    availability.columns=availability.iloc[0]
    availability=availability[1:]

    availability=pd.merge(availability,states_abbrv,on='State')

    return availability

def gimmeGA(prefix = ''):
    """Returns GA AP data counties were fixed manually, atlanta is pulled into Fulton county all other non-county districts are dropped"""
    ga_outcomes=pd.read_csv(prefix + 'data/GA_2019-23_counties.csv')
    return ga_outcomes

def gimmeGA_Counties(prefix = ''):
    """Returns GA counties and cities"""

    ga_counties=[]
    with open(prefix + 'data/GA_counties.txt') as topo_file:
        for line in topo_file:
            if 'County' in line:
                county=line[:-8]
                continue
            else: city=line
            ga_counties.append([county,city.replace("\n", "")])
    ga_counties=pd.DataFrame(ga_counties,columns=['County','City'])
    return ga_counties

