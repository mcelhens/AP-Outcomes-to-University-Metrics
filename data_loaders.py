# Data loaders

import pandas as pd


def gimmeCarnegieLimted():
    """ Returns the limited Carnegie Dataset
        County information should be applied post-loading where appropriate
    """
    carnegie=pd.read_excel('data/CCIHE2021-PublicData_limited.xlsx',sheet_name='Data') # Carnegie university clasification as dataframe
    return carnegie

def gimmeOutcomes(year):
    """Returns the AP outcomes data with state abbreviations along your choice of year"""

    outcomes=pd.read_excel('data/AP_data_fixed-outcome.xlsx',sheet_name=str(year))

    states_abbrv=pd.read_csv('data/State Abbreviation.csv')

    outcomes=pd.merge(outcomes,states_abbrv,on='State')

    return outcomes

def gimmeParticipation(year):
    """Returns the AP participation data with state abbreviations along your choice of year"""

    participation=pd.read_excel('data/AP_data_fixed-participation.xlsx',sheet_name=str(year))

    states_abbrv=pd.read_csv('data/State Abbreviation.csv')
    
    participation.columns=participation.iloc[0]
    participation=participation[1:]

    participation=pd.merge(participation,states_abbrv,on='State')

    return participation

def gimmeAvailability(year):
    """Returns the AP availability data with state abbreviations along your choice of year"""

    availability=pd.read_excel('data/AP_data_fixed-availability.xlsx',sheet_name=str(year))

    states_abbrv=pd.read_csv('data/State Abbreviation.csv')

    availability.columns=availability.iloc[0]
    availability=availability[1:]

    availability=pd.merge(availability,states_abbrv,on='State')

    return availability

def gimmeGA(year):
    """Returns GA AP data along your choice of year counties were fixed manually"""
    ga_outcomes=pd.read_excel('data/GA_AP_combined-limited.xlsx',sheet_name=str(year),index_col='County')

    temp=pd.DataFrame({'Passing Ratio':ga_outcomes['Passing Ratio'].groupby(ga_outcomes.index).mean(),'Number of Tests':ga_outcomes['NUMBER_TESTS_TAKEN'].groupby(ga_outcomes.index).sum(),'Number of Tests Passed':ga_outcomes['NOTESTS_3ORHIGHER'].groupby(ga_outcomes.index).sum()}).reset_index()

    ga_outcomes=temp
    return ga_outcomes

def gimmeIN():
    """Returns Indiana (IN) AP data"""
    in_outcomes = pd.read_csv(path_to_fixed_IN_data)

    return outcomes



def gimmeGA_Counties():
    """Returns GA counties and cities"""

    ga_counties=[]
    with open('data/GA_counties.txt') as topo_file:
        for line in topo_file:
            if 'County' in line:
                county=line[:-7]
                continue
            else: city=line
            ga_counties.append([county,city.replace("\n", "")])
    ga_counties=pd.DataFrame(ga_counties,columns=['County','City'])
    return ga_counties

