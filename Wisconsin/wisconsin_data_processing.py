import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import distance

carnegie = pd.read_csv('../data/carnegie_with_location.csv')
carnegie_full = pd.read_excel('../data/CCIHE2021-PublicData.xlsx',sheet_name='Data') #'carnegie_full' contains some additional columns that we need, which is not in 'carnegie'
uscounties = pd.read_csv('../data/uscounties.csv')

carnegie['stem_rsd'] = carnegie_full['stem_rsd']  # Number of STEM doctoral/research degrees awarded
carnegie['anenr1920'] = carnegie_full['anenr1920']  # Annual enrollment in academic year 2019/20

wisconsin_counties = uscounties[uscounties['state_id'] == 'WI'] # Select only the counties in Wisconsin
wisconsin_counties = wisconsin_counties.reset_index()
wisconsin_counties= wisconsin_counties[['county','lat','lng','population']] # Chose the relevant columns for us.

def closest_five(carnegie_univ_data,lat,long):

    # carnegie_univ_data is meant to be sublist/subdataframe of carnegie dataset (with location)
    # lat is meant to be the latitude of the location (usually county)
    # long is meant to be the longitude of the location
    # Returns the average distance to the five closest universities from the supplied list.

    univ_distance = {'unitid':[],'distance':[]}
    for i in carnegie_univ_data.index:
        univ_distance['unitid']=univ_distance['unitid']+[carnegie_univ_data.unitid[i]]
        univ_distance['distance']=univ_distance['distance']+[distance((lat,long),(carnegie_univ_data.latitude[i],carnegie_univ_data.longitude[i])).miles]
    univ_distance = pd.DataFrame(univ_distance)
    closest_five = univ_distance.sort_values(by = 'distance')[:5]
    return np.mean(closest_five['distance'].values)

carnegie_r1r2 = carnegie[carnegie['basic2021'].isin([15,16])] # new dataframe containing only R1/R2 universities.
wisconsin_counties['closest_five_r1r2_avg'] = wisconsin_counties.apply(lambda x: closest_five(carnegie_r1r2,x.lat, x.lng), axis=1)
# Create a new column with average distance to five closest R1/R2 universities from each county.

carnegie_public = carnegie[carnegie['control'] == 1] # Public universities
wisconsin_counties['closest_five_public_avg'] = wisconsin_counties.apply(lambda x: closest_five(carnegie_public,x.lat, x.lng), axis=1)
# Average distance to five closest public universities from each county

carnegie_private_notprofit = carnegie[carnegie['control'] == 2] # Private not for profit universities
wisconsin_counties['closest_five_private_nfp_avg'] = wisconsin_counties.apply(lambda x: closest_five(carnegie_private_notprofit,x.lat, x.lng), axis=1)
# Average distance to five closest private not for profit universities from each county

carnegie_landgrnt = carnegie[carnegie['landgrnt'] == 1] # Landgrant universities
wisconsin_counties['closest_five_landgrnt_avg'] = wisconsin_counties.apply(lambda x: closest_five(carnegie_landgrnt,x.lat, x.lng), axis=1)
# Average distance to five closest landgrant universities from each county

carnegie_stem = carnegie[carnegie['stem_rsd'] > 0] # We define STEM institute to be the one offering at least one STEM research/scholarship doctoral degrees.
wisconsin_counties['closest_five_stem_avg'] = wisconsin_counties.apply(lambda x: closest_five(carnegie_stem,x.lat, x.lng), axis=1)
# Average distance to five closest stem universities from each county

def closest_five_enrollment(carnegie_univ_data,lat,long):

    # carnegie_univ_data is meant to be sublist/subdataframe of carnegie dataset (with location and anenr1920)
    # lat is meant to be the latitude of the location (usually county)
    # long is meant to be the longitude of the location
    # Returns the average annual enrollment of the five closest universities from the supplied list.

    univ_enrollment = {'unitid':[],'distance':[],'enrollment':[]}
    for i in carnegie_univ_data.index:
        univ_enrollment['unitid']=univ_enrollment['unitid']+[carnegie_univ_data.unitid[i]]
        univ_enrollment['distance']=univ_enrollment['distance']+[distance((lat,long),(carnegie_univ_data.latitude[i],carnegie_univ_data.longitude[i])).miles]
        univ_enrollment['enrollment'] = univ_enrollment['enrollment'] + [carnegie_univ_data['anenr1920'][i]]
    univ_enrollment = pd.DataFrame(univ_enrollment)
    closest_five = univ_enrollment.sort_values(by = 'distance')[:5]
    return np.mean(closest_five['enrollment'].values)

# Analogous columns for average enrollments
wisconsin_counties['closest_five_avg_enrollment_r1r2'] = wisconsin_counties.apply(lambda x: closest_five_enrollment(carnegie_r1r2,x.lat, x.lng), axis=1)
wisconsin_counties['closest_five_avg_enrollment_public'] = wisconsin_counties.apply(lambda x: closest_five_enrollment(carnegie_public,x.lat, x.lng), axis=1)
wisconsin_counties['closest_five_avg_enrollment_private_nfp'] = wisconsin_counties.apply(lambda x: closest_five_enrollment(carnegie_private_notprofit,x.lat, x.lng), axis=1)
wisconsin_counties['closest_five_avg_enrollment_landgrnt'] = wisconsin_counties.apply(lambda x: closest_five_enrollment(carnegie_landgrnt,x.lat, x.lng), axis=1)
wisconsin_counties['closest_five_avg_enrollment_stem'] = wisconsin_counties.apply(lambda x: closest_five_enrollment(carnegie_stem,x.lat, x.lng), axis=1)

# Analogous columns for average number of dorm rooms
def closest_five_rooms(carnegie_univ_data,lat,long):
        # carnegie_univ_data is meant to be sublist/subdataframe of carnegie dataset (with location and anenr1920)
    # lat is meant to be the latitude of the location (usually county)
    # long is meant to be the longitude of the location
    # Returns the average dorm rooms of the five closest universities from the supplied list.

    univ_enrollment = {'unitid':[],'distance':[],'rooms':[]}
    for i in carnegie_univ_data.index:
        univ_enrollment['unitid']=univ_enrollment['unitid']+[carnegie_univ_data.unitid[i]]
        univ_enrollment['distance']=univ_enrollment['distance']+[distance((lat,long),(carnegie_univ_data.latitude[i],carnegie_univ_data.longitude[i])).miles]
        univ_enrollment['rooms'] = univ_enrollment['rooms'] + [carnegie_univ_data['rooms'][i]]
    univ_enrollment = pd.DataFrame(univ_enrollment)
    closest_five = univ_enrollment.sort_values(by = 'distance')[:5]
    return np.mean(closest_five['rooms'].values)

# Import the data_loaders.py file from the main directory
import sys
sys.path.insert(1, '../')
import data_loaders

# Create a new dataframe with per capita income for each county in the years 2018 to 2022.
incomes = data_loaders.gimmeCountyIncomes('../')
incomes = incomes[incomes['State_Abbreviation']=='WI'] # Pick only the counties in Wisconsin
incomes = pd.concat([pd.DataFrame([['Shawano','WI',42033,43883,46611,50004,50444]], columns=incomes.columns), incomes], ignore_index=True)
# Shawano county is not in the list returned by the gimmeCountyIncomes function, so we manually add the data.

# Analogous steps for population for each county in the years 2018 to 2022
population = data_loaders.gimmeCountyPopulation('../')
population=population[population['State_Abbreviation']=='WI']
population = pd.concat([pd.DataFrame([['Shawano','WI',40725,40794,40873,40812,40886]], columns=population.columns), population], ignore_index=True)

# New dataframe with population and per capita income for the five years, with "Year" as a column
wisconsin_data = {'county':[],'year':[],'population':[],'per_capita_income':[]}
years = ['2018','2019','2020','2021','2022']
for county in incomes['County'].values:
    for year in years:
        wisconsin_data['county'] = wisconsin_data['county']+[county]
        wisconsin_data['year'] = wisconsin_data['year']+[year]
        wisconsin_data['population'] = wisconsin_data['population']+[int(population[population['County'] == county][year].values[0])]
        wisconsin_data['per_capita_income']= wisconsin_data['per_capita_income']+[int(incomes[incomes['County'] == county][year].values[0])]
wisconsin_data = pd.DataFrame(wisconsin_data)

# Read the AP performance data for Wisconsin for the five years 2018/19 to 2022/23
wisconsin_ap = pd.read_csv('../data/Wisconsin/Wisconsin_combined.csv')

# We use the year 2018 for academic year 2018/19 and so on.
def clean_year(year):
    return year[:-3]
wisconsin_ap['Year']=wisconsin_ap['Year'].apply(clean_year)

# For consistency, we change county name 'St. Croix' to 'Saint Croix'.
wisconsin_ap=wisconsin_ap.replace(to_replace='Saint Croix',value='St. Croix')
wisconsin_data=wisconsin_data.replace(to_replace='Saint Croix',value='St. Croix')


## We combine all the relevant columns to one dataframe.
lat = []
long = []
pop = []
pci = []
r1r2 = []
public = []
private_notprofit = []
landgrnt  = []
stem  = []
enrollment_r1r2 = []
enrollment_public = []
enrollment_private_nfp = []
enrollment_landgrnt = []
enrollment_stem = []
rooms_r1r2 = []
rooms_public = []
rooms_private_nfp = []
rooms_landgrant = []
rooms_stem = []


for i in wisconsin_ap.index:
    county = wisconsin_ap.iloc[i].COUNTY
    year = wisconsin_ap.iloc[i].Year
    lat = lat + [wisconsin_counties[(wisconsin_counties['county']==county)].lat.values[0]]
    long = long + [wisconsin_counties[(wisconsin_counties['county']==county)].lng.values[0]]
    pop = pop + [wisconsin_data[(wisconsin_data['county']==county) & (wisconsin_data['year'] == year)].population.values[0]]
    pci = pci + [wisconsin_data[(wisconsin_data['county']==county) & (wisconsin_data['year'] == year)].per_capita_income.values[0]]
    r1r2 = r1r2 + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_r1r2_avg.values[0]]
    public = public + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_public_avg.values[0]]
    private_notprofit = private_notprofit + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_private_nfp_avg.values[0]]
    landgrnt = landgrnt + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_landgrnt_avg.values[0]]
    stem = stem + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_stem_avg.values[0]]
    enrollment_r1r2 = enrollment_r1r2 + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_avg_enrollment_r1r2.values[0]]
    enrollment_public = enrollment_public + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_avg_enrollment_public.values[0]]
    enrollment_private_nfp = enrollment_private_nfp + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_avg_enrollment_private_nfp.values[0]]
    enrollment_landgrnt = enrollment_landgrnt + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_avg_enrollment_landgrnt.values[0]]
    enrollment_stem = enrollment_stem + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_avg_enrollment_stem.values[0]]
    rooms_r1r2 = rooms_r1r2 + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_avg_dormrooms_r1r2.values[0]]
    rooms_public = rooms_public + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_avg_dormrooms_public.values[0]]
    rooms_private_nfp = rooms_private_nfp + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_avg_dormrooms_private_nfp.values[0]]
    rooms_landgrant = rooms_landgrant + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_avg_dormrooms_landgrnt.values[0]]
    rooms_stem = rooms_stem + [wisconsin_counties[(wisconsin_counties['county']==county)].closest_five_avg_dormrooms_stem.values[0]]

wisconsin_ap['Latitude'] = lat
wisconsin_ap['Longitude'] = long
wisconsin_ap['population'] = pop
wisconsin_ap['per_capita_income'] = pci
wisconsin_ap['closest_five_r1r2_avg'] = r1r2
wisconsin_ap['closest_five_public_avg'] = public
wisconsin_ap['closest_five_private_nfp_avg'] = private_notprofit
wisconsin_ap['closest_five_landgrnt_avg'] = landgrnt
wisconsin_ap['closest_five_stem_avg'] = stem
wisconsin_ap['closest_five_avg_enrollment_r1r2'] = enrollment_r1r2
wisconsin_ap['closest_five_avg_enrollment_public'] = enrollment_public
wisconsin_ap['closest_five_avg_enrollment_private_nfp'] = enrollment_private_nfp
wisconsin_ap['closest_five_avg_enrollment_landgrnt'] = enrollment_landgrnt
wisconsin_ap['closest_five_avg_enrollment_stem'] = enrollment_stem
wisconsin_ap['closest_five_avg_dormrooms_r1r2'] = rooms_r1r2
wisconsin_ap['closest_five_avg_dormrooms_public'] = rooms_public
wisconsin_ap['closest_five_avg_dormrooms_private_nfp'] = rooms_private_nfp
wisconsin_ap['closest_five_avg_dormrooms_landgrant'] = rooms_landgrant
wisconsin_ap['closest_five_avg_dormrooms_stem'] = rooms_stem


## wisconsin_ap is the dataframe we will use for statistical analysis. So let's save it as a csv file.
wisconsin_ap.to_csv('../data/Wisconsin/train_test_split/Wisconsin_closest_five_method.csv')

# train test split for the dataframe. Also, we save them to separate csv files.
from sklearn.model_selection import train_test_split
training, testing = train_test_split(wisconsin_ap, test_size = 0.2, random_state = 226)
training.to_csv('../data/Wisconsin/train_test_split/training.csv')
testing.to_csv('../data/Wisconsin/train_test_split/testing.csv')