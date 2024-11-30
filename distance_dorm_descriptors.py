import pandas as pd
from geopy.distance import distance
import numpy as np

def closest_five(carnegie_univ_data,lat,long):

    # carnegie_univ_data is meant to be sublist/subdataframe of carnegie dataset (with location)
    # lat is meant to be the latitude of the location (usually county)
    # long is meant to be the longitude of the location
    # Returns the average distance to the five closest universities from the supplied list.

    univ_distance = {'unitid':[],'distance':[]}
    for i in carnegie_univ_data.index:
        univ_distance['unitid']=univ_distance['unitid']+[carnegie_univ_data.unitid[i]]
        univ_distance['distance']=univ_distance['distance']+[distance((lat,long),(carnegie_univ_data.latitude[i],carnegie_univ_data.longitude[i])).km]
    univ_distance = pd.DataFrame(univ_distance)
    closest_five = univ_distance.sort_values(by = 'distance')[:5]
    return np.mean(closest_five['distance'].values)

def closest_five_enrollment(carnegie_univ_data,lat,long):

    # carnegie_univ_data is meant to be sublist/subdataframe of carnegie dataset (with location and anenr1920)
    # lat is meant to be the latitude of the location (usually county)
    # long is meant to be the longitude of the location
    # Returns the average annual enrollment of the five closest universities from the supplied list.

    univ_enrollment = {'unitid':[],'distance':[],'enrollment':[]}
    for i in carnegie_univ_data.index:
        univ_enrollment['unitid']=univ_enrollment['unitid']+[carnegie_univ_data.unitid[i]]
        univ_enrollment['distance']=univ_enrollment['distance']+[distance((lat,long),(carnegie_univ_data.latitude[i],carnegie_univ_data.longitude[i])).km]
        univ_enrollment['enrollment'] = univ_enrollment['enrollment'] + [carnegie_univ_data['anenr1920'][i]]
    univ_enrollment = pd.DataFrame(univ_enrollment)
    closest_five = univ_enrollment.sort_values(by = 'distance')[:5]
    return np.mean(closest_five['enrollment'].values)

def closest_five_rooms(carnegie_univ_data,lat,long):
        # carnegie_univ_data is meant to be sublist/subdataframe of carnegie dataset (with location and anenr1920)
    # lat is meant to be the latitude of the location (usually county)
    # long is meant to be the longitude of the location
    # Returns the average dorm rooms of the five closest universities from the supplied list.

    univ_enrollment = {'unitid':[],'distance':[],'rooms':[]}
    for i in carnegie_univ_data.index:
        univ_enrollment['unitid']=univ_enrollment['unitid']+[carnegie_univ_data.unitid[i]]
        univ_enrollment['distance']=univ_enrollment['distance']+[distance((lat,long),(carnegie_univ_data.latitude[i],carnegie_univ_data.longitude[i])).km]
        univ_enrollment['rooms'] = univ_enrollment['rooms'] + [carnegie_univ_data['rooms'][i]]
    univ_enrollment = pd.DataFrame(univ_enrollment)
    closest_five = univ_enrollment.sort_values(by = 'distance')[:5]
    return np.mean(closest_five['rooms'].values)


