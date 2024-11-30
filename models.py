import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestRegressor
import joblib

N_CORES = joblib.cpu_count(only_physical_cores=True)

def wisconsin_pred_2023_24(lat, long):
    inc_pop = pd.read_csv('data/County_Income_Population_2023.csv')
    training = pd.read_csv('data/Wisconsin/train_test_split/training.csv')
    wisconsin_data = pd.read_csv('data/Wisconsin/train_test_split/Wisconsin_closest_five_method.csv')
    variables = training.columns[9:]
    X_train = training[variables]
    y_train = training['PERCENT_3_OR_ABOVE']
    random_forest = RandomForestRegressor(min_samples_leaf=5, random_state=0, n_jobs=N_CORES)
    random_forest.fit(X_train,y_train)
    location = geolocator.reverse(str(lat)+','+str(long))
    if location.raw.get('address').get('country') == 'United States' and location.raw.get('address').get('state') == 'Wisconsin':
        county = location.raw.get('address').get('county').rsplit(' ', 1)[0]
        state = location.raw.get('address').get('ISO3166-2-lvl4')[-2:]
        pci = inc_pop[(inc_pop.County == county) & (inc_pop.State_Abbreviation == state)]
        inc = int(pci.Income.values[0])
        pop = int(pci.Population.values[0])
        row = wisconsin_data[wisconsin_data.COUNTY == county][wisconsin_data.columns[9:]].iloc[0]
        row = pd.DataFrame(row).T
        row['population'] = pop
        row['per_capita_income'] = inc
        return random_forest.predict(row)
    else:
        return "Invalid coordinates"
    

def WI_predict_perturb(county,pci_change=0,pop_change=0):
    
    #Takes a county in Wisconsin along with the change in income and population, and returns change in pass rate predicted by our model.

    wisconsin_data = pd.read_csv('data/Wisconsin/train_test_split/Wisconsin_closest_five_method.csv')
    random_forest=joblib.load('data/WI_pickled/WI_random_forest_model.pkl')
    row = wisconsin_data[wisconsin_data.COUNTY == county][wisconsin_data.columns[9:]].iloc[0]
    inc_pop = pd.read_csv('data/County_Income_Population_2023.csv')
    pci = inc_pop[(inc_pop.County == county) & (inc_pop.State_Abbreviation == 'WI')]
    inc = int(pci.Income.values[0])
    pop = int(pci.Population.values[0])
    row = pd.DataFrame(row).T
    row['population'] = pop
    row['per_capita_income'] = inc
    prediction = random_forest.predict(row)
    changed_row = row.copy(deep=True)
    changed_row['population'] = pop+pop_change
    changed_row['per_capita_income'] = inc + pci_change
    new_prediction = random_forest.predict(changed_row)
    return new_prediction-prediction

def predict_perturb(county,state,pci_change=0,pop_change=0,public_change=0,private_change=0):
    
    '''Takes a county in WI, GA or NC or a school district in MA and the state abbreviation, along with the change in 
    income, population, average distance to five public and private universities, and returns change in pass rate predicted by our model.'''

    wisconsin_data = pd.read_csv('data/Wisconsin/train_test_split/Wisconsin_closest_five_method.csv')
    massachusetts_data = pd.read_csv('data/Massachusetts/AP_data_combined_18_22.csv')
    georgia_data = pd.read_pickle('data/GA_pickled/ga_outcomes_full_data.pkl')
    northcarolina_data = pd.read_csv('data/North_Carolina/train_test_split/northcarolina_closest_five_method.csv')
    georgia_data.rename(columns={'closest_five_avg_dormrooms_landgrnt':'closest_five_avg_dormrooms_landgrant',
                             'closest_five_landgrant_avg':'closest_five_landgrnt_avg',
                             'closest_five_avg_enrollment_landgrant':'closest_five_avg_enrollment_landgrnt'},inplace=True)
    northcarolina_data.rename(columns={'closest_five_private_notprofit_avg':'closest_five_private_nfp_avg','County':'COUNTY'},inplace=True)
    northcarolina_data=northcarolina_data.dropna()
    model=joblib.load('data/Four_states_combined_models_pickled/four_states_combined_xgboost_pca_model.pkl')
    features = wisconsin_data.columns[9:]
    if state == 'WI':
        row = wisconsin_data[wisconsin_data.COUNTY == county][features].iloc[0]
    elif state == 'MA':
        row = massachusetts_data[massachusetts_data.COUNTY == county][features].iloc[0]
    elif state == 'GA':
        row = georgia_data[georgia_data.COUNTY == county][features].iloc[0]
    elif state == 'NC':
        row = northcarolina_data[northcarolina_data.COUNTY == county][features].iloc[0]
    inc_pop = pd.read_csv('data/County_Income_Population_2023.csv')
    pci = inc_pop[(inc_pop.County == county) & (inc_pop.State_Abbreviation == state)]
    inc = int(pci.Income.values[0])
    pop = int(pci.Population.values[0])
    row = pd.DataFrame(row).T
    row['population'] = pop
    row['per_capita_income'] = inc
    prediction = model.predict(row)
    changed_row = row.copy(deep=True)
    changed_row['population'] = pop+pop_change
    changed_row['per_capita_income'] = inc + pci_change
    changed_row['closest_five_public_avg'] = changed_row['closest_five_public_avg'] + public_change
    changed_row['closest_five_private_nfp_avg'] = changed_row['closest_five_private_nfp_avg'] + private_change
    new_prediction = model.predict(changed_row)
    return new_prediction-prediction