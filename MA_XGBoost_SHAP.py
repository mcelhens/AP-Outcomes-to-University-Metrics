'''
Use XGBoost to train the data
Use SHAP to show feature importance
'''

import numpy as np
import pandas as pd

import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pickle

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import plotly.graph_objects as go

shap.initjs()



MA_AP_all_data = pd.read_csv('data/Massachusetts/AP_data_combined_18_22.csv')
data_MA_inference = MA_AP_all_data.drop(columns=['COUNTY', 'District Code', 'Year', 'Tests Taken'])    ## drop unnecessary columns




## Split the full data set into the train set and the test set

outcome  = data_MA_inference.columns[0]       ## use the column '% Score 3-5' as the outcome of the model
features = data_MA_inference.columns[1:]      ## all other columns are the features of the model

X = data_MA_inference[features]
y = data_MA_inference[outcome]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=216)

# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)


## Use XGBoost to train the model
params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 3,            # Example: Tune this parameter
    "learning_rate": 0.2,      # Example: Lower learning rate
    "subsample": 0.8,          # Example: Subsampling for regularization
    "random_state": 216
}

n = 1000

evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    evals=evals,
    verbose_eval=50,            # Print rmse val every 10 rounds
    early_stopping_rounds=100   # XGBoost will automatically stop the training if validation loss doesn't improve for 100 consecutive rounds
)



'''
Make bar plot with SHAP
'''


explainer = shap.Explainer(xgb_model)
shap_values = explainer.shap_values(X)

#### 
fig = shap.summary_plot(
    shap_values, 
    X,
    plot_type='bar', 
    show=False,
    plot_size=[12, 6]
)

# Save the plot
with open('data/MA_pickled/shap_summary_plot_bar.pkl', 'wb') as f:
    pickle.dump(fig, f)
    
# Save the plot to a file
plt.savefig('data/Massachusetts/plot/shap_summary_plot_bar.png', bbox_inches='tight')
plt.show()
plt.close()





'''
Make the density scatter plot with SHAP
'''

explainer = shap.Explainer(xgb_model)
shap_values = explainer.shap_values(X)


fig = shap.summary_plot(
    shap_values, 
    X, 
    show=False, 
    plot_size=[12, 6])

# Save the plot
with open('data/MA_pickled/shap_summary_plot_density_scatter.pkl', 'wb') as f:
    pickle.dump(fig, f)
    
plt.savefig('data/Massachusetts/plot/shap_summary_plot_density_scatter.png', bbox_inches='tight')
plt.show()
plt.close()

