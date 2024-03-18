import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as pltly

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_prognostic = fetch_ucirepo(id=16) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_prognostic.data.features 
y = breast_cancer_wisconsin_prognostic.data.targets 

# metadata 
source_metadata = breast_cancer_wisconsin_prognostic.metadata
print(source_metadata) 
  
# variable information 
list_variables = breast_cancer_wisconsin_prognostic.variables
print(list_variables) 

print("X.shape", X.shape)

# Handling Null/NaN values
null_value_variables = np.array([]) # initialize an empty array to store the variables with null value
for col in X:
    if(X[col].isnull().sum() != 0):
        print(col,"has null value count ", X[col].isnull().sum())
        X = X.fillna(X[col].mean()) # fill missing value by taking mean of that column
        
        null_value_variables = np.append(null_value_variables, col)   

print("list of null value columns",null_value_variables)

# After null / nan value treatment
for null_value_variable in null_value_variables:
    print(null_value_variable,"has null value count ",X[null_value_variable].isnull().sum())