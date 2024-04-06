import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as pltly
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import joblib

print("\n\n\n\nStarting -------->>>>>>>>>>")
load_data_once = 0
if(load_data_once == 0):
    print(f'{load_data_once}: EDA Run STARTED--->>>>>')  
    # step 1: fetch dataset 
    breast_cancer_wisconsin_prognostic = fetch_ucirepo(id=16) 
    
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_prognostic.data.features 
    y = breast_cancer_wisconsin_prognostic.data.targets 

    # step 2: Handling Null/NaN values replacing with mean value
    for col in X:
        if(X[col].isnull().sum() != 0):
            X = X.fillna(X[col].mean()) # fill missing value by taking mean of that column
            
            
    # transforming categorical data N and R to integers 0s and 1s
    label_encoder = LabelEncoder()
    # Fit label encoder and transform target variable
    #print(y)
    y_dash = np.ravel(y)
    #print(y_dash)
    #Label Encoding the features (N as 0,R as 1)
    y_encoded = label_encoder.fit_transform(y_dash)
    #print(y_encoded)

    # step 3: preparing data for model prediction
    # splitting the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=0)

    #Feature scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    

#  load data only once into memory
load_data_once = 1
print(f'{load_data_once}: EDA Run completed successfully')

#models = [LogisticRegression(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]
model_names =['Logistic_Regression', 'SVM', 'Decision_Tree_Classifier', 'Random_Forest_Classifier', 'KNN']

# View of Model prediction and a few EDA plots
#prin(f"Data visualisation and playround using streamlit")
st.markdown("<h1 style='text-align: center; color: grey;'>Breast Cancer Prognostic Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>UCI Machine Learning Repository</h3>", unsafe_allow_html=True)

label_image_source = "https://myclgnotes.com/wp-content/uploads/2019/08/Guide-to-Machine-Learning-and-AI-1080x675.jpg"
normal_human_breast_tissues = "https://vitrovivo.com/wordpress/wp-content/uploads/2018/03/HuPS-02001A1-Normal-Breast-10x.jpg"
cancer_human_breast_tissues = "https://st.focusedcollection.com/13422768/i/1800/focused_160229056-stock-photo-metastatic-breast-cancer.jpg"


col1, col2, col3 = st.columns([2, 1, 2])

data_selector = st.slider("Pick a test data to label the data", 1, X_test.shape[0], value=9, help="slide to select stored test data for a prediction" )

#if(slider logic) based on majority score from all the y_predicts display image "no cancer" or "reappear cancer"

y_predict_scores = np.array([])
for model_name in model_names:
    # loading the model and run for scores - score calculation
    model_file_name = model_name+".joblib"
    print("Loading Model: ", model_file_name)
    grid_search_model = joblib.load(model_file_name)
    data_point = [X_test[data_selector]]
    #print("X_test[data_selector]", data_point)
    # Making predictions using test data
    y_pred = int(grid_search_model.predict(data_point))
    print(f'data_selector: {data_selector}, y_pred :{y_pred}, model_name :{model_name}')
    
    y_predict_scores=np.append(y_predict_scores, y_pred)

print(f'\n\n\ny_predict_scores : {y_predict_scores}\n\n\n')

unique, frequency = np.unique(y_predict_scores, return_counts = True)
count_array = np.asarray((unique, frequency)).T
print(f'count_array : {count_array} {count_array.shape} unique :{unique} frequency : {frequency}')

# When model produce mixed outcome results
if(count_array.shape[0] == 2):
    if(count_array[0][1] >= count_array[1][1]):
        print("low possibility")
        col3.write("low possibility")
        col2.image(normal_human_breast_tissues)
    else:
        print("very high possibility")
        col3.write("very high possibility")
        col2.image(cancer_human_breast_tissues)

# When all models predict / label the same class
else:
    if(count_array[0][0] == 0):
        print("low possibility")
        col3.write("low possibility")
        col2.image(normal_human_breast_tissues)
    else:
        print("very high possibility")
        col3.write("very high possibility")
        col2.image(cancer_human_breast_tissues)
print("Working Successfull >>>>\n")
print("\n\n------>>>>>>>>>>>>>> Code runs ok to this point\n\n\n\n")

# Tabs for separate data display
tab1, tab2 = st.tabs(["Confusion Matrix and scores", "Data graphs"])

with tab1:
    sample_range_for_confusion_matrix = st.slider("Select a range for the Machine Learning Models.", 1, X.shape[0], value=(21, int(X.shape[0]/2)), help="select stored data range to display confusion matrix" )
    col4, col5, col6 = st.columns([3, 3, 3])

    # Display EDA data using plotly
    with col4:
        #plot two plots
        st.write("col 4 this is EDA analysis")
    with col5:
        # plot two plots
        st.write("col 5 this is EDA analysis")

    with col6:
        # plot two plots
        st.write("col 6 this is EDA analysis")

with tab2:
    col4, col5, col6 = st.columns([3, 3, 3])

    # Display EDA data using plotly
    with col4:
        #plot two plots
        st.write("col 4 this is EDA analysis")
    with col5:
        # plot two plots
        st.write("col 5 this is EDA analysis")

    with col6:
        # plot two plots
        st.write("col 6 this is EDA analysis")