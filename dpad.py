import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as pltly
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import joblib
import io
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#---------genearal EDA
import pygwalker as pyg

print("\n\n\n\nStarting -------->>>>>>>>>>")
load_data_once = 0
if(load_data_once == 0):
    print(f'{load_data_once}: EDA Run STARTED--->>>>>')  
    # step 1: fetch dataset 
    breast_cancer_wisconsin_prognostic = fetch_ucirepo(id=16) 
    
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_prognostic.data.features 
    y = breast_cancer_wisconsin_prognostic.data.targets 
    
    df = pd.concat([X, y], axis=1)
    df["volume"] = (4/3) * 3.14 * df["radius1"] * df["radius2"] * df["radius3"]
    
    # step 2: Handling Null/NaN values replacing with mean value
    for col in X:
        if(X[col].isnull().sum() != 0):
            X = X.fillna(X[col].mean()) # fill missing value by taking mean of that column
            
            
    # transforming categorical data N and R to integers 0s and 1s
    label_encoder = LabelEncoder()
    # Fit label encoder and transform target variable
    #print(y)
    y_dash = np.ravel(y["Outcome"])
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

# Tabs for separate data display
tab1, tab2, tab3 = st.tabs(["Predictions", "Confusion Matrix and scores", "Data Exploration"])

with tab1:
    col1, col2, col3 = st.columns([2, 1, 2])

    data_selector = st.slider("Pick a test data to label the data", 1, X_test.shape[0], value=9, help="slide to select stored test data for a prediction" )

    #if(slider logic) based on majority score from all the y_predicts display image "no cancer" or "reappear cancer"

    y_predict_scores = np.array([])
    for model_name in model_names:
        # loading the model and run for scores - score calculation
        model_file_name = model_name+".joblib"
        print("Loading Model: ", model_file_name)
        grid_search_model = joblib.load(model_file_name)
        data_point = X_test[data_selector-1]
        print("X_test[data_selector-1]", data_point)
        # Making predictions using test data
        y_pred = int(grid_search_model.predict([data_point]))
        print(f'data_selector: {data_selector}, y_pred :{y_pred}, model_name :{model_name}')
        
        y_predict_scores=np.append(y_predict_scores, y_pred)

    print(f'\n\n\ny_predict_scores : {y_predict_scores}\n\n\n')

    unique, frequency = np.unique(y_predict_scores, return_counts = True)
    count_array = np.asarray((unique, frequency)).T
    print(f'count_array :\n {count_array} {count_array.shape} unique :{unique} frequency : {frequency}')
    
    with col1:
        for score, model in zip(y_predict_scores, model_names): 
            st.write(f" {model} : {score}")
        #st.write(f"Votes {count_array}")
        for count in count_array:
            if(count[1] <= 1):
                st.write(f"{int(count[1])} Vote for {int(count[0])}")
            else:
                st.write(f"{int(count[1])} Votes for {int(count[0])}")

    # When model produce mixed outcome results
    if(count_array.shape[0] == 2):
        if(count_array[0][1] >= count_array[1][1]):
            print("if-if -> low possibility")
            #col3.write("low possibility")
            col3.success("low probability of disease - nonrecur")
            col2.image(normal_human_breast_tissues)
            
        else:
            print("if-else -> very high possibility")
            #col3.write("very high possibility")
            col3.error("high probability of disease - recur")
            col2.image(cancer_human_breast_tissues)

    # When all models predict / label the same class
    else:
        if(count_array[0][0] == 0):
            print("else-if -> low possibility")
            #col3.write("low possibility")
            col3.success("low probability of disease - nonrecur")
            col2.image(normal_human_breast_tissues)
            
        else:
            print("else-else -> very high possibility")
            #col3.write("very high possibility")
            col3.error("high probability of disease - recur")
            col2.image(cancer_human_breast_tissues)
   
print("Working Successfull >>>>\n")
print("\n\n------>>>>>>>>>>>>>> Code runs ok to this point\n\n\n\n")


# Confusion matrix and other scores of each model
with tab2:
    sample_range_for_confusion_matrix = st.slider("Select a range for the Machine Learning Models.", 1, X.shape[0], value=(21, int(X.shape[0]/2)), help="select stored data range to display confusion matrix" )
    col4, col5, col6 = st.columns([6, 1, 2])

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

# Data science EDA uing plotly
with tab3:
    
    buffer = io.StringIO()
    sys.stdout = buffer
    df.info()
    sys.stdout = sys.__stdout__
    result = buffer.getvalue()
    st.write("DataFrame  Info:")
    st.code(result, language='text')
    st.write("Data source: https://www.archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic")

    fig0 = px.box(df, 
             y="Outcome", 
             x='Time', 
             log_x=False, 
             points='all', 
             notched=True,
             color='Outcome',
             labels={'Outcome':'N as no-recur and R as recur', 'volume': 'Tumor size or volume'}, 
             title = 'Breast cancer prognostic - no-recurrence, recurrences VS Time', 
             hover_name='Outcome')

    st.plotly_chart(fig0, use_container_width=True)
    
    fig1 = px.box(df, 
             y="Outcome", 
             x='volume', 
             log_x=True, 
             points='all', 
             notched=True,
             color='Outcome',
             labels={'Outcome':'N as no-recur and R as recur', 'volume': 'Tumor size or volume'}, 
             title = 'Breast cancer prognostic - no-recurrence, recurrences and lump size', 
             hover_name='Outcome')

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(df, 
             y="Outcome", 
             x='tumor_size', 
             log_x=True, 
             points='all', 
             notched=True,
             color='Outcome',
             labels={'Outcome':'N as no-recur and R as recur', 'volume': 'Tumor size or volume'}, 
             title = 'Breast cancer prognostic - no-recurrence, recurrences and lump size', 
             hover_name='Outcome')

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(df, 
             y="Outcome", 
             x='lymph_node_status', 
             #log_x=True, 
             points='all', 
             notched=True,
             color='Outcome',
             labels={'Outcome':'N as no-recur and R as recur', 'volume': 'Tumor size or volume'}, 
             title = 'Breast cancer prognostic - no-recurrence, recurrences and lump size', 
             hover_name='Outcome')

    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.violin(df, 
                 y="volume", 
                 points='all', 
                 box=True, 
                 color='Outcome',
                 labels={'Outcome':'N as no-recur and R as recur', 'volume': 'Tumor size'}, 
                 title = 'Breast cancer prognostic - no-recurrence, recurrences and lump size', 
                 hover_name='Outcome')
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.violin(df, 
                 y="Time", 
                 points='all', 
                 box=True, 
                 color='Outcome',
                 labels={'Outcome':'N as no-recur and R as recur', 'volume': 'Tumor size'}, 
                 title = 'Breast cancer prognostic - no-recurrence, recurrences and Time', 
                 hover_name='Outcome')
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.ecdf(df, x="volume", color="Outcome", log_x=True,
               labels={'Outcome':'N as no-recur and R as recur', 'volume': 'Tumor size'}, 
                title = 'Breast cancer prognostic - no-recurrence, recurrences and lump size', 
                hover_name='Outcome',
                markers = False,
                lines = True,
                marginal="rug")
    st.plotly_chart(fig6, use_container_width=True)

    fig7 = px.histogram(df, x="volume", color="Outcome", log_x=False,
               labels={'Outcome':'N as no-recur and R as recur', 'volume': 'Tumor size'}, 
                title = 'Breast cancer prognostic - no-recurrence, recurrences and lump size', 
                hover_name='Outcome',                        
                marginal="box")          
    fig7.update_layout(bargap=0.1)
    fig7.update_layout(
        title='Cancer size VS Label/Outcome',
        yaxis_title='Count',
        xaxis_title='Tumor size')
    
    st.plotly_chart(fig7, use_container_width=True)

    fig8 = px.scatter(df, x="Time", y="tumor_size", color="Outcome", size="tumor_size")
    fig8.update_layout(
        title='Cancer Time VS Tumor size - Label/Outcome',
        xaxis_title='Time',
        yaxis_title='Tumor size',
    )
    st.plotly_chart(fig8, use_container_width=True)

    fig9 = px.scatter(df, x="lymph_node_status", y="tumor_size", color="Outcome", size="Time")
    fig9.update_layout(
            title='Size is Time - Lymph Node status VS Tumor size',
            xaxis_title='Lymph Node status',
            yaxis_title='Tumor size')
    st.plotly_chart(fig9, use_container_width=True)

    for looper in range(1, 4, 1):
        plot = "radius"+str(looper)
        print(plot)
        fig10 = px.scatter(df, x="lymph_node_status", y=plot, color="Outcome", size="Time")
        fig10.update_layout(
            title='Size is Time - Lymph Node VS Radius'+str(looper),
            xaxis_title='Lymph Node Status',
            yaxis_title='Radius'+str(looper))
        st.plotly_chart(fig10, use_container_width=True)

   

    # Create a figure with subplots
    fig11 = make_subplots(rows=3, cols=1)

    # Add traces to the subplots
    fig11.add_trace(go.Scatter(x=df['Time'], y=df['radius1'], mode='markers'), row=1, col=1)
    fig11.add_trace(go.Scatter(x=df['Time'], y=df['radius2'], mode='markers'), row=2, col=1)
    fig11.add_trace(go.Scatter(x=df['Time'], y=df['radius3'], mode='markers'), row=3, col=1)

    # Update layout
    fig11.update_layout(title_text="Subplots Example", showlegend=False)

    fig11.update_layout(
        title='Time VS Radius',
        xaxis_title='Time',
        yaxis_title='Radius',
    )

    # Show the plot
    st.plotly_chart(fig11, use_container_width=True)

