#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 15:51:39 2022

@author: angelahu
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.title("My first dashboard    !!!!!!")

url=r"https://raw.githubusercontent.com/JonathanBechtel/dat-11-15/main/ClassMaterial/Unit1/data/master.csv"

num_rows = st.sidebar.number_input("select Number of Rows to Load", min_value=1000, 
                                   max_value=50000, 
                                   step = 1000)

#print(num_rows)


section = st.sidebar.radio("Choose Application Section", ["Data Explorer", "Model Explorer"] )

print(section)

@st.cache  # function call once, it will nerver call the function again. 
def load_data(num_rows):

    df= pd.read_csv(url, parse_dates=['visit_date'], nrows=num_rows)
    
    return df

#df = load_data(num_rows)

@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean() 
    return grouping
 
def load_model():
    with open('pipe.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)       # this will import the pickled object 
    return model
 

df = load_data(num_rows)
    
if section == "Data Explorer":
        
    x_axis = st.sidebar.selectbox("Choose column for X-axis", 
                                  df.select_dtypes(include = np.object).columns.tolist())
    
    y_axis = st.sidebar.selectbox("Choose column for Y-axis", ['visitors', 'reserve_visitors'])
    
    
    #grouping = df.groupby(x_axis)[y_axis].mean()     
    chart_type = st.sidebar.selectbox("Choose  your chart type", ['line', 'bar', 'area'])
    
    
    if chart_type == "line":
        grouping = create_grouping(x_axis, y_axis)
        st.line_chart(grouping) 
        
    elif chart_type == "bar":
        grouping = create_grouping(x_axis, y_axis)
        st.bar_chart(grouping)
        
    elif chart_type == "area":
        fig=px.strip(df[[x_axis, y_axis]], x=x_axis, y=y_axis)
        
        st.plotly_chart(fig)

    st.write(df)
    
else:
    st.text(" Choose Options to the Side to Explore the Model")
    
    model = load_model()
    
    id_val = st.sidebar.selectbox("Chose Restaurant ID",
                                  df['id'].unique().tolist())
    yesterday = st.sidebar.number_input('How many visitors yesterday', min_value = 0, 
                                        max_value = 100, step = 1, value = 20)
    day_of_week = st.sidebar.selectbox('Day of week', df['day_of_week'].unique().tolist())
    
    sample = {
    'id': id_val,
    'yesterday': yesterday,
    'day_of_week': day_of_week}

    sample = pd.DataFrame(sample, index = [0])
    
    prediction = model.predict(sample)[0]
    
    # print(prediction)
    
    st.title(f"Predicted Attendance: {int(prediction)}")
    
    
    
    


#st.write(df)

