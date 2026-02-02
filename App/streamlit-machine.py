# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:50:45 2024

@author: karthik
"""


import numpy as np
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#Structuring imports

common_path='.'

def run_analytics():

  df = pd.read_csv('{}/Placement_Data_Full_Class.csv'.format(common_path))

  df.head(2)
  #Global dataframe used for all the analytical purposes
  

  plt.figure(figsize=(8,6))
  gender_counts = df['gender'].value_counts()
  
  # Plot the pie chart
  fig = px.pie(
    names=gender_counts.index, 
    values=gender_counts.values,
    labels=gender_counts.index,
    hole=0.3,  # Hole size (0-1)
    color_discrete_sequence=['limegreen', 'lightcoral'],  # Color sequence for each category
    width=370,
    height=370
  )
  
  spec_counts = df['specialisation'].value_counts()
  
  fig1 = px.pie(
    names=spec_counts.index, 
    values=spec_counts.values,
    labels=spec_counts.index,
    hole=0.3,  # Hole size (0-1)
    color_discrete_sequence=['teal', 'skyblue'],  # Color sequence for each category
    width=370,
    height=370
  )
  

  # Show the figure using Streamlit
  #st.plotly_chart(fig)
  
  
  # Plot settings
  plt.figure(figsize=(10, 8))
  
  
  
  
  
  
  col1, col_space, col2 = st.columns([3, 0.2, 3])

  # Display the plots
  with col1:
      st.subheader("Gender map")
      st.plotly_chart(fig)
  
  with col2:
     # Analytics part
     ax = sns.countplot(data=df, x='gender', hue='status', palette=['skyblue', 'green'])
     for p in ax.patches:
         ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                     textcoords='offset points')
     
     st.subheader('Placed Vs Not-Placed')
     plt.xlabel('Gender')
     plt.ylabel('Count')
     st.set_option('deprecation.showPyplotGlobalUse', False)
     # Display the plot using Streamlit
     st.pyplot()
      
  col3, col_space, col4 = st.columns([3,0.2,3])
  with col3:
    st.subheader('specialized ratio')
    st.plotly_chart(fig1)
  with col4:
   #--plot for placement specialization--
   
   # Analytics part
   ax = sns.countplot(data=df, x='specialisation', hue='status', palette=['skyblue', 'green'])
   for p in ax.patches:
       ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                   textcoords='offset points')
   
   st.subheader('Popular Major')
   plt.xlabel('Major')
   plt.ylabel('Count')
   st.set_option('deprecation.showPyplotGlobalUse', False)
   st.pyplot()
  

#This is a test comment to check the status of git
def enhance_input(gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,specialisation,mba_p):
  #Handling Gender 
  if gender=="M":
    gender=1
  else:
    gender=0
  
  #Handling ssc_b
  if ssc_b=="Central":
    ssc_b=0
  else:
    ssc_b=1
  #handling hscb
  if hsc_b=="Central":
    hsc_b=0
  else:
    hsc_b=1
  
  #Handling hsc_s
  if hsc_s=="science":
    hsc_s=2
  elif hsc_s=="arts":
    hsc_s=0
  else:
    hsc_s=1
  
  #Handling degree_t
  if degree_t=="tech":
    degree_t=2
  elif degree_t=="mgmt":
    degree_t=0
  else:
    degree_t=1
  
  #Handling workex
  if workex=="Yes":
    workex=1
  else:
    workex=0
  
  #Handling specialization
  if specialisation=="HR":
    specialisation=1
  else:
    specialisation=0
    
  return predict_data((gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,specialisation,mba_p))
  
  
def rerun_model():
  #This is where the model's re-learning phase has to be triggered. 
  #Original analysis notebook , here we keep only the model building part
  
  print('This block of cell is used to retrigger the modelling ipynb file')
  print('Running....')
  df = pd.read_csv('{}/Placement_Data_Full_Class.csv'.format(common_path))
  #importing my dataframe to rerun my model
  df.drop(columns='sl_no', inplace=True)
  # dropping serial number column
  df['Status'] = df['status'].replace({'Placed': 1, 'Not Placed' : 0})
  #Salary column is dropped so no need to operate , only status is encoded
  le = LabelEncoder()
  df['gender'] = le.fit_transform(df['gender'])
  df['ssc_b'] = le.fit_transform(df['ssc_b'])
  df['hsc_b'] = le.fit_transform(df['hsc_b'])
  df['hsc_s'] = le.fit_transform(df['hsc_s'])
  df['degree_t'] = le.fit_transform(df['degree_t'])
  df['workex'] = le.fit_transform(df['workex'])
  df['specialisation'] = le.fit_transform(df['specialisation'])
  #Operating on the label encoder to encode categorical variables
  # Dropping Salary column as the students who did not get placed have the salary value as Null. 
  #This will create bias while model building as it is representing similar information as the Target variable 'status'
  new_df = df.drop(['salary','status'], axis=1)
  X = new_df.iloc[:,:-1]
  y = new_df.iloc[:,-1]
  x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=101)
  log_it = LogisticRegression(random_state=32) #constructor invocation happens here and a random_state of 32 is set
  log_it.fit(x_train,y_train)
  y_pred_train = log_it.predict(x_train)
  y_pred_test = log_it.predict(x_test)
  # Specifying the parameters that we want to Hypertune

  parameters = {'penalty': ['l1','l2','elasticnet'], 'C': [1,2,3,5,10,20,30,50], 'max_iter': [100,200,300]}
  #At this step we use GridSearchCV to extract the best parameters which suits Logistic regression
  log_it_grid = GridSearchCV(log_it, param_grid=parameters, scoring = 'accuracy', cv=10)
  log_it_grid.fit(x_train,y_train)
  y_pred_grid_test = log_it_grid.predict(x_test)
  y_pred_grid_train = log_it_grid.predict(x_train)
  pickle.dump(log_it_grid,open('{}/trained_model.sav'.format(common_path),'wb'))
  return "Model retrained and saved successfully with rerun analytics"
  
  
  

  
def predict_data(input_data):

  loaded_model = pickle.load(open('{}/trained_model.sav'.format(common_path),'rb'))
  
  input_numpy = np.asarray(input_data)
  
  input_numpy = input_numpy.reshape(1,-1)
  
  prediction = loaded_model.predict(input_numpy)
  
  print(prediction)
  
  if prediction[0]==0:
    
    return "You will not be placed"
  else:
    return "You will be placed"
  
#feed = (1,65.00,1,25,1,1,60,2,0,57.00,0,59.00)
#Streamlit code starts (main)

#gender	ssc_p	ssc_b	hsc_p	hsc_b	hsc_s	degree_p	degree_t	
#workex	etest_p	specialisation	mba_p

# Function save_data
def save_data(gen, sscpercentage, sscboard, hscpercentage, hscboard, hscsubject, degreepercentage, degreesubject, workexp, etestpercentage, spec, mbapercentage, status):
    # Open existing CSV file in pandas as a DataFrame
    df = pd.read_csv('{}/Placement_Data_Full_Class.csv'.format(common_path))

    # Append a new entry to the DataFrame
    new_entry = {'sl_no':[5],'gender':[gen],'ssc_p':[sscpercentage],'ssc_b':[sscboard],'hsc_p':[hscpercentage],'hsc_b':[hscboard],'hsc_s':[hscsubject],'degree_p':[degreepercentage],'degree_t':[degreesubject],'workex':[workexp],'etest_p':[etestpercentage],'specialisation':[spec],'mba_p':[mbapercentage],'status':[status],'salary':[0]}
    new_entry_df = pd.DataFrame(new_entry)
    df = pd.concat([df, new_entry_df])

    # Save the DataFrame as a new CSV file
    df.to_csv('{}/Placement_Data_Full_Class.csv'.format(common_path),index=False)
    
    return "Data Saved"

def main():
    st.title("Campus Placement Prediction Software")
    st.image('{}/image-1.jpg'.format(common_path),use_column_width=True)
    
    st.subheader("Want to know where you stand?? ")
    gender = st.selectbox("Please choose your gender", ["M", "F"], index=0)
    ssc_p=value = st.slider("Enter your secondary school percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
    ssc_b=st.selectbox("Choose your secondary school board",["Central","Others"],index=1)
    #HSC details
    hsc_p=st.slider("Enter your high school percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
    hsc_b=st.selectbox("Choose your high school board",["Central","Others"],index=1)
    hsc_s=st.selectbox("Choose your high school subject",["science","commerce","arts"],index=1)
    #Degree details
    degree_p=st.slider("Enter your degree percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
    degree_t=st.selectbox("Enter your degree subject",["tech","mgmt","others"],index=1)
    #Work ex
    workex=st.selectbox("Do you have work experience?",["Yes","No"],index=1)
    #entrance test percentage
    etest_p=st.slider("Enter your entrance test percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
    #Specialization
    specialisation=st.selectbox("What is your specialisation",["HR","finance"],index=1)
    #mba percentage
    mba_p=st.slider("Enter your MBA percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
    
 
    output=""
    if st.button("Rate your chances!"):
       output=enhance_input(gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,specialisation,mba_p)
    st.success(output)
    
    
    st.subheader("Contribute to our software!!")
    st.image('{}/image-2.png'.format(common_path),use_column_width=True)
    #Lakshmi's part
    gen = st.selectbox("Please choose your gender", ["M","F"], index=0, key="gender_selectbox")

    # Senior Scondary School Details
    sscpercentage = st.slider("Enter your secondary school percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f", key="sscp_slider")
    sscboard = st.selectbox("Choose your secondary school board", ["Central","Others"], index=1, key="ssc_selectbox")
    
    # High school details
    hscpercentage = st.slider("Enter your high school percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f", key="hscp_selectbox")
    hscboard = st.selectbox("Choose your high school board", ["Central","Others"], index=1, key="hscbtselectbox")
    hscsubject = st.selectbox("Choose your high school subject", ["Science","Commerce","Arts"], index=1, key="hscsub_selectbox")
    
    # Degree details
    degreepercentage = st.slider("Enter your degree percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f", key="degreep_slider")
    degreesubjects = st.selectbox("Enter your degree subject", ["Sci&Tech","Comm&Mgmt","Others"], index=1, key="degreesub-selectbox")
    
    # Work Experience
    workexp = st.selectbox("Do you have work experience?", ["Yes","No"], index=1, key="workexp_selectbox")
    
    # Entrance test percentage
    etestpercentage = st.slider("Enter your entrance test percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f", key="etestp_slider")
    
    # Specialization
    spec = st.selectbox("What is your specialisation", ["Mkt&HR","Mkt&Fin"], index=1, key="spec_selectbox")
    
    # MBA percentage
    mbapercentage = st.slider("Enter your MBA percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f", key="mbap_slider")
    
    # Details
    status = st.selectbox("Were you placed?", ["Placed","Not Placed"], index=1, key="status_selectbox") 
    
    op = ""
    if st.button("Save Data !"):
        op = save_data(gen, sscpercentage, sscboard, hscpercentage, hscboard, hscsubject, degreepercentage, degreesubjects, workexp, etestpercentage, spec, mbapercentage, status)
        rerun_model()
    st.success("Successfully saved the data and retrained the model")
    
    
    if(st.checkbox("Run Analytics")):
      run_analytics()
      
    st.image('{}/image-3.jpg'.format(common_path),use_column_width=True)
      
      

if __name__ == '__main__':
    main()

  
  




