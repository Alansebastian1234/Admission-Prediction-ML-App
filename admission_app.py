# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestRegressor # Random Forest
import mapie
# Import MAPIE to calculate prediction intervals
from mapie.regression import MapieRegressor

# To calculate coverage score
from mapie.metrics import regression_coverage_score

# Package for data partitioning
from sklearn.model_selection import train_test_split

# Package to record time
import time

# Module to save and load Python objects to and from files
import pickle 

# Ignore Deprecation Warnings
import warnings
warnings.filterwarnings('ignore')

st.title('Graduate Admission Predictor') 
st.image('admission.jpg')
st.write("This app uses multiple inputs to predict the probability of admission to grad school")

with open('reg_admission.pickle', 'rb') as reg_pickle:
    reg = pickle.load(reg_pickle)

st.sidebar.header('**Enter your profile details**')

gre = st.sidebar.number_input('GRE Score')
toefl = st.sidebar.number_input('TOEFL Score')
gpa = st.sidebar.number_input('CGPA')

res = st.sidebar.selectbox('Research Experience', options = ['Yes', 'No'])
univ = st.sidebar.slider('University Rating', min_value=1.0, max_value=5.0, step=.1)
sop = st.sidebar.slider('Statement of Purpose (SOP)', min_value=1.0, max_value=5.0, step=.1)
lor = st.sidebar.slider('Letter of Recommendation (LOR)', min_value=1.0, max_value=5.0, step=.1)

default_df = pd.read_csv('Admission_Predict.csv')
default_df = default_df.dropna().reset_index(drop = True) 

encode_df = default_df.copy()
encode_df = encode_df.drop(columns = ['Chance of Admit'])

encode_df.loc[len(encode_df)] = [gre, toefl, univ, sop, lor, gpa, res]

encode_dummy_df = pd.get_dummies(encode_df)

user_encoded_df = encode_dummy_df.tail(1)

alpha = 0.1 # For 90% confidence level

# Use mapie.predict() to get predicted values and intervals
y_test_pred, y_test_pis = reg.predict(user_encoded_df, alpha = alpha)
# Using predict() with new data provided by the user

st.subheader("Predicted Admission Probability")
st.success('**{}**'.format(y_test_pred.round(2))) 
low = float(y_test_pis[:, 0, 0])
high = float(y_test_pis[:, 1, 0])
st.write(f"Confidence Interval: ({low*100:.2f}%, {high*100:.2f}%)")