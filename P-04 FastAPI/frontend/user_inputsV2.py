### import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

image_file_path = '/Users/lanceroyston/Downloads/MSBA/Spring 2024/SS ML Deployment/CHLA No-Show Deployment Projects/P-04 FastAPI/backend/LOGOV2.jpg'

# Load the predictor model from a pickle file
model = pickle.load(open('/Users/lanceroyston/Downloads/MSBA/Spring 2024/SS ML Deployment/CHLA No-Show Deployment Projects/P-04 FastAPI/backend/model_V2.pkl', 'rb'))





st.image(image_file_path, use_column_width=True)

# Set up the page layout with titles and logo
col1, col2, col3 = st.columns([1,1,1])
    
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-family: Verdana; color: #FFFFFF;'>No-Show Predictor</h3>", unsafe_allow_html=True)


# Function to load data with caching to enhance performance
@st.cache_resource

def load_data(file_path):
    return pd.read_csv(file_path)

# Load data
df = load_data("/Users/lanceroyston/Downloads/MSBA/Spring 2024/SS ML Deployment/CHLA No-Show Deployment Projects/P-04 FastAPI/backend/CHLA_clean_data_2024_Appointments.csv")
df['APPT_DATE'] = pd.to_datetime(df['APPT_DATE'])

    
# Date input for filtering appointments
x, y = st.columns([1,1])

with x:
    start_datetime = st.date_input("Start Date", min_value=df['APPT_DATE'].min(), max_value=df['APPT_DATE'].max())
with y:
    end_datetime = st.date_input("End Date", min_value=df['APPT_DATE'].min(), max_value=df['APPT_DATE'].max())

start_datetime = pd.to_datetime(start_datetime)
end_datetime = pd.to_datetime(end_datetime)

if start_datetime > end_datetime:
    st.error("End Date should be after Start Date")

    
# Filter DataFrame by the selected date range
if start_datetime and end_datetime:
    mask = (df['APPT_DATE'] >= start_datetime) & (df['APPT_DATE'] <= end_datetime)
    filtered_df = df[mask]
    start_date = start_datetime.date()
    end_date = end_datetime.date()
else:
    st.warning("Please select both start and end dates")   


# Clinic selection and filtering
clinic_selector = st.multiselect("CLINIC", df['CLINIC'].unique())
if len(clinic_selector) == 0: 
    filtered_df = filtered_df.copy()  
    filtered_df = filtered_df[filtered_df['CLINIC'].isin(clinic_selector)]
clinic_strings = []

for i in range(len(clinic_selector)):
    clinic_strings.append(str(clinic_selector[i]).title())



# Prepare data for prediction
fdf = filtered_df[[
    'MRN',
    'APPT_DATE',
    'AGE',
    'CLINIC',
    'TOTAL_NUMBER_OF_CANCELLATIONS',
    'LEAD_TIME',
    'TOTAL_NUMBER_OF_RESCHEDULED',
    'TOTAL_NUMBER_OF_NOSHOW',
    'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT',
    'HOUR_OF_DAY',
    'NUM_OF_MONTH'
]]


pdf = fdf.drop(['MRN', 'APPT_DATE'], axis=1)


# Label encoding for the 'CLINIC' column
le = LabelEncoder()
object_cols = ['CLINIC']
for col in object_cols:
    pdf[col] = le.fit_transform(pdf[col])


# Button to run the prediction model
run_button = st.button('Run')

if run_button:
    
    if pdf.shape[0] == 0:
        st.warning("No data available for the selected date range and clinic(s)")
        st.stop()
      
    predictions = model.predict(pdf)
    predictions_series = pd.Series(predictions)
    fdf = fdf.reset_index(drop=True)
    final_df = pd.concat([fdf, predictions_series], axis=1)
    final_df.columns = [*final_df.columns[:-1], 'NO SHOW (Y/N)']
    final_df = final_df[['MRN', 'APPT_DATE', 'CLINIC', 'NO SHOW (Y/N)']]
    no_show_mapping = {0: 'NO', 1: 'YES'}
    final_df['NO SHOW (Y/N)'] = final_df['NO SHOW (Y/N)'].replace(no_show_mapping)
    final_df['MRN'] = final_df['MRN'].astype(str)
    final_df = final_df.sort_values(by='CLINIC')
    final_df = final_df.sort_values(by='APPT_DATE')
    final_df.rename(columns={'APPT_DATE': 'APPOINTMENT DATE'}, inplace=True)
    st.write(final_df)





    