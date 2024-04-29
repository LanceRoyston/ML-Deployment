# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from titlecase import titlecase
from sklearn.preprocessing import LabelEncoder

# Set up the page layout with titles and logo
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image('LOGOV2.jpg', width=250)
    st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-family: Geneva; color: #FFFFFF;'>NO-SHOW PREDICTOR</h3>", unsafe_allow_html=True)

# Function to load data with caching to enhance performance
@st.cache_resource
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['APPT_DATE'] = pd.to_datetime(df['APPT_DATE'])
    return df

# Load data
df = load_data("CHLA_clean_data_2024_Appointments.csv")

# Date input for filtering appointments
col1, col2 = st.columns([1, 1])
with col1:
    start_datetime = st.date_input("Start Date", min_value=df['APPT_DATE'].min(), max_value=df['APPT_DATE'].max())
with col2:
    end_datetime = st.date_input("End Date", min_value=df['APPT_DATE'].min(), max_value=df['APPT_DATE'].max())

# Ensure the end date is after the start date
if start_datetime > end_datetime:
    st.error("End Date should be after Start Date")

# Filter DataFrame by the selected date range
if start_datetime and end_datetime:
    mask = (df['APPT_DATE'] >= start_datetime) & (df['APPT_DATE'] <= end_datetime)
    filtered_df = df[mask]
    start_date, end_date = start_datetime.date(), end_datetime.date()
    st.caption(f"You have selected appointments between {start_date} and {end_date}")
else:
    st.warning("Please select both start and end dates")

# Clinic selection and filtering
clinic_selector = st.multiselect("CLINIC", df['CLINIC'].unique())
filtered_df = filtered_df[filtered_df['CLINIC'].isin(clinic_selector)] if clinic_selector else filtered_df.copy()

# Prepare data for prediction
fdf = filtered_df[[
    'MRN', 'APPT_DATE', 'AGE', 'CLINIC', 'TOTAL_NUMBER_OF_CANCELLATIONS',
    'LEAD_TIME', 'TOTAL_NUMBER_OF_RESCHEDULED', 'TOTAL_NUMBER_OF_NOSHOW',
    'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT', 'HOUR_OF_DAY', 'NUM_OF_MONTH'
]]
pdf = fdf.drop(['MRN', 'APPT_DATE'], axis=1)

# Load and prepare the prediction model
model = pickle.load(open('model_V1.pkl', 'rb'))
le = LabelEncoder()
for col in ['CLINIC']:
    pdf[col] = le.fit_transform(pdf[col])

# Button to run the prediction model
run_button = st.button('Run')
if run_button:
    predictions = model.predict(pdf)
    predictions_series = pd.Series(predictions)
    fdf = fdf.reset_index(drop=True)
    final_df = pd.concat([fdf, predictions_series], axis=1)
    final_df.columns = [*final_df.columns[:-1], 'NO SHOW (Y/N)']
    final_df['NO SHOW (Y/N)'] = final_df['NO SHOW (Y/N)'].replace({0: 'NO', 1: 'YES'})
    final_df['MRN'] = final_df['MRN'].astype(str)
    final_df = final_df.sort_values(by=['CLINIC', 'APPT_DATE'])
    final_df.rename(columns={'APPT_DATE': 'APPOINTMENT DATE'}, inplace=True)
    st.write(final_df)
