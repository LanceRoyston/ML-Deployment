# Adapted by Lance Royston
# This program is used at the back-end FastAPI to receive the user inputs from
# Streamlit-based front-end user_inputs.py program via JSON data structure
# It then loads up the Model_V2.pkl (Prediction ML Model) and label_encoder_V2.pkl
# Invokes the Model_V2 with input data structure to and generates the prediction
# It returns the Prediction to the front-end via FastAPI's @app.post process

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserInput(BaseModel):
    AGE: int
    CLINIC: str
    TOTAL_NUMBER_OF_CANCELLATIONS: int
    LEAD_TIME: int
    TOTAL_NUMBER_OF_RESCHEDULED: int
    TOTAL_NUMBER_OF_NOSHOW: int
    TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT: int
    HOUR_OF_DAY: int
    NUM_OF_MONTH: int
    IS_NO_SHOW: str
    
    

@app.post("/process/")
async def process_item(user_input: UserInput):
    # logic to predict income level as binary based on user inputs

    def encode_features(df, encoder_dict):
    # For each categorical feature, apply the encoding
        category_col = ['CLINIC', 'IS_REPREAT', 'IS_NOSHOW']
        for col in category_col:
            if col in encoder_dict:
                le = LabelEncoder()
                le.classes_ = np.array(encoder_dict[col], dtype=object)  # Load the encoder classes for this column

            # Handle unknown categories by using 'transform' method and a lambda function
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                df[col] = le.transform(df[col])
        return df

    # Load the predictor model from a pickle file
    model = pickle.load(open('model_V2.pkl', 'rb'))
    # Load the encoder dictionary from a pickle file
    with open('label_encoderV2.pkl.pkl', 'rb') as pkl_file:
        encoder_dict = pickle.load(pkl_file)    

        data = {'CLINIC': user_input.CLINIC, 'LEAD_TIME': user_input.LEAD_TIME, 'IS_REPREAT': user_input.IS_REPEAT, 'APPT_NUM': user_input.APPT_NUM, 'TOTAL_NUMBER_OF_CANCELLATIONS': user_input.TOTAL_NUMBER_OF_CANCELLATIONS, 'TOTAL_NUMBER_OF_RESCHEDULED': user_input.TOTAL_NUMBER_OF_RESCHEDULED, 'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT': user_input.TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT, 'TOTAL_NUMBER_OF_NOSHOW': user_input.TOTAL_NUMBER_OF_NOSHOW, 'AGE': user_input.AGE}

        # Convert the data into a DataFrame for easier manipulation
        df = pd.DataFrame([data])
        # Encode the categorical columns
        df = encode_features(df, encoder_dict)
        # Now, all your features should be numerical, and you can attempt prediction
        features_list = df.values
        prediction = model.predict(features_list)
        noshow = int(prediction[0])
    return {"No-Show": noshow}
    
