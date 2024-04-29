import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the predictor model from a pickle file
model = pickle.load(open('LRmodel_V1.pkl', 'rb'))

# Load the encoder dictionary from a pickle file
with open('//Users/lanceroyston/Downloads/MSBA/Spring 2024/SS ML Deployment/Project 2 CHLA STREAMLIT CLOUD DEPLOY/encoder_V3.pkl', 'rb') as pkl_file:
    encoder_dict = pickle.load(pkl_file)


def encode_features(df, encoder_dict):
    # For each categorical feature, apply the encoding
    category_col = ['CLINIC', 'IS_REPEAT']
    for col in category_col:
        if col in encoder_dict:
            le = LabelEncoder()
            le.classes_ = np.array(encoder_dict[col], dtype=object)  # Load the encoder classes for this column

            # Handle unknown categories by using 'transform' method and a lambda function
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            df[col] = le.transform(df[col])
    return df


def main():
    st.title("NO-SHOW PREDICTIOR")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">No-Show Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    clinic = st.selectbox("Clinic", ["Arcadia Care Center","Bakersfield Care Clinic","Encino Care Center","Santa Monica Clinic","South Bay Care Center","Valenica Care Center"])
    is_repeat = st.selectbox("Is this a repeat appointment?",["Y","N"])
    lead_time = st.text_input("Lead Time","0")
    appt_num = st.text_input("Appointment Number","0")
    total_cancellations = st.text_input("Total number of Cancellations","0")
    total_rescheduled = st.text_input("Total number of Rescheduled","0")
    total_success = st.text_input("Total number of Successful Appointments","0")
    total_noshow = st.text_input("Total number of previous No-Shows","0")
    age = st.text_input("Age","0")


    if st.button("Predict"):

        data = {'CLINIC': clinic, 'IS_REPEAT': is_repeat, 'LEAD_TIME': int(lead_time), 'APPT_NUM': int(appt_num), 'TOTAL_NUMBER_OF_CANCELLATIONS': int(total_cancellations), 'TOTAL_NUMBER_OF_RESCHEDULED': int(total_rescheduled), 'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT': int(total_success), 'TOTAL_NUMBER_OF_NOSHOW': int(total_noshow), 'AGE': int(age)}
        # print(data)
        # Convert the data into a DataFrame for easier manipulation
        df = pd.DataFrame([data])

        # Encode the categorical columns
        df = encode_features(df, encoder_dict)

        # Now, all your features should be numerical, and you can attempt prediction
        features_list = df.values
        prediction = model.predict(features_list)

        output = int(prediction[0])
        if output == 0:
            text = "Show"
        else:
            text = "No-Show"

        st.success('Patient will {}'.format(text))

if __name__=='__main__':
    main()
