import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
st.title("""
Tree Shade predictor for Farms
""")
from PIL import Image
image = Image.open('img.jpg')

st.image(image, caption='Tree shade')

model = load('NDVI_predict.joblib')

def predict_plant_health(temperature):
    test=pd.Series(np.array([temperature]))
    NDVI = model.predict(np.array(test).reshape(-1,1))
    alert_message = "No recommendation"
    if 0.2 >= NDVI <= 0.4:
        alert_message = "sparse vegetation - Rigoursly Increase the shade in the farm"
    elif 0.4 > NDVI <= 0.6:
        alert_message = "moderate vegetation - Increase the shade in the farm"
    elif NDVI > 0.6:
        alert_message = "Highest density of Green leaves! reap your benefits"
    return alert_message, NDVI[0]

#print(predict_plant_health(79.0))

with st.form("my_form"):
    st.write("Lets Make it Green")
    number = st.number_input('Enter the temperature (Or Pull from weather)')
    message = predict_plant_health(number)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write(message)

st.write("Made with love by Aditi and Megha")
