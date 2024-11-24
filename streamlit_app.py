import streamlit as st
import requests
import pandas as pd

st.title('TensorFlow Model Prediction and Retraining')

# Input for prediction
st.header('Make a Prediction')
input_data = st.text_input('Enter data for prediction (comma-separated)', '1.0,2.0,3.0')
input_data = [float(i) for i in input_data.split(',')]
if st.button('Predict'):
    response = requests.post('http://localhost:8000/predict', json={'data': dict(zip(['feature1', 'feature2', 'feature3'], input_data))})
    if response.status_code == 200:
        st.write('Predictions:', response.json()['predictions'])
    else:
        st.write('Error:', response.text)

# Input for retraining
st.header('Retrain the Model')
retrain_data = st.text_area('Enter data for retraining (each row comma-separated)', '1.0,2.0,3.0\n4.0,5.0,6.0')
retrain_labels = st.text_input('Enter labels for retraining (comma-separated)', '0,1')
retrain_data = [[float(i) for i in row.split(',')] for row in retrain_data.split('\n')]
retrain_labels = [float(i) for i in retrain_labels.split(',')]
if st.button('Retrain'):
    response = requests.post('http://localhost:8000/retrain', json={'data': retrain_data, 'labels': retrain_labels})
    if response.status_code == 200:
        st.write(response.json()['status'])
    else:
        st.write('Error:', response.text)
