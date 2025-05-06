
import streamlit as st
import pandas as pd
import joblib

st.title("AI-Powered Intrusion Detection System")

uploaded_file = st.file_uploader("Upload a network traffic CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:", data.head())

    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    data_scaled = scaler.transform(data)
    predictions = model.predict(data_scaled)
    data['Anomaly'] = ['Yes' if pred == -1 else 'No' for pred in predictions]

    st.write("Detection Results:")
    st.dataframe(data)
    st.download_button("Download Results", data.to_csv(index=False), "detection_results.csv")
