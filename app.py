import streamlit as st
from firestore_client import FirestoreClient
from prediction import LandslidePredictor
import os
from dotenv import load_dotenv

load_dotenv()
firestore = FirestoreClient()
predictor = LandslidePredictor()

API_KEY = os.getenv("API_KEY")
st.title("Landslide Prediction Dashboard")

st.sidebar.header("Location Settings")
latitude = st.sidebar.number_input("Enter Latitude", value=12.9716, format="%.6f")  # Default: Bengaluru
longitude = st.sidebar.number_input("Enter Longitude", value=77.5946, format="%.6f")  # Default: Bengaluru

st.header("Recent Sensor Data")
sensor_data = firestore.get_recent_sensor_data()

if sensor_data.empty:
    st.write("No recent sensor data available.")
else:
    sensor_data = firestore.append_weather_data(sensor_data, latitude, longitude, API_KEY)

    # Ensure numeric conversion
    sensor_data[['Temperature', 'Humidity', 'SoilMoisture', 'elevation', 'rainfall']] = \
        sensor_data[['Temperature', 'Humidity', 'SoilMoisture', 'elevation', 'rainfall']].astype(float)

    st.write("Sensor Data with Weather Information:")
    st.dataframe(sensor_data)

    st.header("Predictions")
    predictions = []

    for _, row in sensor_data.iterrows():
        prediction = predictor.predict(
            Temperature=row['Temperature'],
            Humidity=row['Humidity'],
            SoilMoisture=row['SoilMoisture'],
            elevation=row['elevation'],
            rainfall=row['rainfall']  # Now included
        )
        predictions.append(prediction)

    # Store predictions properly
    for i in range(len(sensor_data)):
        sensor_data.loc[sensor_data.index[i], 'predicted_label'] = predictions[i]['predicted_label']
        sensor_data.loc[sensor_data.index[i], 'probabilities'] = str(predictions[i]['probabilities'])  # Convert to string

    st.write("Predictions:")
    st.dataframe(sensor_data[['Temperature', 'Humidity', 'SoilMoisture', 'elevation', 'rainfall', 'predicted_label', 'probabilities']])
