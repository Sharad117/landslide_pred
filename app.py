import streamlit as st
from streamlit_autorefresh import st_autorefresh
from firestore_client import FirestoreClient
from prediction import LandslidePredictor
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

firestore = FirestoreClient()
predictor = LandslidePredictor()
API_KEY = os.getenv("API_KEY")

st.set_page_config(page_title="Landslide Dashboard", layout="wide")
st.title(" Landslide Prediction Dashboard")

st.sidebar.header(" Location Settings")
latitude = st.sidebar.number_input("Enter Latitude", value=12.9716, format="%.6f")
longitude = st.sidebar.number_input("Enter Longitude", value=77.5946, format="%.6f")
refresh_interval = st.sidebar.slider("Auto Refresh Interval (seconds)", min_value=30, max_value=3600, value=30)

count = st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")

manual_refresh = st.button("Refresh Now")

st.info(f"Page refreshed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", icon="🕒")

if count > 0 or manual_refresh:
    st.header("Recent Sensor Data")
    sensor_data = firestore.get_recent_sensor_data()

    if sensor_data.empty:
        st.warning("No recent sensor data available.")
    else:
        sensor_data = firestore.append_weather_data(sensor_data, latitude, longitude, API_KEY)

        sensor_data[['Temperature', 'Humidity', 'SoilMoisture', 'elevation', 'rainfall',
                     'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']] = \
            sensor_data[['Temperature', 'Humidity', 'SoilMoisture', 'elevation', 'rainfall',
                         'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']].astype(float)

        st.dataframe(sensor_data, use_container_width=True)

        st.header("Landslide Risk Predictions")
        predictions = []

        for _, row in sensor_data.iterrows():
            prediction = predictor.corrected_predict(
                Temperature=row['Temperature'],
                Humidity=row['Humidity'],
                SoilMoisture=row['SoilMoisture'],
                elevation=row['elevation'],
                rainfall=row['rainfall'],
                AccelX=row['AccelX'],
                AccelY=row['AccelY'],
                AccelZ=row['AccelZ'],
                GyroX=row['GyroX'],
                GyroY=row['GyroY'],
                GyroZ=row['GyroZ']
            )
            predictions.append(prediction)

        for i in range(len(sensor_data)):
            sensor_data.loc[sensor_data.index[i], 'predicted_label'] = predictions[i]['predicted_label']
            sensor_data.loc[sensor_data.index[i], 'predicted_class'] = predictions[i]['predicted_class']
            sensor_data.loc[sensor_data.index[i], 'confidence_adjusted'] = predictions[i]['confidence_adjusted']
            sensor_data.loc[sensor_data.index[i], 'probabilities'] = str(predictions[i]['probabilities'])

        st.dataframe(sensor_data[['Temperature', 'Humidity', 'SoilMoisture', 'elevation', 'rainfall',
                                  'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ',
                                  'predicted_label', 'predicted_class', 'confidence_adjusted', 'probabilities']],
                     use_container_width=True)
