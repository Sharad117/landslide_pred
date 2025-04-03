import firebase_admin
from firebase_admin import credentials, firestore
import requests
import pandas as pd

class FirestoreClient:
    def __init__(self, creds_path="landslide-4d148-firebase-adminsdk-fbsvc-c9c527dd06.json"):
        """Initialize Firestore client"""
        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_path)
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
    
    def get_recent_sensor_data(self, collection_name="EspData"):
        """
        Retrieve the latest K sensor data entries
        """
        k = 15
        query = self.db.collection(collection_name).limit(k)
        results = query.get()
        
        sensor_data = []
        for doc in results:
            data = doc.to_dict()
            data['id'] = doc.id
            sensor_data.append(data)
        
        return pd.DataFrame(sensor_data) 

    def fetch_weather_data(self, latitude, longitude, api_key):
        """
        Fetch rainfall from OpenWeather API and elevation from Open-Elevation API
        """
        weather_api_url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
        response = requests.get(weather_api_url)

        if response.status_code == 200:
            data = response.json()
            rainfall = data.get('rain', {}).get('1h', 0)  # Rainfall in mm (last 1 hour)
        else:
            raise Exception(f"Failed to fetch weather data: {response.status_code} - {response.text}")

        elevation_api_url = f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}"
        response = requests.get(elevation_api_url)

        if response.status_code == 200:
            elevation_data = response.json()
            elevation = elevation_data['results'][0]['elevation']
        else:
            elevation = None  

        return elevation, rainfall

    def append_weather_data(self, sensor_data, latitude, longitude, api_key):
        """
        Append elevation and rainfall data to the sensor data
        """
        elevation, rainfall = self.fetch_weather_data(latitude, longitude, api_key)
        sensor_data['elevation'] = elevation
        sensor_data['rainfall'] = rainfall
        return sensor_data
