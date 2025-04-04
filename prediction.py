import xgboost as xgb
import numpy as np

class LandslidePredictor:
    def __init__(self, model_path="all_features_model.json"):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.class_labels = ['low', 'moderate', 'high', 'very high']

    def predict(self, Temperature, Humidity, SoilMoisture, elevation, rainfall):
        features = np.array([[Temperature, Humidity, rainfall, SoilMoisture, elevation]])
        probabilities = self.model.predict_proba(features)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        predicted_label = self.class_labels[predicted_class]

        return {
            'probabilities': probabilities[0].tolist(),
            'predicted_class': predicted_class,
            'predicted_label': predicted_label
        }

    def check_confidence(self, AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ, threshold=0.3):
        """
        Returns True if sensor data indicates high motion suggesting possible landslide.
        Simple confidence check based on vector magnitude.
        """
        try:
            ax, ay, az = float(AccelX), float(AccelY), float(AccelZ)
            gx, gy, gz = float(GyroX), float(GyroY), float(GyroZ)
        except ValueError:
            raise ValueError("Invalid string format in acceleration or gyroscope data.")

        acc_magnitude = (ax**2 + ay**2 + az**2)**0.5
        gyro_magnitude = (gx**2 + gy**2 + gz**2)**0.5

        motion_score = acc_magnitude + gyro_magnitude
        return motion_score > threshold  

    def corrected_predict(self, Temperature, Humidity, SoilMoisture, elevation, rainfall,
                          AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ):
        base_result = self.predict(Temperature, Humidity, SoilMoisture, elevation, rainfall)
        motion_flag = self.check_confidence(AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ)

        if motion_flag and base_result['predicted_class'] < 2:
            base_result['predicted_label'] = 'high'
            base_result['predicted_class'] = 2
            base_result['confidence_adjusted'] = True
        else:
            base_result['confidence_adjusted'] = False

        return base_result


if __name__ == "__main__":
    predictor = LandslidePredictor()
    result = predictor.corrected_predict(
        Temperature=35,
        Humidity=89,
        SoilMoisture=88,
        elevation=958,
        rainfall=3.5,
        AccelX="-4.59",
        AccelY="7.40",
        AccelZ="5.41",
        GyroX="-0.01",
        GyroY="0.07",
        GyroZ="0.01"
    )

    print("Probabilities:", result['probabilities'])
    print("Predicted Class:", result['predicted_label'])
    print("Confidence Adjusted:", result['confidence_adjusted'])
