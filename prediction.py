import xgboost as xgb
import numpy as np

class LandslidePredictor:
    def __init__(self, model_path="all_features_model.json"):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.class_labels = ['low', 'moderate', 'high', 'very high']
        
    def predict(self, Temperature, Humidity, SoilMoisture, elevation, rainfall):
        features = np.array([[Temperature, Humidity,rainfall, SoilMoisture, elevation]])
        
        # Get class probabilities
        probabilities = self.model.predict_proba(features)
        
        # Get predicted class
        predicted_class = np.argmax(probabilities, axis=1)[0]
        predicted_label = self.class_labels[predicted_class]
        
        return {
            'probabilities': probabilities[0].tolist(),
            'predicted_class': predicted_class,
            'predicted_label': predicted_label
        }

if __name__ == "__main__":
    predictor = LandslidePredictor()
    result = predictor.predict(35, 89, 88, 958)
    print("Probabilities:", result['probabilities'])
    print("Predicted Class:", result['predicted_label'])