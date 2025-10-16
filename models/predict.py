
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.model_infos = {}
        self.load_models()

    def load_models(self):
        """Load all available models"""
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory {self.model_dir} does not exist")
            return

        for file in os.listdir(self.model_dir):
            if file.endswith('_model.h5'):
                machine_type = file.replace('_model.h5', '')
                try:
                    # Load model
                    model_path = os.path.join(self.model_dir, file)
                    self.models[machine_type] = load_model(model_path)

                    # Load scaler
                    scaler_path = os.path.join(self.model_dir, f'{machine_type}_scaler.pkl')
                    self.scalers[machine_type] = joblib.load(scaler_path)

                    # Load model info
                    info_path = os.path.join(self.model_dir, f'{machine_type}_info.pkl')
                    self.model_infos[machine_type] = joblib.load(info_path)

                    logger.info(f"Loaded model for {machine_type}")
                except Exception as e:
                    logger.error(f"Error loading model for {machine_type}: {e}")

    def predict(self, sensor_data, machine_type, recent_history=None):
        """
        Make prediction for sensor data

        Args:
            sensor_data: dict with keys: vibration, temperature, pressure, flow_rate, rotational_speed
            machine_type: string indicating machine type
            recent_history: list of recent sensor readings for sequence prediction (optional)

        Returns:
            dict: prediction results
        """
        if machine_type not in self.models:
            # Fallback to default model or raise error
            available_types = list(self.models.keys())
            if available_types:
                machine_type = available_types[0]
                logger.warning(f"Model for {machine_type} not found, using {machine_type}")
            else:
                return {
                    'anomaly': False,
                    'confidence': 0.5,
                    'model_version': 'fallback',
                    'error': 'No models available'
                }

        try:
            model = self.models[machine_type]
            scaler = self.scalers[machine_type]
            model_info = self.model_infos[machine_type]

            # Prepare input data
            features = model_info['features']
            input_data = np.array([[sensor_data[feature] for feature in features]])

            # Scale input
            input_scaled = scaler.transform(input_data)

            # For LSTM, we need sequence data
            seq_length = model_info['seq_length']

            if recent_history is None or len(recent_history) < seq_length:
                # If no history, repeat current data to create sequence
                sequence = np.tile(input_scaled, (seq_length, 1))
            else:
                # Use recent history
                history_data = []
                for hist in recent_history[-seq_length:]:
                    hist_features = [hist.get(feature, sensor_data[feature]) for feature in features]
                    history_data.append(hist_features)
                sequence = np.array(history_data)
                sequence = scaler.transform(sequence)

            # Reshape for LSTM input
            sequence = sequence.reshape(1, seq_length, len(features))

            # Make prediction
            prediction = model.predict(sequence, verbose=0)
            anomaly_prob = float(prediction[0][0])

            # Determine anomaly (threshold at 0.5)
            anomaly = anomaly_prob > 0.5
            confidence = max(anomaly_prob, 1 - anomaly_prob)

            return {
                'anomaly': anomaly,
                'confidence': confidence,
                'anomaly_probability': anomaly_prob,
                'model_version': f"v1.0_{machine_type}",
                'predicted_value': sensor_data.get('temperature', 0) + np.random.normal(0, 2)  # Mock prediction
            }

        except Exception as e:
            logger.error(f"Prediction error for {machine_type}: {e}")
            return {
                'anomaly': False,
                'confidence': 0.5,
                'model_version': 'error_fallback',
                'error': str(e)
            }

def predict_anomaly(sensor_data, machine_type, recent_history=None):
    """
    Convenience function for making predictions

    Args:
        sensor_data: dict with sensor readings
        machine_type: string
        recent_history: list of recent readings

    Returns:
        dict: prediction results
    """
    predictor = ModelPredictor()
    return predictor.predict(sensor_data, machine_type, recent_history)


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) > 2:
        sensor_data = json.loads(sys.argv[1])
        machine_type = sys.argv[2]
        result = predict_anomaly(sensor_data, machine_type)
        print(json.dumps(result))
    else:
        # Example usage
        sample_data = {
            'vibration': 2.5,
            'temperature': 28.0,
            'pressure': 2.1,
            'flow_rate': 15.5,
            'rotational_speed': 1520.0
        }
    
        result = predict_anomaly(sample_data, 'pump')
        print("Prediction result:", result)
