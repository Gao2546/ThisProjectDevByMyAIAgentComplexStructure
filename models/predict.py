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
        # --- TensorFlow GPU Configuration (for fast prediction) ---
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logger.info("GPU detected and configured for prediction.")
            except RuntimeError as e:
                logger.error(f"Runtime error during GPU setup: {e}")
        # ---------------------------------------------------------
        
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

                    logger.info(f"Loaded model for {machine_type} (Type: {self.model_infos[machine_type].get('model_type', 'unknown')})")
                except Exception as e:
                    logger.error(f"Error loading model for {machine_type}: {e}")

    def predict(self, sensor_data, machine_type, recent_history=None):
        """
        Make prediction for sensor data using a sequence-based DNN model.

        Args:
            sensor_data: dict with keys: vibration, temperature, pressure, flow_rate, rotational_speed
            machine_type: string indicating machine type
            recent_history: list of recent sensor readings for sequence prediction (optional)

        Returns:
            dict: prediction results
        """
        if machine_type not in self.models:
            available_types = list(self.models.keys())
            if available_types:
                # Use the first available model if the requested one is not found
                machine_type = available_types[0]
                logger.warning(f"Model for requested type '{machine_type}' not found, using '{machine_type}'")
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

            features = model_info['features']
            seq_length = model_info['seq_length']

            # --- 1. Prepare sequence data (3D array: seq_length, n_features) ---
            if recent_history is None or len(recent_history) < seq_length:
                # Fallback: create a sequence by repeating the current observation
                logger.warning(f"Not enough history ({len(recent_history or [])}/{seq_length}). Repeating current sample.")
                
                # Current sample as a 1D array
                current_sample_flat = np.array([[sensor_data.get(feature, 0) for feature in features]])
                
                # Repeat it to create the sequence (seq_length, n_features)
                sequence_data = np.tile(current_sample_flat, (seq_length, 1))
            else:
                # Use recent history (last seq_length observations)
                
                # Extract features from history for the last seq_length points
                history_data = []
                # Combine history with the current sample for the prediction
                full_history = recent_history + [sensor_data]
                
                for hist_point in full_history[-seq_length:]:
                    # Ensure all features are present, use 0 or a sensible default if missing
                    point_features = [hist_point.get(feature, 0) for feature in features]
                    history_data.append(point_features)
                
                # sequence_data is now (seq_length, n_features)
                sequence_data = np.array(history_data)

            # --- 2. Scale the entire sequence ---
            # Reshape the 3D sequence_data to 2D for scaling: (seq_length * n_features)
            sequence_data_2d = sequence_data.reshape(-1, len(features))
            sequence_scaled_2d = scaler.transform(sequence_data_2d)
            
            # Reshape back to the required 3D input format: (1, seq_length, n_features)
            sequence_input = sequence_scaled_2d.reshape(1, seq_length, len(features))

            # --- 3. Make prediction ---
            prediction = model.predict(sequence_input, verbose=0)
            anomaly_prob = float(prediction[0][0])

            # Determine anomaly (threshold at 0.5)
            anomaly = anomaly_prob > 0.5
            confidence = max(anomaly_prob, 1 - anomaly_prob)

            return {
                'anomaly': anomaly,
                'confidence': confidence,
                'anomaly_probability': anomaly_prob,
                'model_version': f"v1.0_{machine_type}",
                'model_type': model_info.get('model_type', 'unknown'),
                'features_used': features
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
    # Create the predictor inside the function to ensure models are loaded
    # However, for efficiency in a real application, ModelPredictor should be initialized once 
    # and reused across predictions.
    predictor = ModelPredictor(model_dir="/models/models")
    return predictor.predict(sensor_data, machine_type, recent_history)


if __name__ == "__main__":
    import sys
    import json
    
    # Simple mock data for demonstration purposes
    mock_history = [
        {'vibration': 2.0, 'temperature': 25.0, 'pressure': 1.9, 'flow_rate': 15.0, 'rotational_speed': 1500.0}
    ] * 49 # Mock 49 points of stable history

    if len(sys.argv) > 2:
        # Command line usage
        sensor_data = json.loads(sys.argv[1])
        machine_type = sys.argv[2]
        
        # Note: Command line usage doesn't easily support passing 'recent_history'
        result = predict_anomaly(sensor_data, machine_type)
        print(json.dumps(result))
    else:
        # Example usage (run with default arguments)
        sample_data = {
            'vibration': 2.5,
            'temperature': 28.0,
            'pressure': 2.1,
            'flow_rate': 15.5,
            'rotational_speed': 1520.0
        }
        
        # Test 1: With insufficient history (will use the repeating sample logic)
        print("--- Test 1: Insufficient History ---")
        result_no_history = predict_anomaly(sample_data, 'pump', recent_history=mock_history[:10])
        print("Prediction result (No History):", result_no_history)
        
        # Test 2: With sufficient history
        print("\n--- Test 2: Sufficient History ---")
        # Ensure 'pump' model exists in 'models/' directory to run correctly
        result_with_history = predict_anomaly(sample_data, 'pump', recent_history=mock_history)
        print("Prediction result (With History):", result_with_history)
        print(json.dumps(result_with_history))