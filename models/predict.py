import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging
import json # Import json for main block usage

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
                    self.models[machine_type] = load_model(model_path, compile=False) # compile=False speeds up loading

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
        Make prediction for sensor data using either a sequence-based or point-in-time model.

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
                # NOTE: This fallback might use an incompatible model, but it prevents a crash.
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
            seq_length = model_info.get('seq_length', 1) # Default to 1 if missing for safety
            n_features = len(features)

            # --- 1. Prepare sequence data (Shape: seq_length, n_features) ---
            if seq_length > 1:
                # Sequence Model Logic (for LSTM/RNN/Sequence DNN)
                if recent_history is None or len(recent_history) < seq_length:
                    # Fallback: create a sequence by repeating the current observation
                    logger.warning(f"Not enough history ({len(recent_history or [])}/{seq_length}). Repeating current sample.")
                    
                    current_sample_flat = np.array([[sensor_data.get(feature, 0) for feature in features]])
                    sequence_data = np.tile(current_sample_flat, (seq_length, 1))
                else:
                    # Use recent history (last seq_length observations)
                    history_data = []
                    full_history = recent_history + [sensor_data]
                    
                    for hist_point in full_history[-seq_length:]:
                        point_features = [hist_point.get(feature, 0) for feature in features]
                        history_data.append(point_features)
                    
                    sequence_data = np.array(history_data)
            else:
                # Point-in-Time Model Logic (for FNN)
                # sequence_data is just the current sample, (1, n_features)
                sequence_data = np.array([[sensor_data.get(feature, 0) for feature in features]])


            # --- 2. Scale the data ---
            # sequence_data is always 2D here: (seq_length, n_features) or (1, n_features)
            sequence_scaled = scaler.transform(sequence_data)
            
            # --- 3. Prepare final input shape for Keras model ---
            if seq_length > 1:
                # 3D input: (batch_size, seq_length, n_features)
                sequence_input = sequence_scaled.reshape(1, seq_length, n_features)
            else:
                # 2D input: (batch_size, n_features). This fixes the ValueError.
                # sequence_scaled is already (1, n_features).
                sequence_input = sequence_scaled 

            # --- 4. Make prediction ---
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
    """
    # Use './models' as the standard path where the training script saves files
    predictor = ModelPredictor(model_dir="../models/models") 
    return predictor.predict(sensor_data, machine_type, recent_history)


if __name__ == "__main__":
    import sys
    
    # Simple mock data for demonstration purposes
    # Set history length to satisfy a potential seq_length=50 model, but use a machine
    # type ('pump_fnn') that should have seq_length=1 in its info file.
    mock_history = [
        {'vibration': 2.0, 'temperature': 25.0, 'pressure': 1.9, 'flow_rate': 15.0, 'rotational_speed': 1500.0}
    ] * 49 

    if len(sys.argv) > 2:
        # Command line usage
        sensor_data = json.loads(sys.argv[1])
        machine_type = sys.argv[2]
        
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
        
        # NOTE: To test the fix, you must have trained a 'pump' model 
        # using the last script (which sets seq_length=1).
        print("--- Testing Point-in-Time FNN Model (Requires 'pump' model with seq_length=1) ---")
        
        # Since the model expects [batch, 5], we ignore history.
        result_fnn = predict_anomaly(sample_data, 'pump', recent_history=None)
        
        print("Prediction result (FNN Model):", json.dumps(result_fnn, indent=4))
        
        print("\n--- Testing Sequence Model Logic (Only if a 'motor' model with seq_length>1 exists) ---")
        # This will use the sequence path in the predict method.
        result_seq_test = predict_anomaly(sample_data, 'motor', recent_history=mock_history)
        print("Prediction result (Sequence Model Test):", json.dumps(result_seq_test, indent=4))