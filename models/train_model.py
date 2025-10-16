import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report # Changed metrics for classification
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os
import argparse
import sys # Added sys for argument parsing note

def generate_sample_data(machine_type, n_samples=10000):
    """Generate sample time series data for different machine types"""
    np.random.seed(42)

    # Base parameters for different machine types
    if machine_type == 'pump':
        base_temp = 25
        base_vib = 2
        base_press = 2
        base_flow = 15
        base_rot = 1500
    elif machine_type == 'motor':
        base_temp = 30
        base_vib = 1.5
        base_press = 1
        base_flow = 10
        base_rot = 1800
    elif machine_type == 'conveyor':
        base_temp = 20
        base_vib = 3
        base_press = 0.5
        base_flow = 5
        base_rot = 1000
    else:  # default
        base_temp = 25
        base_vib = 2
        base_press = 1.5
        base_flow = 12
        base_rot = 1200

    # Generate time series data
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='5S')

    data = []
    for i in range(n_samples):
        # Add some trends and seasonality
        trend = i * 0.001
        seasonal = np.sin(2 * np.pi * i / 100) * 2

        # Random noise
        noise_temp = np.random.normal(0, 2)
        noise_vib = np.random.normal(0, 0.5)
        noise_press = np.random.normal(0, 0.2)
        noise_flow = np.random.normal(0, 1)
        noise_rot = np.random.normal(0, 50)

        # Occasionally add anomalies
        anomaly = np.random.choice([0, 1], p=[0.95, 0.05])
        if anomaly:
            noise_temp *= 5
            noise_vib *= 3
            noise_press *= 2
            noise_flow *= 2
            noise_rot *= 2

        row = {
            'timestamp': timestamps[i],
            'vibration': max(0, base_vib + trend + seasonal + noise_vib),
            'temperature': max(0, base_temp + trend + seasonal + noise_temp),
            'pressure': max(0, base_press + trend + seasonal + noise_press),
            'flow_rate': max(0, base_flow + trend + seasonal + noise_flow),
            'rotational_speed': max(0, base_rot + trend + seasonal + noise_rot),
            'anomaly': anomaly
        }
        data.append(row)

    return pd.DataFrame(data)

# NOTE: The create_sequences function is REMOVED as we are moving to [batch, data] input.

def build_dnn_model(input_shape):
    """Build a classical Deep Neural Network (DNN) model for anomaly detection.
       It is configured for [batch, features] input."""
    
    # Input shape is (n_features,)
    model = Sequential([
        # The first Dense layer automatically handles the input shape (features)
        Dense(128, activation='relu', input_shape=input_shape), 
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification for anomaly
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(machine_type, seq_length=1, epochs=50, batch_size=32):
    """Train model for specific machine type using single-sample input [batch, features]"""
    print(f"Training DNN model for {machine_type} with [batch, features] input...")

    # Generate sample data
    data = generate_sample_data(machine_type)

    # Prepare features and target
    features = ['vibration', 'temperature', 'pressure', 'flow_rate', 'rotational_speed']
    target = 'anomaly'

    X = data[features].values
    y = data[target].values
    
    # Split data (standard train/test split for classification)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Scaling Features (Applied to standard 2D array [n_samples, n_features]) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # --------------------------------------------------------------------------------

    # Build model. Input shape is (n_features,)
    input_shape = (X_train_scaled.shape[1],)
    model = build_dnn_model(input_shape)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'models/{machine_type}_model.h5', monitor='val_loss', save_best_only=True)

    # Train model
    # X_train_scaled is now [n_samples, n_features], compatible with the FNN model
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Evaluate model
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Further evaluation with classification report
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save scaler
    joblib.dump(scaler, f'models/{machine_type}_scaler.pkl')

    # Save model info
    # We set seq_length to 1 to signify single-sample/point-in-time prediction
    model_info = {
        'machine_type': machine_type,
        'seq_length': 1, # Set to 1 for point-in-time classification
        'accuracy': accuracy,
        'loss': loss,
        'model_type': 'FNN_PointInTime', # Updated model type
        'features': features
    }

    joblib.dump(model_info, f'models/{machine_type}_info.pkl')

    print(f"Model for {machine_type} trained and saved.")
    return model, scaler, model_info

def main():
    # Helper to allow running without command line args for testing
    if len(sys.argv) == 1:
        # No arguments provided, default to 'pump' for demonstration
        print("Using default machine_type='pump' for demonstration. Run with --machine_type <type> to specify.")
        args = argparse.Namespace(machine_type='pump', seq_length=1, epochs=50)
    else:
        parser = argparse.ArgumentParser(description='Train ML model for machine monitoring')
        parser.add_argument('--machine_type', type=str, required=True,
                           help='Type of machine (pump, motor, conveyor, etc.)')
        parser.add_argument('--seq_length', type=int, default=1, # Default to 1 for point-in-time
                           help='Sequence length (set to 1 for point-in-time classification)')
        parser.add_argument('--epochs', type=int, default=50,
                           help='Number of training epochs')
        args = parser.parse_args()
    
    # --- GPU Configuration: Check for and configure GPU use ---
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("--- GPU detected and configured for training. ---")
        except RuntimeError as e:
            print(f"Runtime error during GPU setup: {e}")
    else:
        print("--- No GPU detected. Training will run on CPU. ---")
    # ---------------------------------------------------------

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Train model
    # Note: seq_length is now effectively ignored or fixed at 1 for this architecture
    model, scaler, info = train_model(args.machine_type, args.seq_length, args.epochs)

    print(f"\nTraining completed for {args.machine_type}")
    print(f"Model saved as: models/{args.machine_type}_model.h5")
    print(f"Scaler saved as: models/{args.machine_type}_scaler.pkl")
    print(f"Model info saved as: models/{args.machine_type}_info.pkl")

if __name__ == "__main__":
    main()