import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout # Added Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os
import argparse

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

def create_sequences(data, seq_length=50):
    """Create sequences for model input"""
    sequences = []
    targets = []

    # Iterate through the data to create overlapping sequences
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length, :-1]  # All columns except anomaly
        target = data[i+seq_length, -1]  # Anomaly label at the prediction step
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)

def build_dnn_model(input_shape):
    """Build a classical Deep Neural Network (DNN) model for anomaly detection
       It uses a Flatten layer to handle the sequence input structure."""
    
    # Input shape is (seq_length, n_features). Flatten converts this to (seq_length * n_features)
    model = Sequential([
        Flatten(input_shape=input_shape), # Flattens the time sequences into a single feature vector
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification for anomaly
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(machine_type, seq_length=50, epochs=50, batch_size=32):
    """Train model for specific machine type"""
    print(f"Training DNN model for {machine_type}...")

    # Generate sample data
    data = generate_sample_data(machine_type)

    # Prepare features
    features = ['vibration', 'temperature', 'pressure', 'flow_rate', 'rotational_speed', 'anomaly']
    data_values = data[features].values

    # Create sequences
    X, y = create_sequences(data_values, seq_length)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Scaling Features (Applied across features in the sequence) ---
    scaler = StandardScaler()
    # Reshape (n_samples, seq_length, n_features) -> (n_samples * seq_length, n_features) for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[2])

    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    # -----------------------------------------------------------------

    # Build model using the new DNN architecture
    # Input shape is (seq_length, n_features - 1) for the Flatten layer
    input_shape = (seq_length, X.shape[2])
    model = build_dnn_model(input_shape)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'models/{machine_type}_model.h5', monitor='val_loss', save_best_only=True)

    # Train model
    # We pass the scaled sequences (X_train_scaled) to the model. The Flatten layer handles the 3D to 2D conversion inside Keras.
    history = model.fit(
        X_train_scaled[:, :, :], y_train,  # Remove anomaly from input features
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Evaluate model
    loss, accuracy = model.evaluate(X_test_scaled[:, :, :], y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Save scaler
    joblib.dump(scaler, f'models/{machine_type}_scaler.pkl')

    # Save model info
    model_info = {
        'machine_type': machine_type,
        'seq_length': seq_length,
        'accuracy': accuracy,
        'loss': loss,
        'model_type': 'DNN_FFN', # Updated model type
        'features': features[:-1]  # Exclude anomaly
    }

    joblib.dump(model_info, f'models/{machine_type}_info.pkl')

    print(f"Model for {machine_type} trained and saved.")
    return model, scaler, model_info

def main():
    parser = argparse.ArgumentParser(description='Train ML model for machine monitoring')
    parser.add_argument('--machine_type', type=str, required=True,
                       help='Type of machine (pump, motor, conveyor, etc.)')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='Sequence length for DNN')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')

    args = parser.parse_args()
    
    # --- GPU Configuration: Check for and configure GPU use ---
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Enable memory growth to avoid allocating all GPU memory at once
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("--- GPU detected and configured for training. ---")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(f"Runtime error during GPU setup: {e}")
    else:
        print("--- No GPU detected. Training will run on CPU. ---")
    # ---------------------------------------------------------

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Train model
    model, scaler, info = train_model(args.machine_type, args.seq_length, args.epochs)

    print(f"Training completed for {args.machine_type}")
    print(f"Model saved as: models/{args.machine_type}_model.h5")
    print(f"Scaler saved as: models/{args.machine_type}_scaler.pkl")
    print(f"Model info saved as: models/{args.machine_type}_info.pkl")

if __name__ == "__main__":
    main()
