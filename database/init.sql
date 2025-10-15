
-- Create database schema for manufacturing monitoring system

-- Table for machine types
CREATE TABLE machine_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for machines
CREATE TABLE machines (
    id SERIAL PRIMARY KEY,
    machine_type_id INTEGER REFERENCES machine_types(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for sensor data
CREATE TABLE sensor_data (
    id SERIAL PRIMARY KEY,
    machine_id INTEGER REFERENCES machines(id) ON DELETE CASCADE,
    vibration FLOAT,
    temperature FLOAT,
    pressure FLOAT,
    flow_rate FLOAT,
    rotational_speed FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for predictions
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    sensor_data_id INTEGER REFERENCES sensor_data(id) ON DELETE CASCADE,
    prediction_result JSONB,
    confidence_score FLOAT,
    model_version VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for users/roles
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) UNIQUE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('System', 'Agent', 'Admin')),
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for alerts
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    machine_id INTEGER REFERENCES machines(id) ON DELETE CASCADE,
    alert_type VARCHAR(100),
    message TEXT,
    severity VARCHAR(50),
    status VARCHAR(50) DEFAULT 'active',
    notified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_sensor_data_machine_id ON sensor_data(machine_id);
CREATE INDEX idx_sensor_data_timestamp ON sensor_data(timestamp);
CREATE INDEX idx_predictions_sensor_data_id ON predictions(sensor_data_id);
CREATE INDEX idx_alerts_machine_id ON alerts(machine_id);

-- Insert initial admin user
INSERT INTO users (username, role, password_hash) VALUES ('admin', 'Admin', '$2a$10$KjSxYfR4xGOKf2tWn4jajOCy6/kdn04qCo2jItcRPQgTlZSnmj3lu');
