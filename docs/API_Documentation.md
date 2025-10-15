
# Manufacturing Monitoring System API Documentation

## Overview
This API provides endpoints for monitoring manufacturing equipment, collecting sensor data, running predictions, and managing alerts.

## Accessing the API

In the Docker Compose setup, the API can be accessed in two ways:

1. **Direct Backend Access**: `http://localhost:3001` (recommended for AI agents and external integrations)
2. **Through Frontend Proxy**: `http://localhost:3002` (used by the React frontend)

The frontend Nginx configuration proxies the following API paths to the backend service:

**Proxied Endpoints:**
- `/auth/` - Authentication routes
- `/users` - User management
- `/machines` - Machine management
- `/machine-types` - Machine type management
- `/sensor-data` - Sensor data operations
- `/predictions` - Prediction data
- `/alerts` - Alert management
- `/analyze-data` - LLM analytics
- `/simulate-sensor-data` - Sensor data simulation
- `/health` - Health check

This proxy setup allows the React frontend to make API calls using relative paths (e.g., `/auth/login`), which are transparently forwarded to the backend.

**Note**: For AI agent integration and programmatic access, use the direct backend URL (`http://localhost:3001`). The examples in this documentation use the direct backend access.

## Authentication

### Login
```http
POST /auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "password"
}
```

Response:
```json
{
  "token": "jwt_token_here",
  "user": {
    "id": 1,
    "username": "admin",
    "role": "Admin"
  }
}
```

## Machine Management

### Get All Machines
```http
GET /machines
Authorization: Bearer <token>
```

### Create Machine
```http
POST /machines
Authorization: Bearer <token>
Content-Type: application/json

{
  "machine_type_id": 1,
  "name": "Pump Station A",
  "location": "Floor 1, Section B"
}
```

### Get Machine Types
```http
GET /machine-types
Authorization: Bearer <token>
```

### Create Machine Type
```http
POST /machine-types
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "pump",
  "description": "Centrifugal pump for fluid transfer"
}
```

## Sensor Data

### Get Sensor Data
```http
GET /sensor-data?machine_id=1&start_date=2023-01-01&end_date=2023-12-31&limit=100
Authorization: Bearer <token>
```

### Submit Sensor Data
```http
POST /sensor-data
Authorization: Bearer <token>
Content-Type: application/json

{
  "machine_id": 1,
  "vibration": 2.5,
  "temperature": 28.0,
  "pressure": 2.1,
  "flow_rate": 15.5,
  "rotational_speed": 1520.0
}
```

Response:
```json
{
  "sensorData": {
    "id": 123,
    "machine_id": 1,
    "vibration": 2.5,
    "temperature": 28.0,
    "pressure": 2.1,
    "flow_rate": 15.5,
    "rotational_speed": 1520.0,
    "timestamp": "2023-10-15T10:30:00Z"
  },
  "prediction": {
    "anomaly": false,
    "confidence": 0.85,
    "anomaly_probability": 0.15,
    "model_version": "v1.0_pump",
    "predicted_value": 29.2
  }
}
```

### Simulate Sensor Data
```http
POST /simulate-sensor-data
Authorization: Bearer <token>
Content-Type: application/json

{
  "machine_id": 1
}
```

## Predictions

### Get Predictions
```http
GET /predictions?machine_id=1&limit=50
Authorization: Bearer <token>
```

## Alerts

### Get Alerts
```http
GET /alerts
Authorization: Bearer <token>
```

## Analytics

### Analyze Data with LLM
```http
POST /analyze-data
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "What are the trends in temperature over the last week?",
  "machine_id": 1,
  "start_date": "2023-10-01",
  "end_date": "2023-10-15"
}
```

Response:
```json
{
  "analysis": "Based on the sensor data, temperature has been steadily increasing...",
  "data_count": 2016
}
```

## Health Check

### System Health
```http
GET /health
```

Response:
```json
{
  "status": "OK",
  "timestamp": "2023-10-15T10:30:00Z"
}
```


## Error Responses
All endpoints return appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid input data
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response format:
```json
{
  "error": "Error message description"
}
```

## Rate Limiting
API endpoints are rate limited to 100 requests per 15 minutes per IP address.

## Data Formats

### Sensor Data Structure
```json
{
  "machine_id": "integer (required)",
  "vibration": "float (required)",
  "temperature": "float (required)",
  "pressure": "float (required)",
  "flow_rate": "float (required)",
  "rotational_speed": "float (required)"
}
```

### Machine Structure
```json
{
  "id": "integer",
  "machine_type_id": "integer",
  "name": "string",
  "location": "string",
  "status": "string",
  "created_at": "timestamp"
}
```

### Alert Structure
```json
{
  "id": "integer",
  "machine_id": "integer",
  "alert_type": "string",
  "message": "string",
  "severity": "string",
  "status": "string",
  "created_at": "timestamp"
}
```

## WebSocket Integration (Future Enhancement)
Real-time updates can be received via WebSocket connection for live monitoring.

## AI Agent Integration

### Fetch Sensor Data
AI agents can retrieve sensor data using:
```http
GET /sensor-data?machine_id=<id>&limit=<count>
```

### Submit Sensor Data for Prediction
```http
POST /sensor-data
```

### Query Data with LLM
```http
POST /analyze-data
```

### Get Predictions
```http
GET /predictions?machine_id=<id>
```

### Manage Machines
- `GET /machines` - List all machines
- `POST /machines` - Add new machine
- `GET /machine-types` - List machine types
- `POST /machine-types` - Add new machine type

## Security Notes
- All data transmission is encrypted using HTTPS
- JWT tokens expire after 24 hours
- Passwords are hashed using bcrypt
- Input validation is performed on all endpoints
- Rate limiting prevents abuse
- CORS is configured for cross-origin requests
