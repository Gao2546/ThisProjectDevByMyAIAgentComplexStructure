
# Manufacturing Monitoring System

A comprehensive system for monitoring manufacturing equipment using sensor data, machine learning predictions, and real-time analytics.

## Features

- **Real-time Sensor Data Collection**: Collects data from vibration, temperature, pressure, flow rate, and rotational speed sensors
- **Machine Learning Predictions**: Uses LSTM models to detect anomalies and predict equipment failures
- **Database Storage**: PostgreSQL database for storing sensor data, predictions, and alerts
- **User Interface**: React-based dashboard for monitoring and analytics
- **API Integration**: RESTful API for external integrations and AI agent access
- **LLM Analytics**: Integration with Ollama for natural language queries on sensor data
- **Alert System**: Automated alerts for anomalies and equipment issues
- **Docker Deployment**: Containerized deployment with Docker Compose

## Architecture

The system consists of five main components:

1. **Backend (Node.js/Express)**: API server handling data collection, predictions, and database operations
2. **Frontend (React)**: User interface for monitoring and analytics
3. **Database (PostgreSQL)**: Data storage for sensor readings, predictions, and metadata
4. **Models (Python/TensorFlow)**: Machine learning models for anomaly detection
5. **Ollama**: LLM service for natural language analytics on sensor data


## Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.8+ (for model training)

### Using Docker Compose

1. Clone the repository
2. Navigate to the project directory
3. Copy environment files and configure:
   ```bash
   cp backend/.env.example backend/.env
   # Edit backend/.env with your configuration
   ```
4. Start the system:
   ```bash
   docker-compose up -d
   ```


The system will be available at:
- Frontend: http://localhost:3002
- Backend API: http://localhost:3001
- Database: localhost:5433
- Ollama API: localhost:11435


### Local Development

#### Backend Setup
```bash
cd backend
npm install
cp .env.example .env
# Edit .env file
npm run dev
```

#### Frontend Setup
```bash
cd frontend
npm install
npm start
```

#### Database Setup
```bash
# Start PostgreSQL
docker run -d --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=securepassword postgres:15

# Run initialization script
docker exec -i postgres psql -U postgres < database/init.sql
```

#### Model Training
```bash
cd models
pip install -r requirements.txt
python train_model.py --machine_type pump
python train_model.py --machine_type motor
python train_model.py --machine_type conveyor
```

## API Usage

### Authentication
```bash
curl -X POST http://localhost:3001/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"username": "admin", "password": "password"}'
```

### Submit Sensor Data
```bash
curl -X POST http://localhost:3001/sensor-data \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "machine_id": 1,
    "vibration": 2.5,
    "temperature": 28.0,
    "pressure": 2.1,
    "flow_rate": 15.5,
    "rotational_speed": 1520.0
  }'
```

### Query Data with LLM
```bash
curl -X POST http://localhost:3001/analyze-data \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "What are the temperature trends?",
    "machine_id": 1
  }'
```


## Configuration

### Environment Variables

#### Backend (.env)
```
DATABASE_URL=postgresql://admin:securepassword@postgres:5432/manufacturing_db
JWT_SECRET=your_super_secret_jwt_key_here
PORT=3000
OLLAMA_BASE_URL=http://ollama:11434
SENSOR_INTERVAL=5000
```

#### Database
The system supports multiple machine types. To add a new machine type:

1. Add to database:
   ```sql
   INSERT INTO machine_types (name, description) VALUES ('new_type', 'Description');
   ```

2. Train a model:
   ```bash
   python models/train_model.py --machine_type new_type
   ```

3. Update the backend code to handle the new type in `callPredictionModel`

## Security Features

- JWT-based authentication
- Password hashing with bcrypt
- Rate limiting
- Input validation with Joi
- HTTPS encryption
- CORS configuration
- Winston logging

## Monitoring and Alerts

- Real-time sensor data collection every 5 seconds
- Anomaly detection with configurable thresholds
- Email notifications (configurable)
- Dashboard alerts
- Historical data analysis

## AI Agent Integration

The system provides comprehensive API endpoints for AI agents to:

- Fetch sensor data: `GET /sensor-data`
- Submit sensor readings: `POST /sensor-data`
- Query analytics: `POST /analyze-data`
- Manage machines: `GET/POST /machines`
- Access predictions: `GET /predictions`

## Scaling Considerations

- Horizontal scaling with load balancers
- Redis for caching and session management
- Database indexing for performance
- Model versioning for A/B testing
- Container orchestration with Kubernetes

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check PostgreSQL container is running
   - Verify DATABASE_URL in .env

2. **Model Prediction Errors**
   - Ensure models are trained and saved
   - Check model file permissions

3. **Ollama Connection Issues**
   - Verify Ollama container is running
   - Check OLLAMA_BASE_URL configuration

### Logs
- Backend logs: Available in `backend/error.log` and `backend/combined.log`
- Database logs: `docker logs postgres`
- Ollama logs: `docker logs ollama`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details
