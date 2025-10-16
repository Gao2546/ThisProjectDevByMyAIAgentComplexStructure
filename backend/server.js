
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const cron = require('node-cron');
const winston = require('winston');
const { Pool } = require('pg');
const dotenv = require('dotenv');
const Joi = require('joi');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');

const { spawn } = require('child_process');
const path = require('path');


dotenv.config();


const app = express();
app.set('trust proxy', 1);

const PORT = process.env.PORT || 5000;


console.log("starting");
// Database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

// Logger setup
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
  ],
});

if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.simple(),
  }));
}

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Middleware to disable caching for all API responses
const noCache = (req, res, next) => {
  res.set('Cache-Control', 'no-store, no-cache, must-revalidate, private');
  res.set('Pragma', 'no-cache');
  res.set('Expires', '0');
  next();
};


// Rate limiting
const limiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 60 minutes
  max: 1000, // limit each IP to 1000 requests per windowMs
});

app.use(limiter);

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) return res.sendStatus(401);

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) {
      console.log("no token")
      return res.sendStatus(403);
    }
    req.user = user;
    next();
  });
};

// Validation schemas
const sensorDataSchema = Joi.object({
  machine_id: Joi.number().integer().required(),
  vibration: Joi.number().required(),
  temperature: Joi.number().required(),
  pressure: Joi.number().required(),
  flow_rate: Joi.number().required(),
  rotational_speed: Joi.number().required(),
});


const machineSchema = Joi.object({
  machine_type_id: Joi.number().integer().required(),
  name: Joi.string().required(),
  location: Joi.string().optional(),
  status: Joi.string().valid('active', 'inactive').optional().default('active'),
});


// Routes

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Authentication routes
app.post('/auth/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    const result = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
    const user = result.rows[0];

    if (!user || !(await bcrypt.compare(password, user.password_hash))) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign({ id: user.id, role: user.role }, process.env.JWT_SECRET, { expiresIn: '24h' });
    res.json({ token, user: { id: user.id, username: user.username, role: user.role } });
  } catch (error) {
    logger.error('Login error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/auth/register', authenticateToken, async (req, res) => {
  if (req.user.role !== 'Admin') return res.sendStatus(403);

  try {
    const { username, email, password, role } = req.body;
    const hashedPassword = await bcrypt.hash(password, 10);

    const result = await pool.query(
      'INSERT INTO users (username, email, password_hash, role) VALUES ($1, $2, $3, $4) RETURNING id, username, role',
      [username, email, hashedPassword, role]
    );

    res.status(201).json(result.rows[0]);
  } catch (error) {
    logger.error('Registration error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }

});


// GET users
app.get('/users', authenticateToken, async (req, res) => {
  console.log('GET /users called by user:', req.user);
  logger.error(`User role is not Admin: ${req.user.role}`);
  if (req.user.role !== 'Admin') {
    console.log('User role is not Admin:', req.user.role);
    logger.error(`User role is not Admin: ${req.user.role}`);
    return res.sendStatus(403);
  }

  try {
    const result = await pool.query('SELECT id, username, email, role, created_at FROM users');
    
    // FIX: Set headers to prevent caching
    res.set('Cache-Control', 'no-store'); 
    
    res.json(result.rows);
  } catch (error) {

    console.log('Error fetching users:', error);
    logger.error('Get users error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});



// Machine management
app.get('/machines', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query(`

      SELECT m.*, mt.name as machine_type_name
      FROM machines m
      JOIN machine_types mt ON m.machine_type_id = mt.id
    `);
    res.json(result.rows);
  } catch (error) {
    logger.error('Get machines error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/machines', authenticateToken, async (req, res) => {
  if (req.user.role !== 'Admin') return res.sendStatus(403);

  try {
    const { error } = machineSchema.validate(req.body);
    if (error) return res.status(400).json({ error: error.details[0].message });

    const { machine_type_id, name, location, status = 'active' } = req.body;

    const result = await pool.query(
      'INSERT INTO machines (machine_type_id, name, location, status) VALUES ($1, $2, $3, $4) RETURNING *',
      [machine_type_id, name, location, status]
    );


    res.status(201).json(result.rows[0]);
  } catch (error) {
    logger.error('Create machine error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


// Machine types
app.get('/machine-types', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM machine_types');
    res.json(result.rows);
  } catch (error) {
    logger.error('Get machine types error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST machine type
app.post('/machine-types', authenticateToken, async (req, res) => {
  if (req.user.role !== 'Admin') return res.sendStatus(403);

  try {
    const { name, description } = req.body;
    const result = await pool.query(
      'INSERT INTO machine_types (name, description) VALUES ($1, $2) RETURNING *',
      [name, description]
    );
    res.status(201).json(result.rows[0]);
  } catch (error) {
    logger.error('Create machine type error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// DELETE machine type
app.delete('/machine-types/:id', authenticateToken, async (req, res) => {
  if (req.user.role !== 'Admin') return res.sendStatus(403);

  try {
    const { id } = req.params;
    const result = await pool.query('DELETE FROM machine_types WHERE id = $1 RETURNING *', [id]);
    if (result.rows.length === 0) return res.status(404).json({ error: 'Machine type not found' });
    res.json({ message: 'Machine type deleted' });
  } catch (error) {
    logger.error('Delete machine type error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});




// Sensor data routes
app.get('/sensor-data', authenticateToken, async (req, res) => {
  try {
    const { machine_id, start_date, end_date, limit = 1000000 } = req.query;
    let query = `
      SELECT sd.*, m.name as machine_name, mt.name as machine_type
      FROM sensor_data sd
      JOIN machines m ON sd.machine_id = m.id
      JOIN machine_types mt ON m.machine_type_id = mt.id
    `;
    const params = [];
    const conditions = [];

    if (machine_id) {
      conditions.push(`sd.machine_id = $${params.length + 1}`);
      params.push(machine_id);
    }

    if (start_date) {
      conditions.push(`sd.timestamp >= $${params.length + 1}`);
      params.push(start_date);
    }

    if (end_date) {
      conditions.push(`sd.timestamp <= $${params.length + 1}`);
      params.push(end_date);
    }

    if (conditions.length > 0) {
      query += ' WHERE ' + conditions.join(' AND ');
    }

    query += ` ORDER BY sd.timestamp DESC LIMIT $${params.length + 1}`;
    params.push(limit);

    const result = await pool.query(query, params);
    // console.log(result);
    // console.log(result.rows);
    res.json(result.rows);
  } catch (error) {
    logger.error('Get sensor data error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/sensor-data', authenticateToken, async (req, res) => {
  try {
    const { error } = sensorDataSchema.validate(req.body);
    if (error) return res.status(400).json({ error: error.details[0].message });

    const { machine_id, vibration, temperature, pressure, flow_rate, rotational_speed } = req.body;

    // Insert sensor data
    const sensorResult = await pool.query(
      'INSERT INTO sensor_data (machine_id, vibration, temperature, pressure, flow_rate, rotational_speed) VALUES ($1, $2, $3, $4, $5, $6) RETURNING *',
      [machine_id, vibration, temperature, pressure, flow_rate, rotational_speed]
    );

    const sensorData = sensorResult.rows[0];

    // Get machine type for model selection
    const machineTypeResult = await pool.query(`
      SELECT mt.name as machine_type
      FROM machines m
      JOIN machine_types mt ON m.machine_type_id = mt.id
      WHERE m.id = $1
    `, [machine_id]);

    const machineType = machineTypeResult.rows[0].machine_type;

    // Call prediction model
    const prediction = await callPredictionModel(sensorData, machineType);

    // Store prediction
    await pool.query(
      'INSERT INTO predictions (sensor_data_id, prediction_result, confidence_score, model_version) VALUES ($1, $2, $3, $4)',
      [sensorData.id, JSON.stringify(prediction.result), prediction.confidence, prediction.model_version]
    );

    // Check for anomalies and create alerts
    if (prediction.result.anomaly) {
      await createAlert(machine_id, 'Anomaly Detected', `Anomaly detected for machine ${machine_id}`, 'high');
    }

    res.status(201).json({ sensorData, prediction });
  } catch (error) {
    logger.error('Post sensor data error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Prediction routes
app.get('/predictions', authenticateToken, async (req, res) => {
  try {
    const { machine_id, limit = 50 } = req.query;
    let query = `
      SELECT p.*, sd.machine_id, m.name as machine_name
      FROM predictions p
      JOIN sensor_data sd ON p.sensor_data_id = sd.id
      JOIN machines m ON sd.machine_id = m.id
    `;
    const params = [];

    if (machine_id) {
      query += ' WHERE sd.machine_id = $1';
      params.push(machine_id);
    }

    query += ` ORDER BY p.created_at DESC LIMIT $${params.length + 1}`;
    params.push(limit);

    const result = await pool.query(query, params);
    res.json(result.rows);
  } catch (error) {
    logger.error('Get predictions error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Alerts
app.get('/alerts', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT a.*, m.name as machine_name
      FROM alerts a
      JOIN machines m ON a.machine_id = m.id
      ORDER BY a.created_at DESC
    `);
    res.json(result.rows);
  } catch (error) {
    logger.error('Get alerts error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Sensor simulation (for testing)
app.post('/simulate-sensor-data', authenticateToken, async (req, res) => {
  try {
    const { machine_id } = req.body;
    if (!machine_id) return res.status(400).json({ error: 'machine_id is required' });

    // Simulate sensor data
    const sensorData = {
      machine_id,
      vibration: Math.random() * 10,
      temperature: 20 + Math.random() * 30,
      pressure: 1 + Math.random() * 5,
      flow_rate: 10 + Math.random() * 20,
      rotational_speed: 1000 + Math.random() * 2000,
    };

    // Post the simulated data
    const response = await axios.post(`http://localhost:${PORT}/sensor-data`, sensorData, {
      headers: { Authorization: req.headers.authorization }
    });

    res.json({ message: 'Sensor data simulated', data: response.data });
  } catch (error) {
    logger.error('Simulate sensor data error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// LLM Analysis
app.post('/analyze-data', authenticateToken, async (req, res) => {
  try {
    const { query, machine_id, start_date, end_date } = req.body;

    // Fetch relevant data
    let dataQuery = `
      SELECT sd.*, m.name as machine_name, mt.name as machine_type, p.prediction_result
      FROM sensor_data sd
      JOIN machines m ON sd.machine_id = m.id
      JOIN machine_types mt ON m.machine_type_id = mt.id
      LEFT JOIN predictions p ON sd.id = p.sensor_data_id
      WHERE 1=1
    `;
    const params = [];
    const conditions = [];

    if (machine_id) {
      conditions.push(`sd.machine_id = $${params.length + 1}`);
      params.push(machine_id);
    }

    if (start_date) {
      conditions.push(`sd.timestamp >= $${params.length + 1}`);
      params.push(start_date);
    }

    if (end_date) {
      conditions.push(`sd.timestamp <= $${params.length + 1}`);
      params.push(end_date);
    }

    if (conditions.length > 0) {
      dataQuery += ' AND ' + conditions.join(' AND ');
    }

    dataQuery += ' ORDER BY sd.timestamp DESC LIMIT 10';

    const dataResult = await pool.query(dataQuery, params);
    const data = dataResult.rows;

    // Format data for LLM
    const formattedData = data.map(row => ({
      timestamp: row.timestamp,
      machine: row.machine_name,
      machine_type: row.machine_type,
      sensors: {
        vibration: row.vibration,
        temperature: row.temperature,
        pressure: row.pressure,
        flow_rate: row.flow_rate,
        rotational_speed: row.rotational_speed,
      },
      prediction: row.prediction_result,
    }));

    // Call Ollama LLM
    console.log(`Analyze the following manufacturing sensor data and answer the query: "${query}"

Data:
${JSON.stringify(formattedData, null, 2)}

Please provide insights, trends, and answer the specific query.`);
    console.log(formattedData.length);
    const llmResponse = await axios.post(`${process.env.OLLAMA_BASE_URL}/api/generate`, {
      model: 'gemma3:4b', // or any available model
      prompt: `Analyze the following manufacturing sensor data and answer the query: "${query}"

Data:
${JSON.stringify(formattedData, null, 2)}

Please provide insights, trends, and answer the specific query.`,
      stream: false,
      timeout: 100000000,
    });

    res.json({ analysis: llmResponse.data.response, data_count: data.length });
  } catch (error) {
    logger.error('Analyze data error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


// Helper functions
async function callPredictionModel(sensorData, machineType) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, '../models/predict.py');
    const pythonProcess = spawn('/opt/venv/bin/python3', [scriptPath, JSON.stringify(sensorData), machineType]);

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout.trim());
          logger.info("Prediction result:", result);
          resolve({
            result: {
              anomaly: result.anomaly,
              predicted_value: result.predicted_value
            },
            confidence: result.confidence,
            model_version: result.model_version,
          });
        } catch (parseError) {
          logger.error('Error parsing prediction result:', parseError);
          reject(new Error('Failed to parse prediction result'));
        }
      } else {
        logger.error('Prediction script error:', stderr);
        reject(new Error(`Prediction script failed: ${stderr}`));
      }
    });

    pythonProcess.on('error', (error) => {
      logger.error('Error running prediction script:', error);
      reject(error);
    });
  });
}


async function createAlert(machineId, type, message, severity) {
  try {
    await pool.query(
      'INSERT INTO alerts (machine_id, alert_type, message, severity) VALUES ($1, $2, $3, $4)',
      [machineId, type, message, severity]
    );

    // Here you could add email notification logic
    logger.info(`Alert created: ${message}`);
  } catch (error) {
    logger.error('Create alert error:', error);
  }
}

// Scheduled sensor data collection (every 5 seconds)
cron.schedule('*/5 * * * * *', async () => {
  try{
    // Get all machines
    const machinesResult = await pool.query('SELECT id FROM machines');

    const machines = machinesResult.rows;

    for (const machine of machines) {
      // Simulate sensor data collection
      const sensorData = {
        machine_id: machine.id,
        vibration: Math.random() * 10,
        temperature: 20 + Math.random() * 30,
        pressure: 1 + Math.random() * 5,
        flow_rate: 10 + Math.random() * 20,
        rotational_speed: 1000 + Math.random() * 2000,
      };

      // Insert sensor data
      const sensorResult = await pool.query(
        'INSERT INTO sensor_data (machine_id, vibration, temperature, pressure, flow_rate, rotational_speed) VALUES ($1, $2, $3, $4, $5, $6) RETURNING *',
        [sensorData.machine_id, sensorData.vibration, sensorData.temperature, sensorData.pressure, sensorData.flow_rate, sensorData.rotational_speed]
      );

      const insertedData = sensorResult.rows[0];

      // Get machine type
      const machineTypeResult = await pool.query(`
        SELECT mt.name as machine_type
        FROM machines m
        JOIN machine_types mt ON m.machine_type_id = mt.id
        WHERE m.id = $1
      `, [machine.id]);

      const machineType = machineTypeResult.rows[0].machine_type;

      // Call prediction
      const prediction = await callPredictionModel(insertedData, machineType);

      // Store prediction
      await pool.query(
        'INSERT INTO predictions (sensor_data_id, prediction_result, confidence_score, model_version) VALUES ($1, $2, $3, $4)',
        [insertedData.id, JSON.stringify(prediction.result), prediction.confidence, prediction.model_version]
      );

      // Check for anomalies
      if (prediction.result.anomaly) {
        await createAlert(machine.id, 'Anomaly Detected', `Anomaly detected for machine ${machine.id}`, 'high');
      }
    }

    logger.info(`Collected sensor data for ${machines.length} machines`);
  } catch (error) {
    logger.error('Scheduled sensor data collection error:', error);
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// Start server
app.listen(PORT, () => {
  logger.info(`Server running on port ${PORT}`);
});

module.exports = app;
