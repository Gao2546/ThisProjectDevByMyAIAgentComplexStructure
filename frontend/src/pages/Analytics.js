
import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Form, Select, DatePicker, Button, Input, Spin, message } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import axios from 'axios';
import moment from 'moment';

const { Option } = Select;
const { TextArea } = Input;
const { RangePicker } = DatePicker;

const Analytics = () => {
  const [sensorData, setSensorData] = useState([]);
  const [machines, setMachines] = useState([]);
  const [loading, setLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    fetchMachines();
  }, []);

  const fetchMachines = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('/machines', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMachines(response.data);
    } catch (error) {
      message.error('Failed to fetch machines');
    }
  };

  const fetchSensorData = async (values) => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const params = {};

      if (values.machine_id) params.machine_id = values.machine_id;
      if (values.dateRange) {
        params.start_date = values.dateRange[0].format('YYYY-MM-DD');
        params.end_date = values.dateRange[1].format('YYYY-MM-DD');
      }

      const response = await axios.get('/sensor-data', {
        headers: { Authorization: `Bearer ${token}` },
        params: { ...params, limit: 1000 }
      });

      setSensorData(response.data);
    } catch (error) {
      message.error('Failed to fetch sensor data');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    const values = form.getFieldsValue();
    if (!values.query) {
      message.error('Please enter a query');
      return;
    }

    setAnalyzing(true);
    try {
      const token = localStorage.getItem('token');
      const payload = {
        query: values.query,
      };

      if (values.machine_id) payload.machine_id = values.machine_id;
      if (values.dateRange) {
        payload.start_date = values.dateRange[0].format('YYYY-MM-DD');
        payload.end_date = values.dateRange[1].format('YYYY-MM-DD');
      }

      const response = await axios.post('/analyze-data', payload, {
        headers: { Authorization: `Bearer ${token}` }
      });

      setAnalysisResult(response.data.analysis);
    } catch (error) {
      message.error('Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const chartData = sensorData.slice(-100).map(item => ({
    time: moment(item.timestamp).format('MM-DD HH:mm'),
    temperature: item.temperature,
    vibration: item.vibration,
    pressure: item.pressure,
    flow_rate: item.flow_rate,
    rotational_speed: item.rotational_speed,
  }));

  const correlationData = sensorData.map(item => ({
    temperature: item.temperature,
    vibration: item.vibration,
  }));

  return (
    <div>
      <h1>Analytics</h1>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card title="Data Filters">
            <Form form={form} layout="inline" onFinish={fetchSensorData}>
              <Form.Item name="machine_id" label="Machine">
                <Select placeholder="Select machine" style={{ width: 200 }}>
                  <Option value="">All Machines</Option>
                  {machines.map(machine => (
                    <Option key={machine.id} value={machine.id}>
                      {machine.name}
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item name="dateRange" label="Date Range">
                <RangePicker />
              </Form.Item>

              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading}>
                  Load Data
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </Col>
      </Row>

      {sensorData.length > 0 && (
        <>
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={24}>
              <Card title="Sensor Trends" style={{ height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="temperature" stroke="#8884d8" name="Temperature (°C)" />
                    <Line type="monotone" dataKey="vibration" stroke="#82ca9d" name="Vibration" />
                    <Line type="monotone" dataKey="pressure" stroke="#ffc658" name="Pressure" />
                    <Line type="monotone" dataKey="flow_rate" stroke="#ff7300" name="Flow Rate" />
                    <Line type="monotone" dataKey="rotational_speed" stroke="#00ff00" name="Rotational Speed" />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>

          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={12}>
              <Card title="Temperature vs Vibration Correlation" style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart data={correlationData}>
                    <CartesianGrid />
                    <XAxis type="number" dataKey="temperature" name="Temperature" />
                    <YAxis type="number" dataKey="vibration" name="Vibration" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Data Points" data={correlationData} fill="#8884d8" />
                  </ScatterChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="Data Statistics">
                <p><strong>Total Data Points:</strong> {sensorData.length}</p>
                <p><strong>Average Temperature:</strong> {(sensorData.reduce((sum, item) => sum + item.temperature, 0) / sensorData.length).toFixed(2)}°C</p>
                <p><strong>Average Vibration:</strong> {(sensorData.reduce((sum, item) => sum + item.vibration, 0) / sensorData.length).toFixed(2)}</p>
                <p><strong>Average Pressure:</strong> {(sensorData.reduce((sum, item) => sum + item.pressure, 0) / sensorData.length).toFixed(2)}</p>
                <p><strong>Time Range:</strong> {moment(sensorData[0]?.timestamp).format('YYYY-MM-DD HH:mm')} to {moment(sensorData[sensorData.length - 1]?.timestamp).format('YYYY-MM-DD HH:mm')}</p>
              </Card>
            </Col>
          </Row>
        </>
      )}

      <Row gutter={16}>
        <Col span={24}>
          <Card title="AI-Powered Analysis">
            <Form form={form} layout="vertical">
              <Form.Item name="query" label="Analysis Query">
                <TextArea
                  placeholder="Ask questions about your sensor data (e.g., 'What are the temperature trends?', 'Are there any anomalies?', 'How does vibration correlate with temperature?')"
                  rows={4}
                />
              </Form.Item>

              <Form.Item>
                <Button type="primary" onClick={handleAnalyze} loading={analyzing}>
                  Analyze with AI
                </Button>
              </Form.Item>
            </Form>

            {analysisResult && (
              <div style={{ marginTop: 16, padding: 16, background: '#f5f5f5', borderRadius: 4 }}>
                <h3>Analysis Result:</h3>
                <p style={{ whiteSpace: 'pre-wrap' }}>{analysisResult}</p>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Analytics;
