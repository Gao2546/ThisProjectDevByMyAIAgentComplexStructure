
import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Alert, Spin } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import moment from 'moment';

const Dashboard = () => {
  const [machines, setMachines] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [sensorData, setSensorData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };

      const [machinesRes, alertsRes, sensorRes] = await Promise.all([
        axios.get('/machines', { headers }),
        axios.get('/alerts', { headers }),
        axios.get('/sensor-data?limit=100', { headers })
      ]);

      setMachines(machinesRes.data);
      setAlerts(alertsRes.data.slice(0, 5)); // Show latest 5 alerts
      setSensorData(sensorRes.data);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const activeAlerts = alerts.filter(alert => alert.status === 'active');
  const chartData = sensorData.slice(-20).map(item => ({
    time: moment(item.timestamp).format('HH:mm:ss'),
    temperature: item.temperature,
    vibration: item.vibration,
    pressure: item.pressure,
  }));

  if (loading) {
    return <Spin size="large" style={{ display: 'block', margin: '50px auto' }} />;
  }

  return (
    <div>
      <h1>Manufacturing Dashboard</h1>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic title="Total Machines" value={machines.length} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="Active Machines" value={machines.filter(m => m.status === 'active').length} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="Active Alerts" value={activeAlerts.length} suffix={<span style={{ color: 'red' }}>⚠️</span>} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="Data Points (Last 24h)" value={sensorData.length} />
          </Card>
        </Col>
      </Row>

      {activeAlerts.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <h2>Active Alerts</h2>
          {activeAlerts.map(alert => (
            <Alert
              key={alert.id}
              message={`${alert.machine_name}: ${alert.message}`}
              type={alert.severity === 'high' ? 'error' : 'warning'}
              showIcon
              style={{ marginBottom: 8 }}
            />
          ))}
        </div>
      )}

      <Row gutter={16}>
        <Col span={24}>
          <Card title="Sensor Trends (Last 20 Readings)" style={{ height: 400 }}>
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
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
