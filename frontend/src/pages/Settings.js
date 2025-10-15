
import React, { useState, useEffect } from 'react';
import { Card, Form, Input, Button, Select, Switch, message, Row, Col, Table, Modal, InputNumber } from 'antd';
import axios from 'axios';

const { Option } = Select;

const Settings = () => {
  const [machineTypes, setMachineTypes] = useState([]);
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [editingType, setEditingType] = useState(null);
  const [form] = Form.useForm();
  const [userForm] = Form.useForm();
  const [typeForm] = Form.useForm();

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };

      const [typesRes, usersRes] = await Promise.all([
        axios.get('/machine-types', { headers }),
        axios.get('/users', { headers })
      ]);

      setMachineTypes(typesRes.data);
      setUsers(usersRes.data);
    } catch (error) {
      message.error('Failed to fetch settings data');
    }
  };

  const handleAddMachineType = async () => {
    try {
      const values = await typeForm.validateFields();
      const token = localStorage.getItem('token');

      await axios.post('/machine-types', values, {
        headers: { Authorization: `Bearer ${token}` }
      });

      message.success('Machine type added successfully');
      typeForm.resetFields();
      fetchData();
    } catch (error) {
      message.error('Failed to add machine type');
    }
  };

  const handleAddUser = async () => {
    try {
      const values = await userForm.validateFields();
      const token = localStorage.getItem('token');

      await axios.post('/auth/register', values, {
        headers: { Authorization: `Bearer ${token}` }
      });

      message.success('User added successfully');
      userForm.resetFields();
      fetchData();
    } catch (error) {
      message.error('Failed to add user');
    }
  };

  const handleSimulateData = async () => {
    if (machineTypes.length === 0) {
      message.error('No machines available for simulation');
      return;
    }

    setLoading(true);
    try {
      const token = localStorage.getItem('token');

      // Get first active machine for simulation
      const machinesRes = await axios.get('/machines', {
        headers: { Authorization: `Bearer ${token}` }
      });

      const activeMachines = machinesRes.data.filter(m => m.status === 'active');

      if (activeMachines.length === 0) {
        message.error('No active machines found');
        return;
      }

      await axios.post('/simulate-sensor-data', {
        machine_id: activeMachines[0].id
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });

      message.success('Sensor data simulated successfully');
    } catch (error) {
      message.error('Simulation failed');
    } finally {
      setLoading(false);
    }
  };

  const machineTypeColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (timestamp) => new Date(timestamp).toLocaleDateString(),
    },
  ];

  const userColumns = [
    {
      title: 'Username',
      dataIndex: 'username',
      key: 'username',
    },
    {
      title: 'Email',
      dataIndex: 'email',
      key: 'email',
    },
    {
      title: 'Role',
      dataIndex: 'role',
      key: 'role',
      render: (role) => (
        <span style={{
          color: role === 'Admin' ? '#ff4d4f' : role === 'Agent' ? '#1890ff' : '#52c41a'
        }}>
          {role}
        </span>
      ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (timestamp) => new Date(timestamp).toLocaleDateString(),
    },
  ];

  return (
    <div>
      <h1>Settings</h1>

      <Row gutter={16}>
        <Col span={12}>
          <Card title="Machine Types" style={{ marginBottom: 16 }}>
            <Form form={typeForm} layout="inline" style={{ marginBottom: 16 }}>
              <Form.Item
                name="name"
                rules={[{ required: true, message: 'Please input machine type name' }]}
              >
                <Input placeholder="Machine Type Name" />
              </Form.Item>
              <Form.Item name="description">
                <Input placeholder="Description" />
              </Form.Item>
              <Form.Item>
                <Button type="primary" onClick={handleAddMachineType}>
                  Add Type
                </Button>
              </Form.Item>
            </Form>

            <Table
              columns={machineTypeColumns}
              dataSource={machineTypes}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </Card>
        </Col>

        <Col span={12}>
          <Card title="User Management" style={{ marginBottom: 16 }}>
            <Form form={userForm} layout="vertical" style={{ marginBottom: 16 }}>
              <Row gutter={8}>
                <Col span={12}>
                  <Form.Item
                    name="username"
                    rules={[{ required: true, message: 'Please input username' }]}
                  >
                    <Input placeholder="Username" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="email"
                    rules={[{ required: true, type: 'email', message: 'Please input valid email' }]}
                  >
                    <Input placeholder="Email" />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={8}>
                <Col span={12}>
                  <Form.Item
                    name="password"
                    rules={[{ required: true, message: 'Please input password' }]}
                  >
                    <Input.Password placeholder="Password" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="role"
                    rules={[{ required: true, message: 'Please select role' }]}
                  >
                    <Select placeholder="Select Role">
                      <Option value="System">System</Option>
                      <Option value="Agent">Agent</Option>
                      <Option value="Admin">Admin</Option>
                    </Select>
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item>
                <Button type="primary" onClick={handleAddUser}>
                  Add User
                </Button>
              </Form.Item>
            </Form>

            <Table
              columns={userColumns}
              dataSource={users}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card title="System Controls">
            <Row gutter={16}>
              <Col span={8}>
                <Card size="small" title="Data Simulation">
                  <p>Generate simulated sensor data for testing</p>
                  <Button
                    type="primary"
                    onClick={handleSimulateData}
                    loading={loading}
                    block
                  >
                    Simulate Data
                  </Button>
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small" title="System Health">
                  <p>Check system status and connections</p>
                  <Button type="default" block>
                    Health Check
                  </Button>
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small" title="Data Export">
                  <p>Export sensor data and reports</p>
                  <Button type="default" block>
                    Export Data
                  </Button>
                </Card>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Settings;
