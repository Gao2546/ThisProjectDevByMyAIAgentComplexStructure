
import React, { useState, useEffect } from 'react';
import { Table, Button, Modal, Form, Input, Select, message, Space, Tag } from 'antd';
import axios from 'axios';

const { Option } = Select;

const Machines = () => {
  const [machines, setMachines] = useState([]);
  const [machineTypes, setMachineTypes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [editingMachine, setEditingMachine] = useState(null);
  const [form] = Form.useForm();

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };

      const [machinesRes, typesRes] = await Promise.all([
        axios.get('/machines', { headers }),
        axios.get('/machine-types', { headers })
      ]);

      setMachines(machinesRes.data);
      setMachineTypes(typesRes.data);
    } catch (error) {
      message.error('Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  const handleAdd = () => {
    setEditingMachine(null);
    form.resetFields();
    setIsModalVisible(true);
  };

  const handleEdit = (record) => {
    setEditingMachine(record);
    form.setFieldsValue(record);
    setIsModalVisible(true);
  };

  const handleDelete = async (id) => {
    try {
      const token = localStorage.getItem('token');
      await axios.delete(`/machines/${id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      message.success('Machine deleted successfully');
      fetchData();
    } catch (error) {
      message.error('Failed to delete machine');
    }
  };

  const handleModalOk = async () => {
    try {
      const values = await form.validateFields();
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };

      if (editingMachine) {
        await axios.put(`/machines/${editingMachine.id}`, values, { headers });
        message.success('Machine updated successfully');
      } else {
        await axios.post('/machines', values, { headers });
        message.success('Machine added successfully');
      }

      setIsModalVisible(false);
      fetchData();
    } catch (error) {
      message.error('Operation failed');
    }
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Machine Type',
      dataIndex: 'machine_type_name',
      key: 'machine_type_name',
    },
    {
      title: 'Location',
      dataIndex: 'location',
      key: 'location',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status === 'active' ? 'green' : 'red'}>
          {status.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button type="link" onClick={() => handleEdit(record)}>
            Edit
          </Button>
          <Button type="link" danger onClick={() => handleDelete(record.id)}>
            Delete
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h1>Machines</h1>
        <Button type="primary" onClick={handleAdd}>
          Add Machine
        </Button>
      </div>

      <Table
        columns={columns}
        dataSource={machines}
        rowKey="id"
        loading={loading}
      />

      <Modal
        title={editingMachine ? 'Edit Machine' : 'Add Machine'}
        open={isModalVisible}
        onOk={handleModalOk}
        onCancel={() => setIsModalVisible(false)}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="machine_type_id"
            label="Machine Type"
            rules={[{ required: true, message: 'Please select machine type' }]}
          >
            <Select placeholder="Select machine type">
              {machineTypes.map(type => (
                <Option key={type.id} value={type.id}>
                  {type.name}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="name"
            label="Machine Name"
            rules={[{ required: true, message: 'Please input machine name' }]}
          >
            <Input />
          </Form.Item>

          <Form.Item
            name="location"
            label="Location"
          >
            <Input />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default Machines;
