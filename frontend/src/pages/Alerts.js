
import React, { useState, useEffect } from 'react';
import { Table, Tag, Button, Space, Modal, Form, Input, Select, message } from 'antd';
import axios from 'axios';
import moment from 'moment';

const { Option } = Select;

const Alerts = () => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [form] = Form.useForm();

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchAlerts = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('/alerts', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setAlerts(response.data);
    } catch (error) {
      message.error('Failed to fetch alerts');
    } finally {
      setLoading(false);
    }
  };

  const handleResolve = async (alertId) => {
    try {
      const token = localStorage.getItem('token');
      await axios.patch(`/alerts/${alertId}/resolve`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      message.success('Alert resolved');
      fetchAlerts();
    } catch (error) {
      message.error('Failed to resolve alert');
    }
  };

  const handleViewDetails = (alert) => {
    setSelectedAlert(alert);
    setIsModalVisible(true);
  };

  const columns = [
    {
      title: 'Machine',
      dataIndex: 'machine_name',
      key: 'machine_name',
    },
    {
      title: 'Type',
      dataIndex: 'alert_type',
      key: 'alert_type',
    },
    {
      title: 'Message',
      dataIndex: 'message',
      key: 'message',
      ellipsis: true,
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity) => {
        const color = severity === 'high' ? 'red' : severity === 'medium' ? 'orange' : 'yellow';
        return <Tag color={color}>{severity.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status === 'active' ? 'red' : 'green'}>
          {status.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (timestamp) => moment(timestamp).format('YYYY-MM-DD HH:mm:ss'),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button type="link" onClick={() => handleViewDetails(record)}>
            View
          </Button>
          {record.status === 'active' && (
            <Button type="link" onClick={() => handleResolve(record.id)}>
              Resolve
            </Button>
          )}
        </Space>
      ),
    },
  ];

  return (
    <div>
      <h1>Alerts</h1>

      <Table
        columns={columns}
        dataSource={alerts}
        rowKey="id"
        loading={loading}
        pagination={{
          pageSize: 20,
          showSizeChanger: true,
          showQuickJumper: true,
        }}
      />

      <Modal
        title="Alert Details"
        open={isModalVisible}
        onCancel={() => setIsModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setIsModalVisible(false)}>
            Close
          </Button>,
          selectedAlert && selectedAlert.status === 'active' && (
            <Button
              key="resolve"
              type="primary"
              onClick={() => {
                handleResolve(selectedAlert.id);
                setIsModalVisible(false);
              }}
            >
              Resolve Alert
            </Button>
          ),
        ]}
        width={600}
      >
        {selectedAlert && (
          <div>
            <p><strong>Machine:</strong> {selectedAlert.machine_name}</p>
            <p><strong>Type:</strong> {selectedAlert.alert_type}</p>
            <p><strong>Message:</strong> {selectedAlert.message}</p>
            <p><strong>Severity:</strong> <Tag color={selectedAlert.severity === 'high' ? 'red' : 'orange'}>{selectedAlert.severity.toUpperCase()}</Tag></p>
            <p><strong>Status:</strong> <Tag color={selectedAlert.status === 'active' ? 'red' : 'green'}>{selectedAlert.status.toUpperCase()}</Tag></p>
            <p><strong>Created:</strong> {moment(selectedAlert.created_at).format('YYYY-MM-DD HH:mm:ss')}</p>
            {selectedAlert.notified_at && (
              <p><strong>Notified:</strong> {moment(selectedAlert.notified_at).format('YYYY-MM-DD HH:mm:ss')}</p>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default Alerts;
