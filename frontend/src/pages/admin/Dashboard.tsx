import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import Navbar from '../../components/shared/Navbar';
import { adminAPI } from '../../services/api';
import type { Analytics } from '../../types';
import {
  UsersIcon,
  ChatBubbleLeftRightIcon,
  CalendarIcon,
  ClockIcon,
  ArrowTrendingUpIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';

const Dashboard: React.FC = () => {
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    try {
      const data = await adminAPI.getAnalytics();
      setAnalytics(data);
    } catch (error) {
      console.error('Error loading analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  const stats = [
    {
      name: 'Total Users',
      value: analytics?.total_users || 0,
      icon: UsersIcon,
      color: 'from-blue-500 to-blue-600',
      bgColor: 'bg-blue-50',
      textColor: 'text-blue-600',
    },
    {
      name: 'Chat Sessions',
      value: analytics?.total_sessions || 0,
      icon: ChatBubbleLeftRightIcon,
      color: 'from-purple-500 to-purple-600',
      bgColor: 'bg-purple-50',
      textColor: 'text-purple-600',
    },
    {
      name: 'Total Bookings',
      value: analytics?.total_bookings || 0,
      icon: CalendarIcon,
      color: 'from-green-500 to-green-600',
      bgColor: 'bg-green-50',
      textColor: 'text-green-600',
    },
    {
      name: 'Pending Bookings',
      value: analytics?.pending_bookings || 0,
      icon: ClockIcon,
      color: 'from-yellow-500 to-yellow-600',
      bgColor: 'bg-yellow-50',
      textColor: 'text-yellow-600',
    },
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <div className="flex items-center justify-center h-screen">
          <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-primary-600"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Admin Dashboard</h1>
          <p className="text-gray-600 mt-2">
            Welcome back! Here's an overview of your platform.
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {stats.map((stat) => {
            const Icon = stat.icon;
            return (
              <div
                key={stat.name}
                className="bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition-shadow"
              >
                <div className="flex items-center justify-between mb-4">
                  <div className={`w-12 h-12 ${stat.bgColor} rounded-lg flex items-center justify-center`}>
                    <Icon className={`h-6 w-6 ${stat.textColor}`} />
                  </div>
                  <ArrowTrendingUpIcon className="h-5 w-5 text-green-500" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-1">
                  {stat.value.toLocaleString()}
                </h3>
                <p className="text-sm text-gray-600">{stat.name}</p>
              </div>
            );
          })}
        </div>

        {/* Activity Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Recent Activity */}
          <div className="bg-white rounded-xl shadow-md p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">
              Platform Overview
            </h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center">
                  <UsersIcon className="h-8 w-8 text-blue-600 mr-3" />
                  <div>
                    <p className="font-medium text-gray-900">Active Users</p>
                    <p className="text-sm text-gray-600">
                      Currently using the platform
                    </p>
                  </div>
                </div>
                <span className="text-2xl font-bold text-blue-600">
                  {analytics?.active_users || 0}
                </span>
              </div>

              <div className="flex items-center justify-between p-4 bg-purple-50 rounded-lg">
                <div className="flex items-center">
                  <ChatBubbleLeftRightIcon className="h-8 w-8 text-purple-600 mr-3" />
                  <div>
                    <p className="font-medium text-gray-900">Total Sessions</p>
                    <p className="text-sm text-gray-600">
                      All-time chat sessions
                    </p>
                  </div>
                </div>
                <span className="text-2xl font-bold text-purple-600">
                  {analytics?.total_sessions || 0}
                </span>
              </div>

              <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
                <div className="flex items-center">
                  <CheckCircleIcon className="h-8 w-8 text-green-600 mr-3" />
                  <div>
                    <p className="font-medium text-gray-900">Bookings</p>
                    <p className="text-sm text-gray-600">
                      Confirmed appointments
                    </p>
                  </div>
                </div>
                <span className="text-2xl font-bold text-green-600">
                  {(analytics?.total_bookings || 0) - (analytics?.pending_bookings || 0)}
                </span>
              </div>
            </div>
          </div>

          {/* System Status */}
          <div className="bg-white rounded-xl shadow-md p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">
              System Status
            </h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 border-l-4 border-green-500 bg-green-50 rounded">
                <div>
                  <p className="font-medium text-gray-900">Database</p>
                  <p className="text-sm text-gray-600">MongoDB Connected</p>
                </div>
                <span className="px-3 py-1 bg-green-500 text-white text-sm font-medium rounded-full">
                  Healthy
                </span>
              </div>

              <div className="flex items-center justify-between p-4 border-l-4 border-green-500 bg-green-50 rounded">
                <div>
                  <p className="font-medium text-gray-900">AI Models</p>
                  <p className="text-sm text-gray-600">Facial + Text + LLM</p>
                </div>
                <span className="px-3 py-1 bg-green-500 text-white text-sm font-medium rounded-full">
                  Active
                </span>
              </div>

              <div className="flex items-center justify-between p-4 border-l-4 border-green-500 bg-green-50 rounded">
                <div>
                  <p className="font-medium text-gray-900">API Server</p>
                  <p className="text-sm text-gray-600">FastAPI Backend</p>
                </div>
                <span className="px-3 py-1 bg-green-500 text-white text-sm font-medium rounded-full">
                  Running
                </span>
              </div>

              <div className="flex items-center justify-between p-4 border-l-4 border-yellow-500 bg-yellow-50 rounded">
                <div>
                  <p className="font-medium text-gray-900">Pending Actions</p>
                  <p className="text-sm text-gray-600">Bookings awaiting review</p>
                </div>
                <span className="px-3 py-1 bg-yellow-500 text-white text-sm font-medium rounded-full">
                  {analytics?.pending_bookings || 0}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-gradient-to-r from-primary-600 to-primary-500 rounded-xl shadow-md p-8 text-white">
          <h2 className="text-2xl font-bold mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            
            <Link 
              to="/admin/sessions"
              className="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg p-4 transition-all"
            >
              <ChatBubbleLeftRightIcon className="h-8 w-8 mb-2" />
              <h3 className="font-semibold mb-1">View Sessions</h3>
              <p className="text-sm text-primary-100">
                Monitor all chat sessions
              </p>
            </Link>
            
            <Link 
              to="/admin/bookings"
              className="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg p-4 transition-all"
            >
              <CalendarIcon className="h-8 w-8 mb-2" />
              <h3 className="font-semibold mb-1">Manage Bookings</h3>
              <p className="text-sm text-primary-100">
                Review and approve bookings
              </p>
            </Link>
            
            <Link 
              to="/admin/counselors"
              className="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg p-4 transition-all"
            >
              <UsersIcon className="h-8 w-8 mb-2" />
              <h3 className="font-semibold mb-1">Manage Counselors</h3>
              <p className="text-sm text-primary-100">
                Add or remove counselors
              </p>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;