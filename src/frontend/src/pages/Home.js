import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Camera, 
  Upload, 
  History, 
  Shield, 
  Zap, 
  Eye, 
  Database,
  TrendingUp,
  Clock,
  CheckCircle
} from 'lucide-react';
import apiService from '../utils/api';

export default function Home() {
  const [stats, setStats] = useState({
    totalDetections: 0,
    validPlates: 0,
    recentDetections: 0,
    systemHealth: 'healthy'
  });
  const [recentDetections, setRecentDetections] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      // Fetch recent detections
      const historyResponse = await apiService.getHistory(10, 0);
      if (historyResponse.success) {
        setRecentDetections(historyResponse.detections || []);
        
        // Calculate stats from recent detections
        const validCount = historyResponse.detections?.filter(d => d.text && d.text.length > 0).length || 0;
        setStats(prev => ({
          ...prev,
          totalDetections: historyResponse.total || 0,
          validPlates: validCount,
          recentDetections: historyResponse.detections?.length || 0
        }));
      }
      
      // Check system health
      const healthResponse = await apiService.checkHealth();
      if (healthResponse.status) {
        setStats(prev => ({
          ...prev,
          systemHealth: healthResponse.status
        }));
      }
      
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const features = [
    {
      name: 'Real-time Detection',
      description: 'Live camera feed with instant license plate detection and recognition',
      icon: Camera,
      href: '/camera',
      color: 'text-blue-600 bg-blue-100'
    },
    {
      name: 'Upload & Process',
      description: 'Upload images for batch processing and text extraction',
      icon: Upload,
      href: '/upload',
      color: 'text-green-600 bg-green-100'
    },
    {
      name: 'Detection History',
      description: 'Browse, search, and export your detection history',
      icon: History,
      href: '/history',
      color: 'text-purple-600 bg-purple-100'
    },
  ];

  const statCards = [
    {
      name: 'Total Detections',
      value: stats.totalDetections.toLocaleString(),
      icon: Eye,
      color: 'text-blue-600'
    },
    {
      name: 'Valid Plates',
      value: stats.validPlates.toLocaleString(),
      icon: CheckCircle,
      color: 'text-green-600'
    },
    {
      name: 'Recent Activity',
      value: stats.recentDetections.toLocaleString(),
      icon: TrendingUp,
      color: 'text-purple-600'
    },
    {
      name: 'System Status',
      value: stats.systemHealth === 'healthy' ? 'Online' : 'Offline',
      icon: Database,
      color: stats.systemHealth === 'healthy' ? 'text-green-600' : 'text-red-600'
    }
  ];

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown';
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner w-8 h-8"></div>
        <span className="ml-3 text-gray-600">Loading dashboard...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Welcome Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg shadow-lg p-6 text-white">
        <div className="flex items-center">
          <Shield className="h-12 w-12 mr-4" />
          <div>
            <h1 className="text-3xl font-bold">License Plate Recognition System</h1>
            <p className="text-blue-100 mt-1">
              Advanced AI-powered license plate detection and OCR technology
            </p>
          </div>
        </div>
        
        <div className="mt-6 flex items-center space-x-6">
          <div className="flex items-center">
            <Zap className="h-5 w-5 mr-2" />
            <span className="text-sm">Real-time Processing</span>
          </div>
          <div className="flex items-center">
            <Shield className="h-5 w-5 mr-2" />
            <span className="text-sm">Encrypted Storage</span>
          </div>
          <div className="flex items-center">
            <Eye className="h-5 w-5 mr-2" />
            <span className="text-sm">High Accuracy Detection</span>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {statCards.map((item) => {
          const Icon = item.icon;
          return (
            <div key={item.name} className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <Icon className={`h-6 w-6 ${item.color}`} />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        {item.name}
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        {item.value}
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {features.map((feature) => {
          const Icon = feature.icon;
          return (
            <Link
              key={feature.name}
              to={feature.href}
              className="relative group bg-white p-6 focus-within:ring-2 focus-within:ring-inset focus-within:ring-blue-500 rounded-lg shadow hover:shadow-md transition-all duration-200"
            >
              <div>
                <span className={`rounded-lg inline-flex p-3 ${feature.color}`}>
                  <Icon className="h-6 w-6" />
                </span>
              </div>
              <div className="mt-8">
                <h3 className="text-lg font-medium text-gray-900 group-hover:text-blue-600">
                  {feature.name}
                </h3>
                <p className="mt-2 text-sm text-gray-500">
                  {feature.description}
                </p>
              </div>
              <span className="absolute top-6 right-6 text-gray-300 group-hover:text-gray-400">
                <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="m11.293 17.293 1.414 1.414L19.414 12l-6.707-6.707-1.414 1.414L15.586 11H5v2h10.586l-4.293 4.293z" />
                </svg>
              </span>
            </Link>
          );
        })}
      </div>

      {/* Recent Detections */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900 flex items-center">
              <Clock className="h-5 w-5 mr-2 text-gray-400" />
              Recent Detections
            </h3>
            <Link
              to="/history"
              className="text-sm font-medium text-blue-600 hover:text-blue-500"
            >
              View all →
            </Link>
          </div>
          
          {recentDetections.length > 0 ? (
            <div className="space-y-3">
              {recentDetections.slice(0, 5).map((detection) => (
                <div
                  key={detection.id}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${
                      detection.text && detection.text.length > 0 
                        ? 'bg-green-500' 
                        : 'bg-yellow-500'
                    }`} />
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        {detection.text || 'No text detected'}
                      </p>
                      <p className="text-xs text-gray-500">
                        {formatDate(detection.timestamp)}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-gray-500">
                      {detection.confidence ? `${(detection.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-6">
              <Eye className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No detections yet</h3>
              <p className="mt-1 text-sm text-gray-500">
                Start by uploading an image or using the live camera feed.
              </p>
              <div className="mt-6 flex justify-center space-x-3">
                <Link
                  to="/upload"
                  className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
                >
                  <Upload className="h-4 w-4 mr-2" />
                  Upload Image
                </Link>
                <Link
                  to="/camera"
                  className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                >
                  <Camera className="h-4 w-4 mr-2" />
                  Live Camera
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}