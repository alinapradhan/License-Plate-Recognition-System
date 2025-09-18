import React, { useState, useEffect, useRef } from 'react';
import { 
  Camera, 
  Square, 
  Play, 
  Pause, 
  Download,
  AlertCircle,
  Eye,
  Settings,
  Maximize
} from 'lucide-react';
import apiService from '../utils/api';

export default function CameraPage() {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    totalDetections: 0,
    validPlates: 0,
    fps: 0
  });
  const [isFullscreen, setIsFullscreen] = useState(false);

  const wsRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());
  const containerRef = useRef(null);

  useEffect(() => {
    return () => {
      disconnectWebSocket();
    };
  }, []);

  const connectWebSocket = () => {
    if (wsRef.current) {
      disconnectWebSocket();
    }

    setIsLoading(true);
    setError(null);

    try {
      wsRef.current = apiService.connectWebSocket(
        handleWebSocketMessage,
        handleWebSocketError,
        handleWebSocketClose
      );
    } catch (error) {
      handleWebSocketError(error);
    }
  };

  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
    setIsLoading(false);
    setCurrentFrame(null);
    setDetections([]);
  };

  const handleWebSocketMessage = (data) => {
    if (!isConnected) {
      setIsConnected(true);
      setIsLoading(false);
    }

    // Update current frame
    if (data.frame) {
      setCurrentFrame(data.frame);
    }

    // Update detections
    if (data.plates) {
      const formattedDetections = data.plates.map(plate => ({
        ...plate,
        id: `${Date.now()}_${Math.random()}`,
        timestamp: data.timestamp
      }));
      setDetections(formattedDetections);
    }

    // Update stats
    setStats(prev => ({
      totalDetections: prev.totalDetections + (data.detections || 0),
      validPlates: prev.validPlates + (data.valid_plates || 0),
      fps: calculateFPS()
    }));
  };

  const handleWebSocketError = (error) => {
    console.error('WebSocket error:', error);
    setError('Failed to connect to camera service. Please check your camera permissions and try again.');
    setIsLoading(false);
    setIsConnected(false);
  };

  const handleWebSocketClose = (event) => {
    setIsConnected(false);
    setIsLoading(false);
    
    if (event.code !== 1000) { // Not a normal closure
      setError('Connection to camera service lost. Please reconnect.');
    }
  };

  const calculateFPS = () => {
    frameCountRef.current++;
    const now = Date.now();
    const elapsed = now - lastFpsUpdateRef.current;
    
    if (elapsed >= 1000) { // Update FPS every second
      const fps = (frameCountRef.current * 1000) / elapsed;
      frameCountRef.current = 0;
      lastFpsUpdateRef.current = now;
      return Math.round(fps);
    }
    
    return stats.fps;
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen().then(() => {
        setIsFullscreen(true);
      }).catch((error) => {
        console.error('Failed to enter fullscreen:', error);
      });
    } else {
      document.exitFullscreen().then(() => {
        setIsFullscreen(false);
      });
    }
  };

  const downloadCurrentFrame = () => {
    if (currentFrame) {
      const link = document.createElement('a');
      link.href = currentFrame;
      link.download = `license_plate_detection_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const formatBoundingBox = (bbox) => {
    if (!bbox || bbox.length !== 4) return null;
    
    const [x1, y1, x2, y2] = bbox;
    return {
      left: `${x1}px`,
      top: `${y1}px`,
      width: `${x2 - x1}px`,
      height: `${y2 - y1}px`
    };
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Live Camera Detection</h1>
          <p className="text-gray-600">Real-time license plate detection and recognition</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <div className={`status-indicator ${isConnected ? 'online' : 'offline'}`}></div>
          <span className="text-sm text-gray-600">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {!isConnected ? (
              <button
                onClick={connectWebSocket}
                disabled={isLoading}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>
                    <div className="loading-spinner w-4 h-4 mr-2"></div>
                    Connecting...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start Camera
                  </>
                )}
              </button>
            ) : (
              <button
                onClick={disconnectWebSocket}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700"
              >
                <Pause className="h-4 w-4 mr-2" />
                Stop Camera
              </button>
            )}

            {currentFrame && (
              <button
                onClick={downloadCurrentFrame}
                className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
              >
                <Download className="h-4 w-4 mr-2" />
                Save Frame
              </button>
            )}
          </div>

          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <div className="flex items-center">
              <Eye className="h-4 w-4 mr-1" />
              <span>{stats.totalDetections} total</span>
            </div>
            <div className="flex items-center">
              <Square className="h-4 w-4 mr-1" />
              <span>{stats.validPlates} valid</span>
            </div>
            <div className="flex items-center">
              <Settings className="h-4 w-4 mr-1" />
              <span>{stats.fps} FPS</span>
            </div>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <AlertCircle className="h-5 w-5 text-red-400" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Connection Error</h3>
              <p className="mt-1 text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Camera Feed */}
      <div 
        ref={containerRef}
        className={`bg-white rounded-lg shadow overflow-hidden ${
          isFullscreen ? 'fixed inset-0 z-50 flex items-center justify-center bg-black' : ''
        }`}
      >
        <div className="relative">
          {currentFrame ? (
            <div className="camera-container relative">
              <img
                src={currentFrame}
                alt="Live camera feed"
                className="camera-feed max-w-full h-auto"
              />
              
              {/* Bounding box overlays */}
              {detections.map((detection, index) => {
                const style = formatBoundingBox(detection.bbox);
                if (!style) return null;
                
                return (
                  <div
                    key={detection.id || index}
                    className={`plate-overlay ${!detection.valid ? 'invalid' : ''}`}
                    style={style}
                  >
                    <div className={`plate-label ${!detection.valid ? 'invalid' : ''}`}>
                      {detection.text || 'Detecting...'}
                      {detection.confidence && (
                        <span className="ml-1">
                          ({(detection.confidence * 100).toFixed(1)}%)
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
              
              {/* Fullscreen toggle */}
              <button
                onClick={toggleFullscreen}
                className="absolute top-4 right-4 p-2 bg-black bg-opacity-50 text-white rounded-lg hover:bg-opacity-70 transition-all"
              >
                <Maximize className="h-4 w-4" />
              </button>
            </div>
          ) : (
            <div className="aspect-video bg-gray-100 flex items-center justify-center">
              <div className="text-center">
                <Camera className="mx-auto h-16 w-16 text-gray-400 mb-4" />
                <h3 className="text-lg font-medium text-gray-900">
                  {isLoading ? 'Connecting to camera...' : 'Camera feed not active'}
                </h3>
                <p className="text-gray-600 mt-2">
                  {isLoading 
                    ? 'Please allow camera access when prompted'
                    : 'Click "Start Camera" to begin live detection'
                  }
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Live Detections */}
      {detections.length > 0 && (
        <div className="bg-white rounded-lg shadow">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Current Detections ({detections.length})
            </h3>
            
            <div className="space-y-3">
              {detections.map((detection, index) => (
                <div
                  key={detection.id || index}
                  className={`detection-card ${!detection.valid ? 'invalid' : ''}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${
                        detection.valid ? 'bg-green-500' : 'bg-yellow-500'
                      }`} />
                      <div>
                        <p className="text-sm font-medium text-gray-900">
                          {detection.text || 'Processing...'}
                        </p>
                        <p className="text-xs text-gray-500">
                          Box: [{detection.bbox?.join(', ') || 'Unknown'}]
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium text-gray-900">
                        {detection.confidence 
                          ? `${(detection.confidence * 100).toFixed(1)}%`
                          : 'Processing...'
                        }
                      </p>
                      <p className="text-xs text-gray-500">Confidence</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex">
          <Camera className="h-5 w-5 text-blue-400" />
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">How to use live detection</h3>
            <div className="mt-2 text-sm text-blue-700">
              <ul className="list-disc list-inside space-y-1">
                <li>Ensure your camera is connected and working properly</li>
                <li>Allow camera access when prompted by your browser</li>
                <li>Position license plates clearly in view for best results</li>
                <li>Valid detections will be highlighted in green, processing in yellow</li>
                <li>Click the fullscreen button for better visibility</li>
                <li>Use "Save Frame" to download the current detection result</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}