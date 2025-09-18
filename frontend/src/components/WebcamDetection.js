import React, { useState, useRef, useCallback } from 'react';

const WebcamDetection = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [detections, setDetections] = useState([]);
  const [lastDetectionTime, setLastDetectionTime] = useState(null);
  const [stats, setStats] = useState({
    totalDetections: 0,
    uniquePlates: new Set(),
    averageConfidence: 0
  });
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);

  const startWebcam = useCallback(async () => {
    setError(null);
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'environment' // Prefer back camera on mobile
        }
      });
      
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
        
        // Start detection loop
        startDetectionLoop();
      }
    } catch (err) {
      setError('Failed to access camera: ' + err.message);
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    setIsStreaming(false);
  }, []);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (video && canvas) {
      const ctx = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      return new Promise((resolve) => {
        canvas.toBlob(resolve, 'image/jpeg', 0.8);
      });
    }
    
    return null;
  }, []);

  const processFrame = useCallback(async () => {
    if (!isStreaming || !videoRef.current) return;
    
    try {
      const blob = await captureFrame();
      if (!blob) return;
      
      const formData = new FormData();
      formData.append('file', blob, 'webcam_frame.jpg');
      formData.append('save_detection', 'true');
      formData.append('camera_source', 'webcam');
      
      const response = await fetch('/api/detect/image', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const results = await response.json();
        
        if (results && results.length > 0) {
          setDetections(prev => {
            const newDetections = [...results, ...prev].slice(0, 10); // Keep last 10
            return newDetections;
          });
          
          setLastDetectionTime(new Date());
          
          // Update stats
          setStats(prev => {
            const newUnique = new Set(prev.uniquePlates);
            results.forEach(result => {
              if (result.license_plate_text) {
                newUnique.add(result.license_plate_text);
              }
            });
            
            const totalNew = prev.totalDetections + results.length;
            const avgConf = results.reduce((sum, r) => sum + r.detection_confidence, 0) / results.length;
            
            return {
              totalDetections: totalNew,
              uniquePlates: newUnique,
              averageConfidence: (prev.averageConfidence + avgConf) / 2
            };
          });
        }
      }
    } catch (err) {
      console.error('Frame processing error:', err);
    }
  }, [isStreaming, captureFrame]);

  const startDetectionLoop = useCallback(() => {
    // Process frames every 2 seconds to avoid overwhelming the API
    detectionIntervalRef.current = setInterval(processFrame, 2000);
  }, [processFrame]);

  const clearDetections = () => {
    setDetections([]);
    setStats({
      totalDetections: 0,
      uniquePlates: new Set(),
      averageConfidence: 0
    });
  };

  const takeSnapshot = async () => {
    try {
      const blob = await captureFrame();
      if (blob) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `webcam_snapshot_${new Date().getTime()}.jpg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      }
    } catch (err) {
      setError('Failed to take snapshot: ' + err.message);
    }
  };

  return (
    <div className="webcam-detection">
      <h2>📹 Live Camera Detection</h2>
      
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}
      
      <div className="webcam-container">
        {/* Video Stream */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="webcam-video"
          style={{
            display: isStreaming ? 'block' : 'none',
            maxWidth: '100%',
            height: 'auto'
          }}
        />
        
        {/* Hidden canvas for frame capture */}
        <canvas
          ref={canvasRef}
          style={{ display: 'none' }}
        />
        
        {!isStreaming && (
          <div style={{
            width: '640px',
            height: '480px',
            maxWidth: '100%',
            background: '#f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '2px dashed #ccc',
            borderRadius: '8px'
          }}>
            <p style={{ color: '#666' }}>📹 Camera not active</p>
          </div>
        )}
        
        {/* Controls */}
        <div className="webcam-controls">
          {!isStreaming ? (
            <button className="btn btn-primary" onClick={startWebcam}>
              📹 Start Camera
            </button>
          ) : (
            <>
              <button className="btn btn-danger" onClick={stopWebcam}>
                ⏹️ Stop Camera
              </button>
              <button className="btn btn-secondary" onClick={takeSnapshot}>
                📸 Snapshot
              </button>
              <button className="btn btn-secondary" onClick={clearDetections}>
                🗑️ Clear Results
              </button>
            </>
          )}
        </div>
        
        {/* Detection Stats */}
        {isStreaming && (
          <div style={{
            background: '#f9f9f9',
            padding: '16px',
            borderRadius: '8px',
            margin: '20px 0',
            textAlign: 'left'
          }}>
            <h4>📊 Detection Statistics</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px' }}>
              <div>
                <strong>Total Detections:</strong> {stats.totalDetections}
              </div>
              <div>
                <strong>Unique Plates:</strong> {stats.uniquePlates.size}
              </div>
              <div>
                <strong>Last Detection:</strong> {lastDetectionTime ? lastDetectionTime.toLocaleTimeString() : 'None'}
              </div>
              <div>
                <strong>Avg Confidence:</strong> {(stats.averageConfidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Recent Detections */}
      {detections.length > 0 && (
        <div style={{ marginTop: '30px' }}>
          <h3>🎯 Recent Detections</h3>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
            gap: '16px',
            marginTop: '16px'
          }}>
            {detections.map((detection, index) => (
              <div key={index} className="detection-card">
                <div style={{ 
                  fontSize: '18px', 
                  fontWeight: 'bold', 
                  color: detection.license_plate_text ? '#2e7d32' : '#666',
                  marginBottom: '8px'
                }}>
                  {detection.license_plate_text || 'No text detected'}
                </div>
                <div className="confidence-info">
                  Detection: {(detection.detection_confidence * 100).toFixed(1)}%
                </div>
                <div className="confidence-info">
                  Text: {(detection.text_confidence * 100).toFixed(1)}%
                </div>
                <div className="confidence-info">
                  Time: {new Date(detection.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Instructions */}
      <div style={{
        background: '#e3f2fd',
        padding: '20px',
        borderRadius: '8px',
        margin: '30px 0',
        textAlign: 'left'
      }}>
        <h4>💡 Tips for Better Detection</h4>
        <ul>
          <li>Ensure good lighting conditions</li>
          <li>Position license plates clearly in the camera view</li>
          <li>Avoid excessive camera movement</li>
          <li>Clean camera lens for better image quality</li>
          <li>Keep a reasonable distance from the license plate</li>
        </ul>
      </div>
    </div>
  );
};

export default WebcamDetection;