import React, { useState, useCallback } from 'react';
import axios from 'axios';

const ImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResults, setDetectionResults] = useState([]);
  const [error, setError] = useState(null);
  const [saveToHistory, setSaveToHistory] = useState(true);

  const handleFileSelect = useCallback((file) => {
    setSelectedFile(file);
    setDetectionResults([]);
    setError(null);
    
    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const processImage = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('save_detection', saveToHistory);
      formData.append('camera_source', 'web_upload');

      const response = await axios.post('/api/detect/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setDetectionResults(response.data);
      
      if (response.data.length === 0) {
        setError('No license plates detected in the image');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process image');
    } finally {
      setIsProcessing(false);
    }
  };

  const clearResults = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setDetectionResults([]);
    setError(null);
  };

  return (
    <div className="image-upload-container">
      <h2>Upload Image for License Plate Detection</h2>
      
      {/* File Upload Area */}
      <div 
        className="upload-area"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
        />
        <p>📁 Drag & drop an image here, or click to select</p>
        <p style={{ fontSize: '14px', color: '#666' }}>
          Supported formats: JPG, PNG, BMP, TIFF
        </p>
      </div>

      {/* Save to History Checkbox */}
      <div style={{ margin: '20px 0' }}>
        <label>
          <input
            type="checkbox"
            checked={saveToHistory}
            onChange={(e) => setSaveToHistory(e.target.checked)}
            style={{ marginRight: '8px' }}
          />
          Save detections to history
        </label>
      </div>

      {/* Image Preview */}
      {previewUrl && (
        <div>
          <h3>Selected Image:</h3>
          <img src={previewUrl} alt="Preview" className="image-preview" />
          <div style={{ margin: '20px 0' }}>
            <button 
              className="btn btn-primary" 
              onClick={processImage}
              disabled={isProcessing}
              style={{ marginRight: '10px' }}
            >
              {isProcessing ? 'Processing...' : '🔍 Detect License Plates'}
            </button>
            <button 
              className="btn btn-secondary" 
              onClick={clearResults}
            >
              🗑️ Clear
            </button>
          </div>
        </div>
      )}

      {/* Processing Indicator */}
      {isProcessing && (
        <div>
          <div className="processing-spinner"></div>
          <p>Analyzing image for license plates...</p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {/* Detection Results */}
      {detectionResults.length > 0 && (
        <div>
          <h3>🎉 Detection Results ({detectionResults.length} found):</h3>
          {detectionResults.map((result, index) => (
            <div key={index} className="detection-result">
              <div className="detection-text">
                License Plate: {result.license_plate_text || 'Text not recognized'}
              </div>
              <div className="confidence-info">
                Detection Confidence: {(result.detection_confidence * 100).toFixed(1)}%
              </div>
              <div className="confidence-info">
                Text Confidence: {(result.text_confidence * 100).toFixed(1)}%
              </div>
              <div className="confidence-info">
                Bounding Box: [{result.bbox.join(', ')}]
              </div>
              <div className="confidence-info">
                Timestamp: {new Date(result.timestamp).toLocaleString()}
              </div>
              {result.id && (
                <div className="confidence-info">
                  Saved to database with ID: {result.id}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ImageUpload;