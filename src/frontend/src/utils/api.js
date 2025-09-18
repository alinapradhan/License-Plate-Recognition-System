/**
 * API Service for License Plate Recognition System
 * Handles all communication with the backend FastAPI server
 */

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 second timeout
    });

    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.api.interceptors.response.use(
      (response) => {
        return response.data;
      },
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        
        // Handle common errors
        if (error.response?.status === 503) {
          throw new Error('Service temporarily unavailable. Please try again later.');
        } else if (error.response?.status === 404) {
          throw new Error('Requested resource not found.');
        } else if (error.response?.status >= 500) {
          throw new Error('Server error. Please try again later.');
        }
        
        throw error.response?.data || error;
      }
    );
  }

  // Health check endpoint
  async checkHealth() {
    try {
      return await this.api.get('/health');
    } catch (error) {
      return { status: 'unhealthy', error: error.message };
    }
  }

  // Detect license plates in image (detection only)
  async detectPlates(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);

    return await this.api.post('/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  // Full recognition (detection + OCR)
  async recognizePlates(imageFile, options = {}) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const params = new URLSearchParams();
    if (options.saveResult !== undefined) {
      params.append('save_result', options.saveResult);
    }
    if (options.includeImage !== undefined) {
      params.append('include_image', options.includeImage);
    }

    const url = `/recognize${params.toString() ? '?' + params.toString() : ''}`;
    
    return await this.api.post(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  // Get detection history
  async getHistory(limit = 100, offset = 0, filters = {}) {
    const params = new URLSearchParams();
    params.append('limit', limit.toString());
    params.append('offset', offset.toString());
    
    if (filters.search) {
      params.append('search', filters.search);
    }
    if (filters.startDate) {
      params.append('start_date', filters.startDate);
    }
    if (filters.endDate) {
      params.append('end_date', filters.endDate);
    }

    return await this.api.get(`/history?${params.toString()}`);
  }

  // Delete detection
  async deleteDetection(detectionId) {
    return await this.api.delete(`/history/${detectionId}`);
  }

  // Export history
  async exportHistory(format = 'json', filters = {}) {
    const params = new URLSearchParams();
    params.append('format', format);
    
    if (filters.startDate) {
      params.append('start_date', filters.startDate);
    }
    if (filters.endDate) {
      params.append('end_date', filters.endDate);
    }

    return await this.api.post(`/export?${params.toString()}`);
  }

  // WebSocket connection for real-time camera feed
  connectWebSocket(onMessage, onError, onClose) {
    const wsUrl = API_BASE_URL.replace('http', 'ws') + '/camera';
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) onError(error);
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      if (onClose) onClose(event);
    };

    return ws;
  }

  // Utility method to create blob URL from base64
  createBlobUrl(base64Data, mimeType = 'image/jpeg') {
    try {
      // Remove data URL prefix if present
      const base64 = base64Data.replace(/^data:image\/[a-z]+;base64,/, '');
      
      // Convert base64 to blob
      const byteCharacters = atob(base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: mimeType });
      
      return URL.createObjectURL(blob);
    } catch (error) {
      console.error('Failed to create blob URL:', error);
      return null;
    }
  }

  // Utility method to download file
  downloadFile(data, filename, mimeType = 'application/json') {
    const blob = new Blob([data], { type: mimeType });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // Format detection result for display
  formatDetectionResult(result) {
    return {
      id: result.id || 'unknown',
      plateText: result.text || 'No text detected',
      confidence: result.text_confidence || result.confidence || 0,
      detectionConfidence: result.detection_confidence || 0,
      bbox: result.bbox || result.detection_bbox || [],
      timestamp: result.timestamp || new Date().toISOString(),
      valid: result.valid_plate || (result.text && result.text.length > 0),
      cropImage: result.crop_image || null
    };
  }

  // Validate image file
  validateImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!validTypes.includes(file.type)) {
      throw new Error('Invalid file type. Please upload JPEG, PNG, or WebP images.');
    }

    if (file.size > maxSize) {
      throw new Error('File too large. Please upload images smaller than 10MB.');
    }

    return true;
  }

  // Resize image before upload (if needed)
  async resizeImage(file, maxWidth = 1920, maxHeight = 1080, quality = 0.8) {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();

      img.onload = () => {
        // Calculate new dimensions
        let { width, height } = img;
        
        if (width > maxWidth) {
          height = (height * maxWidth) / width;
          width = maxWidth;
        }
        
        if (height > maxHeight) {
          width = (width * maxHeight) / height;
          height = maxHeight;
        }

        // Draw resized image
        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);

        // Convert to blob
        canvas.toBlob(resolve, file.type, quality);
      };

      img.src = URL.createObjectURL(file);
    });
  }

  // Get system statistics
  async getSystemStats() {
    try {
      const health = await this.checkHealth();
      const recentHistory = await this.getHistory(50, 0);
      
      return {
        systemHealth: health.status || 'unknown',
        totalDetections: recentHistory.total || 0,
        recentDetections: recentHistory.detections?.length || 0,
        validPlates: recentHistory.detections?.filter(d => d.text && d.text.length > 0).length || 0,
        averageConfidence: this.calculateAverageConfidence(recentHistory.detections || [])
      };
    } catch (error) {
      console.error('Failed to get system stats:', error);
      return {
        systemHealth: 'error',
        totalDetections: 0,
        recentDetections: 0,
        validPlates: 0,
        averageConfidence: 0
      };
    }
  }

  // Calculate average confidence from detections
  calculateAverageConfidence(detections) {
    if (!detections || detections.length === 0) return 0;
    
    const validDetections = detections.filter(d => d.confidence && d.confidence > 0);
    if (validDetections.length === 0) return 0;
    
    const sum = validDetections.reduce((acc, d) => acc + d.confidence, 0);
    return sum / validDetections.length;
  }
}

// Create singleton instance
const apiService = new ApiService();

export default apiService;