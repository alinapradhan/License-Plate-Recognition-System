import React, { useState, useEffect } from 'react';
import axios from 'axios';

const DetectionHistory = () => {
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [filters, setFilters] = useState({
    filterText: '',
    dateFrom: '',
    dateTo: ''
  });

  const pageSize = 20;

  const fetchDetections = async (page = 1) => {
    setLoading(true);
    setError(null);

    try {
      const params = {
        page,
        page_size: pageSize,
        ...(filters.filterText && { filter_text: filters.filterText }),
        ...(filters.dateFrom && { date_from: filters.dateFrom }),
        ...(filters.dateTo && { date_to: filters.dateTo })
      };

      const response = await axios.get('/api/history', { params });
      
      setDetections(response.data.detections);
      setTotalPages(Math.ceil(response.data.total_count / pageSize));
      setCurrentPage(page);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch detection history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDetections(1);
  }, []);

  const handleFilterChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const applyFilters = () => {
    fetchDetections(1);
  };

  const clearFilters = () => {
    setFilters({
      filterText: '',
      dateFrom: '',
      dateTo: ''
    });
    // Fetch without filters after clearing
    setTimeout(() => fetchDetections(1), 100);
  };

  const deleteDetection = async (detectionId) => {
    if (!window.confirm('Are you sure you want to delete this detection?')) {
      return;
    }

    try {
      await axios.delete(`/api/history/${detectionId}`);
      
      // Refresh the current page
      fetchDetections(currentPage);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to delete detection');
    }
  };

  const exportData = async (format = 'json') => {
    try {
      const params = {
        format,
        ...(filters.filterText && { filter_text: filters.filterText }),
        ...(filters.dateFrom && { date_from: filters.dateFrom }),
        ...(filters.dateTo && { date_to: filters.dateTo })
      };

      const response = await axios.get('/api/export', { 
        params,
        responseType: 'blob'
      });

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `detections_export.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to export data');
    }
  };

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      fetchDetections(newPage);
    }
  };

  return (
    <div className="detection-history">
      <h2>📋 Detection History</h2>
      
      {/* Filters */}
      <div className="filters-section" style={{ 
        background: '#f9f9f9', 
        padding: '20px', 
        borderRadius: '8px', 
        margin: '20px 0' 
      }}>
        <h3>Filters</h3>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: '16px',
          alignItems: 'end'
        }}>
          <div>
            <label>License Plate Text:</label>
            <input
              type="text"
              value={filters.filterText}
              onChange={(e) => handleFilterChange('filterText', e.target.value)}
              placeholder="Search by license plate..."
              style={{ 
                width: '100%', 
                padding: '8px', 
                borderRadius: '4px', 
                border: '1px solid #ddd',
                marginTop: '4px'
              }}
            />
          </div>
          <div>
            <label>From Date:</label>
            <input
              type="datetime-local"
              value={filters.dateFrom}
              onChange={(e) => handleFilterChange('dateFrom', e.target.value)}
              style={{ 
                width: '100%', 
                padding: '8px', 
                borderRadius: '4px', 
                border: '1px solid #ddd',
                marginTop: '4px'
              }}
            />
          </div>
          <div>
            <label>To Date:</label>
            <input
              type="datetime-local"
              value={filters.dateTo}
              onChange={(e) => handleFilterChange('dateTo', e.target.value)}
              style={{ 
                width: '100%', 
                padding: '8px', 
                borderRadius: '4px', 
                border: '1px solid #ddd',
                marginTop: '4px'
              }}
            />
          </div>
          <div>
            <button className="btn btn-primary" onClick={applyFilters}>
              🔍 Apply Filters
            </button>
            <button className="btn btn-secondary" onClick={clearFilters}>
              🗑️ Clear
            </button>
          </div>
        </div>
      </div>

      {/* Export Options */}
      <div style={{ margin: '20px 0', textAlign: 'left' }}>
        <strong>Export Data:</strong>
        <button 
          className="btn btn-secondary" 
          onClick={() => exportData('json')}
          style={{ marginLeft: '10px' }}
        >
          📄 Export JSON
        </button>
        <button 
          className="btn btn-secondary" 
          onClick={() => exportData('csv')}
          style={{ marginLeft: '10px' }}
        >
          📊 Export CSV
        </button>
      </div>

      {/* Loading Indicator */}
      {loading && (
        <div style={{ textAlign: 'center', margin: '40px 0' }}>
          <div className="processing-spinner"></div>
          <p>Loading detection history...</p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {/* Detection Table */}
      {!loading && detections.length > 0 && (
        <>
          <table className="history-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>License Plate</th>
                <th>Detection Confidence</th>
                <th>Text Confidence</th>
                <th>Timestamp</th>
                <th>Camera Source</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {detections.map((detection) => (
                <tr key={detection.id}>
                  <td>{detection.id}</td>
                  <td style={{ 
                    fontWeight: 'bold', 
                    color: detection.license_plate_text ? '#2e7d32' : '#666' 
                  }}>
                    {detection.license_plate_text || 'No text detected'}
                  </td>
                  <td>
                    {(detection.detection_confidence * 100).toFixed(1)}%
                  </td>
                  <td>
                    {(detection.text_confidence * 100).toFixed(1)}%
                  </td>
                  <td>
                    {new Date(detection.timestamp).toLocaleString()}
                  </td>
                  <td>{detection.camera_source}</td>
                  <td>
                    <button 
                      className="btn btn-danger"
                      onClick={() => deleteDetection(detection.id)}
                      style={{ fontSize: '12px', padding: '6px 12px' }}
                    >
                      🗑️ Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            gap: '10px',
            margin: '20px 0'
          }}>
            <button 
              className="btn btn-secondary"
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
            >
              ← Previous
            </button>
            <span>
              Page {currentPage} of {totalPages}
            </span>
            <button 
              className="btn btn-secondary"
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
            >
              Next →
            </button>
          </div>
        </>
      )}

      {/* No Data Message */}
      {!loading && detections.length === 0 && !error && (
        <div style={{ 
          textAlign: 'center', 
          margin: '40px 0',
          padding: '40px',
          background: '#f9f9f9',
          borderRadius: '8px'
        }}>
          <p style={{ fontSize: '18px', color: '#666' }}>
            📭 No detection records found
          </p>
          <p style={{ color: '#999' }}>
            Upload some images or use the live camera to start detecting license plates!
          </p>
        </div>
      )}
    </div>
  );
};

export default DetectionHistory;