import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import DetectionHistory from './components/DetectionHistory';
import WebcamDetection from './components/WebcamDetection';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('upload');

  const renderContent = () => {
    switch(activeTab) {
      case 'upload':
        return <ImageUpload />;
      case 'webcam':
        return <WebcamDetection />;
      case 'history':
        return <DetectionHistory />;
      default:
        return <ImageUpload />;
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🚗 License Plate Detection System</h1>
        <p>Real-time license plate detection and recognition</p>
      </header>
      
      <div className="main-container">
        <nav className="nav-tabs">
          <button 
            className={`nav-tab ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            📁 Upload Image
          </button>
          <button 
            className={`nav-tab ${activeTab === 'webcam' ? 'active' : ''}`}
            onClick={() => setActiveTab('webcam')}
          >
            📹 Live Camera
          </button>
          <button 
            className={`nav-tab ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            📋 Detection History
          </button>
        </nav>
        
        {renderContent()}
      </div>
      
      <footer style={{marginTop: '40px', padding: '20px', textAlign: 'center', color: '#666'}}>
        <p>License Plate Detection System v1.0</p>
        <p>Powered by YOLOv8 and EasyOCR</p>
      </footer>
    </div>
  );
}

export default App;