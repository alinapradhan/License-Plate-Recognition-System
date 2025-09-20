  
# License Plate Recognition System
 
A comprehensive, cross-platform license plate detection and recognition system using YOLOv8 for detection and EasyOCR for text recognition. Features real-time camera processing, encrypted storage, and both web and mobile interfaces.

##  Features 

### Core Functionality
- **YOLOv8-based License Plate Detection**: High-accuracy detection with customizable confidence thresholds
- **EasyOCR Text Recognition**: Robust OCR with preprocessing for various lighting conditions
- **Real-time Processing**: Live camera feed with WebSocket-based streaming
- **Cross-platform Support**: Web (React), Mobile (React Native), and API endpoints

### Privacy & Security
- **Encrypted Storage**: All images and sensitive data encrypted at rest
- **Data Retention Policies**: Configurable automatic data deletion 
- **Privacy Controls**: User consent management and data export options
- **Secure API**: Authentication and rate limiting support

### Advanced Features
- **Multiple Export Formats**: ONNX, TensorFlow Lite, Core ML for on-device inference
- **GPS Location Tracking**: Optional location tagging for detections
- **Detection History**: Searchable history with filtering and export
- **Performance Analytics**: Real-time statistics and system monitoring

##  Prerequisites

### System Requirements
- Python 3.8+ (recommended: 3.11)
- Node.js 16+ (for frontend)
- 4GB+ RAM
- GPU support optional but recommended for training

### Required Hardware
- Camera (for real-time detection)
- GPU (NVIDIA with CUDA support) - optional but improves performance

##  Installation

### 1. Clone Repository
```bash
git clone https://github.com/alinapradhan/AdaBoost-Image-Projects.git
cd AdaBoost-Image-Projects
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download and Prepare Dataset
```bash
# Install Kaggle API (if not already installed)
pip install kaggle

# Configure Kaggle credentials (follow instructions at https://github.com/Kaggle/kaggle-api)
# Download and prepare dataset
python scripts/prepare_dataset.py --download --visualize
```

### 4. Train Model (Optional)
```bash
# Train YOLOv8 model on prepared dataset
python scripts/train_model.py --config data/processed/dataset.yaml --epochs 100 --batch-size 16
```

### 5. Set Up Frontend
```bash
cd src/frontend
npm install
npm run build
cd ../..
```

##  Quick Start

### Using Docker (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

The application will be available at:
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Manual Startup

#### Start Backend API
```bash
# Set environment variables (optional)
export MODEL_PATH="path/to/your/trained/model.pt"
export DETECTION_CONFIDENCE=0.5
export OCR_CONFIDENCE=0.3

# Start FastAPI server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Start Frontend (Development)
```bash
cd src/frontend
npm start
```

##  Usage Guide

### 1. Web Interface
1. **Home Dashboard**: Overview of system status and recent detections
2. **Live Camera**: Real-time license plate detection from camera feed
3. **Upload & Process**: Upload images for batch processing
4. **Detection History**: Browse and manage detection history
5. **Settings**: Configure system preferences and privacy options

### 2. API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Upload Image for Recognition
```bash
curl -X POST "http://localhost:8000/recognize" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg" \
  -F "save_result=true" \
  -F "include_image=true"
```

#### Get Detection History
```bash
curl "http://localhost:8000/history?limit=10&offset=0"
```

### 3. Real-time Camera Processing
Connect to WebSocket endpoint for live processing:
```javascript
const ws = new WebSocket('ws://localhost:8000/camera');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle real-time detection data
};
```

##  Training Custom Models

### 1. Prepare Your Dataset
```bash
# Organize your data in Pascal VOC format:
# data/raw/
# ├── images/
# │   ├── image1.jpg
# │   └── image2.jpg
# └── annotations/
#     ├── image1.xml
#     └── image2.xml

# Process the dataset
python scripts/prepare_dataset.py --data-dir data --train-split 0.8
```

### 2. Train YOLOv8 Model
```bash
python scripts/train_model.py \
  --config data/processed/dataset.yaml \
  --model-size n \
  --epochs 100 \
  --batch-size 16 \
  --export onnx tflite
```

### 3. Evaluate Model
```bash
# View training results in runs/train/license_plate_*/
# Check training_summary.json for metrics and exported models
```

##  Mobile Development

### React Native Setup
```bash
cd mobile
npm install

# iOS
cd ios && pod install && cd ..
npx react-native run-ios

# Android
npx react-native run-android
```

##  Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
# Model Configuration
MODEL_PATH=data/models/best.pt
DETECTION_CONFIDENCE=0.5
OCR_CONFIDENCE=0.3
USE_GPU=true

# Database
DATABASE_URL=sqlite:///data/plates.db

# Storage
STORAGE_KEY=your-encryption-key-here

# API Configuration  
API_HOST=0.0.0.0
API_PORT=8000
```

### System Settings
- **Detection Confidence**: Minimum confidence for plate detection (0.0-1.0)
- **OCR Confidence**: Minimum confidence for text recognition (0.0-1.0)
- **Data Retention**: Automatic deletion period (days)
- **GPU Usage**: Enable/disable GPU acceleration

##  Testing

### Run Tests
```bash
# Backend tests
python -m pytest tests/ -v

# Frontend tests
cd src/frontend
npm test
```

### Performance Testing
```bash
# Load test API endpoints
python scripts/performance_test.py --concurrent 10 --requests 100
```

##  Performance Optimization

### Model Optimization
- Use appropriate YOLOv8 variant (n/s/m/l/x) based on speed vs accuracy needs
- Export to TensorFlow Lite or ONNX for faster inference
- Enable GPU acceleration when available

### System Optimization
- Adjust frame rate for real-time processing
- Configure batch processing for multiple images
- Use Redis for caching (optional)
- Implement load balancing for high traffic

##  Privacy & Security

### Data Protection
- All stored images are encrypted using Fernet symmetric encryption
- Configurable data retention with automatic cleanup
- User consent tracking and data deletion options
- No data transmitted to external services without explicit consent

### Security Features
- Input validation and sanitization
- Rate limiting on API endpoints
- HTTPS support with SSL/TLS certificates
- Secure file upload handling

##  Deployment

### Production Deployment
1. **Environment Setup**:
   ```bash
   # Set production environment variables
   export ENVIRONMENT=production
   export SECRET_KEY=your-secret-key
   export DATABASE_URL=postgresql://user:pass@localhost/plates
   ```

2. **Docker Deployment**:
   ```bash
   # Build and deploy
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Reverse Proxy Setup** (nginx example):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location /api/ {
           proxy_pass http://localhost:8000/;
       }
       
       location / {
           proxy_pass http://localhost:3000/;
       }
   }
   ```

### Cloud Deployment
- **AWS**: Use ECS, EC2, or Lambda for serverless
- **Google Cloud**: Deploy on Cloud Run or Compute Engine
- **Azure**: Use Container Instances or App Service
- **Kubernetes**: Use provided k8s manifests

##  Troubleshooting

### Common Issues

1. **Camera Not Detected**:
   - Check camera permissions in browser
   - Verify camera device is not in use by other applications
   - Test with different browsers (Chrome recommended)

2. **Model Loading Errors**:
   - Ensure model file exists at specified path
   - Check model format compatibility
   - Verify CUDA installation for GPU usage

3. **Poor Detection Accuracy**:
   - Adjust detection confidence threshold
   - Ensure good lighting conditions
   - Check image quality and resolution
   - Consider retraining with domain-specific data

4. **WebSocket Connection Issues**:
   - Check firewall settings
   - Verify WebSocket support in browser
   - Test with different network configurations

### Performance Issues
- Monitor system resources (CPU, RAM, GPU)
- Adjust batch sizes and processing intervals
- Use appropriate model size for your hardware
- Consider implementing caching strategies

##  Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint for JavaScript/React code
- Write tests for new features
- Update documentation for API changes
- Use meaningful commit messages

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **Dataset**: [Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) by Andrew Mvd
- **YOLOv8**: Ultralytics for the YOLOv8 framework
- **EasyOCR**: JaidedAI for the OCR library
- **React**: Meta for the React framework
- **FastAPI**: Sebastián Ramirez for the FastAPI framework

##  Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `/docs`
- Review troubleshooting section above

