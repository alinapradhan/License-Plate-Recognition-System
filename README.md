
# Car License Plate Detection and Recognition System

A comprehensive system for detecting and recognizing car license plates using YOLOv8 for detection and EasyOCR for text recognition. The system supports both real-time camera input and batch processing, with cross-platform deployment capabilities.

## Features

- **Real-time Detection**: Live camera feed processing with instant plate detection
- **OCR Recognition**: Accurate text extraction from detected license plates
- **Cross-platform Support**: Web app, mobile export (TensorFlow Lite/ONNX)
- **Privacy-focused**: Encrypted storage with deletion options
- **History Management**: Search, filter, and export detection history
- **GPS Integration**: Location tracking for detections
- **Cloud & Edge**: Both API endpoints and on-device inference

## Dataset

Uses the [Kaggle Car Plate Detection dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) for training and validation.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download the dataset (requires Kaggle API setup)
python scripts/download_dataset.py
```

### Training

```bash
# Train YOLOv8 model on car plates
python src/training/train_detector.py

# Evaluate model performance
python src/training/evaluate.py
```

### Running the App

```bash
# Start FastAPI backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start React frontend
cd frontend && npm start
```

### Testing with Images

```bash
# Process single image
python src/inference/detect_plates.py --image path/to/image.jpg

# Process video
python src/inference/detect_plates.py --video path/to/video.mp4

# Real-time webcam
python src/inference/detect_plates.py --webcam
```

## Project Structure

```
├── src/
│   ├── data/               # Data processing and augmentation
│   ├── models/             # Model definitions and utilities
│   ├── training/           # Training scripts and configs
│   ├── inference/          # Inference and detection logic
│   ├── api/               # FastAPI backend
│   ├── database/          # Database models and operations
│   └── utils/             # Utility functions
├── frontend/              # React web application
├── mobile/               # React Native mobile app
├── scripts/              # Setup and utility scripts
├── configs/              # Configuration files
├── tests/                # Test suite
└── docker/               # Docker configurations
```

## API Endpoints

- `POST /api/detect/image` - Detect plates in uploaded image
- `POST /api/detect/video` - Process video file
- `GET /api/history` - Retrieve detection history
- `DELETE /api/history/{id}` - Delete specific detection
- `GET /api/export` - Export detection data

## Mobile Support

Export models for mobile deployment:

```bash
# Export to TensorFlow Lite
python src/export/to_tflite.py

# Export to ONNX
python src/export/to_onnx.py
```

## Privacy & Security

- End-to-end encryption for stored data
- Configurable data retention policies  
- GDPR compliance features
- Secure API authentication
- Local processing options

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kaggle Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

