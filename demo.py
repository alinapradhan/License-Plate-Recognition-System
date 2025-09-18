"""
Quick demo script for license plate detection system.
"""
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.append('src')

def create_demo_image():
    """Create a demo license plate image."""
    # Create white background
    img = np.ones((300, 600, 3), dtype=np.uint8) * 255
    
    # Draw license plate background
    cv2.rectangle(img, (100, 100), (500, 200), (200, 200, 200), -1)
    cv2.rectangle(img, (100, 100), (500, 200), (0, 0, 0), 3)
    
    # Add text
    cv2.putText(img, "DEMO-123", (150, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Save image
    demo_path = "demo_plate.jpg"
    cv2.imwrite(demo_path, img)
    print(f"✅ Demo image created: {demo_path}")
    return demo_path

def run_basic_demo():
    """Run basic detection demo."""
    print("🚗 License Plate Detection Demo")
    print("=" * 40)
    
    try:
        # Test basic imports
        print("Testing imports...")
        from models.detector import LicensePlateDetector
        print("✅ Imports successful!")
        
        # Create demo image
        demo_path = create_demo_image()
        
        # Load image
        image = cv2.imread(demo_path)
        if image is None:
            print("❌ Could not load demo image")
            return False
        
        print(f"✅ Demo image loaded: {demo_path}")
        print(f"   Image size: {image.shape}")
        
        print("\n🎯 Demo completed successfully!")
        print(f"Demo image saved as: {demo_path}")
        print("\nTo run full detection (requires model download):")
        print("python src/inference/detect_plates.py --image demo_plate.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def show_project_structure():
    """Show the project structure."""
    print("\n📁 Project Structure:")
    print("=" * 40)
    
    structure = """
src/
├── api/               # FastAPI backend
│   └── main.py       # Main API server
├── models/           # Detection models
│   └── detector.py   # YOLOv8 + EasyOCR detector
├── training/         # Model training
│   └── train_detector.py
├── inference/        # Inference scripts
│   └── detect_plates.py
├── database/         # Database models
│   └── models.py     # SQLAlchemy models with encryption
├── data/            # Data processing
│   └── dataset.py   # Dataset preparation
└── export/          # Model export utilities
    └── to_deployment.py

frontend/            # React web application
├── src/
│   ├── components/  # React components
│   ├── App.js      # Main app
│   └── index.js    # Entry point
└── public/

docker/              # Docker deployment
├── Dockerfile
├── docker-compose.yml
└── nginx.conf

scripts/             # Utility scripts
└── download_dataset.py
"""
    
    print(structure)

def show_usage_examples():
    """Show usage examples."""
    print("\n💻 Usage Examples:")
    print("=" * 40)
    
    examples = """
1. Quick Start:
   ./setup.sh                    # Run setup script
   ./start_app.sh               # Start complete application

2. API Server Only:
   python -m src.api.main       # Start FastAPI backend
   # Visit: http://localhost:8000/docs

3. Image Detection:
   python src/inference/detect_plates.py --image photo.jpg

4. Video Processing:
   python src/inference/detect_plates.py --video video.mp4

5. Live Camera:
   python src/inference/detect_plates.py --webcam

6. Train Custom Model:
   python src/training/train_detector.py --data dataset.yaml

7. Export Models:
   python src/export/to_deployment.py --model best.pt --format all

8. Docker Deployment:
   cd docker && docker-compose up -d
"""
    
    print(examples)

def show_features():
    """Show implemented features."""
    print("\n🌟 Features Implemented:")
    print("=" * 40)
    
    features = """
✅ YOLOv8 Object Detection
✅ EasyOCR Text Recognition  
✅ FastAPI REST API
✅ React Web Frontend
✅ Encrypted Database Storage
✅ Real-time Webcam Detection
✅ Batch Image/Video Processing
✅ Detection History & Search
✅ Data Export (JSON/CSV)
✅ Privacy Features (Encryption, Deletion)
✅ Cross-platform Model Export (ONNX, TFLite, TorchScript)
✅ Docker Deployment
✅ GPS Location Support
✅ Comprehensive Documentation
"""
    
    print(features)

def main():
    """Main demo function."""
    run_basic_demo()
    show_project_structure()
    show_usage_examples()
    show_features()
    
    print("\n" + "=" * 50)
    print("🎉 Welcome to the License Plate Detection System!")
    print("🚗 Ready for detecting license plates with high accuracy")
    print("📱 Cross-platform deployment supported")
    print("🔒 Privacy-focused with encryption capabilities")
    print("=" * 50)

if __name__ == "__main__":
    main()