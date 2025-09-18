"""
Simple test script to verify the license plate detection system.
"""
import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add src directory to path
sys.path.append('src')

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from models.detector import LicensePlateDetector, VideoProcessor
        from database.models import DatabaseManager
        from api.main import app
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_detector_initialization():
    """Test detector initialization."""
    print("\nTesting detector initialization...")
    
    try:
        from models.detector import LicensePlateDetector
        
        print("Initializing detector (this may take a moment to download models)...")
        detector = LicensePlateDetector()
        print("✅ Detector initialized successfully!")
        return detector
    except Exception as e:
        print(f"❌ Detector initialization error: {e}")
        return None

def test_database():
    """Test database functionality."""
    print("\nTesting database...")
    
    try:
        from database.models import DatabaseManager
        
        # Initialize database
        db_manager = DatabaseManager()
        print("✅ Database initialized successfully!")
        
        # Test encryption
        test_text = "TEST-123"
        encrypted = db_manager.encryption_manager.encrypt(test_text)
        decrypted = db_manager.encryption_manager.decrypt(encrypted)
        
        if test_text == decrypted:
            print("✅ Encryption/decryption working correctly!")
        else:
            print("❌ Encryption/decryption failed!")
            return False
        
        return db_manager
    except Exception as e:
        print(f"❌ Database error: {e}")
        return None

def create_test_image():
    """Create a simple test image with text that looks like a license plate."""
    print("\nCreating test image...")
    
    try:
        # Create a simple test image with white background
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        # Add a rectangle to simulate a license plate
        cv2.rectangle(img, (50, 70), (350, 130), (0, 0, 0), 2)
        cv2.rectangle(img, (52, 72), (348, 128), (255, 255, 255), -1)
        
        # Add some text
        cv2.putText(img, "ABC-123", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        # Save test image
        test_img_path = "test_license_plate.jpg"
        cv2.imwrite(test_img_path, img)
        
        print(f"✅ Test image created: {test_img_path}")
        return test_img_path
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return None

def test_detection(detector, test_img_path):
    """Test license plate detection on test image."""
    print("\nTesting detection on test image...")
    
    try:
        if not detector or not test_img_path:
            print("❌ Missing detector or test image")
            return False
        
        # Load test image
        image = cv2.imread(test_img_path)
        if image is None:
            print("❌ Could not load test image")
            return False
        
        # Run detection
        print("Running detection...")
        results = detector.process_frame(image)
        
        print(f"✅ Detection completed! Found {len(results)} detections")
        
        for i, result in enumerate(results):
            print(f"  Detection {i+1}:")
            print(f"    Text: '{result['text']}'")
            print(f"    Detection confidence: {result['detection_confidence']:.3f}")
            print(f"    Text confidence: {result.get('text_confidence', 0):.3f}")
            print(f"    Bounding box: {result['bbox']}")
        
        # Draw results and save
        annotated_img = detector.draw_results(image, results)
        output_path = "test_result.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"✅ Annotated result saved to: {output_path}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Detection test error: {e}")
        return False

def test_database_storage(db_manager):
    """Test storing detection results in database."""
    print("\nTesting database storage...")
    
    try:
        if not db_manager:
            print("❌ Database manager not available")
            return False
        
        # Create test detection data
        test_detection = {
            'bbox': [50, 70, 350, 130],
            'text': 'TEST-123',
            'detection_confidence': 0.95,
            'text_confidence': 0.88
        }
        
        # Store in database
        stored = db_manager.store_detection(
            detection_data=test_detection,
            image_path="test_license_plate.jpg",
            camera_source="test_script"
        )
        
        print(f"✅ Detection stored with ID: {stored.id}")
        
        # Retrieve from database
        detections = db_manager.get_detections(limit=1)
        
        if detections and len(detections) > 0:
            retrieved = detections[0]
            print(f"✅ Successfully retrieved detection: {retrieved['license_plate_text']}")
            
            # Clean up - delete test detection
            deleted = db_manager.delete_detection(stored.id)
            if deleted:
                print("✅ Test detection cleaned up")
            
            return True
        else:
            print("❌ Could not retrieve stored detection")
            return False
            
    except Exception as e:
        print(f"❌ Database storage test error: {e}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    print("\nCleaning up test files...")
    
    test_files = ["test_license_plate.jpg", "test_result.jpg"]
    
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"✅ Cleaned up: {file}")
            except Exception as e:
                print(f"⚠️  Could not remove {file}: {e}")

def main():
    """Main test function."""
    print("🧪 License Plate Detection System Test Suite")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_tests_passed = False
        print("❌ Basic import test failed. Cannot continue.")
        return False
    
    # Test 2: Database
    db_manager = test_database()
    if not db_manager:
        all_tests_passed = False
    
    # Test 3: Detector initialization
    detector = test_detector_initialization()
    if not detector:
        all_tests_passed = False
        print("❌ Detector initialization failed. Skipping detection tests.")
    else:
        # Test 4: Create test image
        test_img_path = create_test_image()
        
        # Test 5: Detection
        if test_img_path:
            detection_success = test_detection(detector, test_img_path)
            if not detection_success:
                all_tests_passed = False
        else:
            all_tests_passed = False
    
    # Test 6: Database storage
    if db_manager:
        storage_success = test_database_storage(db_manager)
        if not storage_success:
            all_tests_passed = False
    
    # Cleanup
    cleanup_test_files()
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! The system is working correctly.")
        print("\nNext steps:")
        print("1. Start the API server: python -m src.api.main")
        print("2. Test with real images: python src/inference/detect_plates.py --image your_image.jpg")
        print("3. Try the web interface by building the React frontend")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Common issues:")
        print("- Missing dependencies (run: pip install -r requirements.txt)")
        print("- Network issues downloading models")
        print("- Insufficient system resources")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)