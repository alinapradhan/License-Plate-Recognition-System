"""
License plate detection and recognition inference script.
"""
import cv2
import argparse
import sys
import time
from pathlib import Path
import json

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from models.detector import LicensePlateDetector, VideoProcessor


def process_single_image(detector: LicensePlateDetector, 
                        image_path: str, 
                        output_path: str = None,
                        show_result: bool = True):
    """Process a single image for license plate detection."""
    try:
        print(f"Processing image: {image_path}")
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Detect and recognize license plates
        start_time = time.time()
        results = detector.process_frame(image)
        processing_time = time.time() - start_time
        
        print(f"Processing time: {processing_time:.3f}s")
        print(f"Found {len(results)} license plate(s)")
        
        # Display results
        for i, result in enumerate(results):
            print(f"\nPlate {i+1}:")
            print(f"  Detection confidence: {result['detection_confidence']:.3f}")
            print(f"  Recognized text: '{result['text']}'")
            print(f"  Text confidence: {result.get('text_confidence', 0):.3f}")
            print(f"  Bounding box: {result['bbox']}")
        
        # Draw results on image
        annotated_image = detector.draw_results(image, results)
        
        # Save output if path provided
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to: {output_path}")
        
        # Show result
        if show_result:
            cv2.imshow('License Plate Detection', annotated_image)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def process_video(detector: LicensePlateDetector,
                 video_path: str,
                 output_path: str = None,
                 skip_frames: int = 1):
    """Process video for license plate detection."""
    try:
        print(f"Processing video: {video_path}")
        
        video_processor = VideoProcessor(detector)
        
        start_time = time.time()
        all_detections = video_processor.process_video_file(
            video_path=video_path,
            output_path=output_path,
            skip_frames=skip_frames
        )
        processing_time = time.time() - start_time
        
        print(f"\nVideo processing completed!")
        print(f"Total processing time: {processing_time:.2f}s")
        print(f"Total detections: {len(all_detections)}")
        
        # Group detections by frame
        frames_with_detections = set(d['frame_number'] for d in all_detections)
        print(f"Frames with detections: {len(frames_with_detections)}")
        
        # Show unique license plate texts
        unique_plates = set(d['text'] for d in all_detections if d['text'])
        if unique_plates:
            print(f"Unique license plates detected: {len(unique_plates)}")
            for plate in unique_plates:
                print(f"  - {plate}")
        
        return all_detections
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return None


def process_webcam(detector: LicensePlateDetector, 
                  camera_index: int = 0,
                  save_detections: bool = False):
    """Process webcam stream for real-time license plate detection."""
    
    detections_log = []
    
    def detection_callback(results, annotated_frame):
        """Callback function for handling detections."""
        if save_detections and results:
            timestamp = time.time()
            for result in results:
                result['timestamp'] = timestamp
                detections_log.append(result)
            
            # Print detected plates
            for result in results:
                if result['text']:
                    print(f"Detected: {result['text']} (confidence: {result.get('text_confidence', 0):.3f})")
    
    try:
        print(f"Starting webcam stream (camera {camera_index})")
        print("Press 'q' to quit, 's' to save current frame")
        
        video_processor = VideoProcessor(detector)
        
        # Modify the camera stream processing to save frames
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Process frame
            start_time = time.time()
            results = detector.process_frame(frame)
            processing_time = time.time() - start_time
            
            # Add timestamp to results
            timestamp = time.time()
            for result in results:
                result['timestamp'] = timestamp
            
            # Draw results
            annotated_frame = detector.draw_results(frame, results)
            
            # Add FPS and processing time to display
            fps_text = f"FPS: {1/processing_time:.1f} | Processing: {processing_time*1000:.1f}ms"
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Call detection callback
            if results:
                detection_callback(results, annotated_frame)
            
            # Show frame
            cv2.imshow('License Plate Detection - Live', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_path = f"detection_frame_{frame_count:06d}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"Frame saved: {save_path}")
            
            frame_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if save_detections and detections_log:
            # Save detections to JSON file
            output_file = f"webcam_detections_{int(time.time())}.json"
            with open(output_file, 'w') as f:
                json.dump(detections_log, f, indent=2)
            print(f"Detections saved to: {output_file}")


def main():
    """Main function for license plate detection inference."""
    parser = argparse.ArgumentParser(description="License Plate Detection and Recognition")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to input image")
    group.add_argument("--video", type=str, help="Path to input video")
    group.add_argument("--webcam", action="store_true", help="Use webcam input")
    
    # Model options
    parser.add_argument("--model", type=str, help="Path to trained YOLOv8 model")
    parser.add_argument("--conf-threshold", type=float, default=0.5, 
                       help="Confidence threshold for detection")
    parser.add_argument("--nms-threshold", type=float, default=0.4,
                       help="NMS threshold for detection")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output path for annotated result")
    parser.add_argument("--save-detections", action="store_true",
                       help="Save detection results to JSON file")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't display result images")
    
    # Video specific options
    parser.add_argument("--skip-frames", type=int, default=1,
                       help="Skip frames in video processing (for speed)")
    
    # Camera options
    parser.add_argument("--camera-index", type=int, default=0,
                       help="Camera device index for webcam")
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        print("Initializing license plate detector...")
        detector = LicensePlateDetector(
            detection_model_path=args.model,
            confidence_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold
        )
        print("Detector initialized successfully!")
        
        # Process based on input type
        if args.image:
            results = process_single_image(
                detector=detector,
                image_path=args.image,
                output_path=args.output,
                show_result=not args.no_display
            )
            
            if args.save_detections and results:
                output_file = f"{Path(args.image).stem}_detections.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Detection results saved to: {output_file}")
        
        elif args.video:
            results = process_video(
                detector=detector,
                video_path=args.video,
                output_path=args.output,
                skip_frames=args.skip_frames
            )
            
            if args.save_detections and results:
                output_file = f"{Path(args.video).stem}_detections.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Detection results saved to: {output_file}")
        
        elif args.webcam:
            process_webcam(
                detector=detector,
                camera_index=args.camera_index,
                save_detections=args.save_detections
            )
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())