"""
Camera Management Utility
=========================

Handles camera access, frame capture, and real-time video processing
for the license plate recognition system.
"""

import cv2
import numpy as np
import asyncio
from typing import Optional, Dict, Any, List, Callable
import logging
from datetime import datetime
import threading
from queue import Queue, Empty
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraStream:
    """Individual camera stream handler."""
    
    def __init__(self, camera_id: int, resolution: tuple = (640, 480), fps: int = 30):
        """
        Initialize camera stream.
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
            resolution: Desired resolution (width, height)
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.frame_queue = Queue(maxsize=10)
        self.is_running = False
        self.capture_thread = None
        self.stream_id = str(uuid.uuid4())
        
        # Statistics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_frame_time = None
        self.actual_fps = 0.0
        
    def start(self) -> bool:
        """Start the camera stream."""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Cannot open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera {self.camera_id} opened: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera {self.camera_id}: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the camera stream."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        logger.info(f"Camera {self.camera_id} stopped")
    
    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        frame_interval = 1.0 / self.fps
        last_time = datetime.now()
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame from camera {self.camera_id}")
                    continue
                
                current_time = datetime.now()
                
                # Calculate actual FPS
                if self.last_frame_time:
                    time_diff = (current_time - self.last_frame_time).total_seconds()
                    if time_diff > 0:
                        self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1.0 / time_diff)
                
                self.last_frame_time = current_time
                self.frames_captured += 1
                
                # Add frame to queue (drop if full)
                try:
                    self.frame_queue.put_nowait({
                        'frame': frame,
                        'timestamp': current_time,
                        'frame_number': self.frames_captured
                    })
                except:
                    # Queue is full, drop frame
                    self.frames_dropped += 1
                
                # Control frame rate
                elapsed = (current_time - last_time).total_seconds()
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    cv2.waitKey(int(sleep_time * 1000))
                
                last_time = current_time
                
            except Exception as e:
                logger.error(f"Capture error for camera {self.camera_id}: {e}")
                break
    
    def get_frame(self) -> Optional[Dict[str, Any]]:
        """Get the latest frame."""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            "camera_id": self.camera_id,
            "stream_id": self.stream_id,
            "resolution": self.resolution,
            "target_fps": self.fps,
            "actual_fps": round(self.actual_fps, 2),
            "frames_captured": self.frames_captured,
            "frames_dropped": self.frames_dropped,
            "is_running": self.is_running,
            "queue_size": self.frame_queue.qsize()
        }


class CameraManager:
    """
    Manages multiple camera streams and provides unified interface.
    """
    
    def __init__(self):
        """Initialize camera manager."""
        self.streams: Dict[str, CameraStream] = {}
        self.available_cameras = self._detect_available_cameras()
        logger.info(f"Detected {len(self.available_cameras)} available cameras")
    
    def _detect_available_cameras(self) -> List[int]:
        """Detect available camera devices."""
        available = []
        
        # Test cameras 0-9 (usually sufficient)
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        
        return available
    
    async def start_camera(self, 
                          camera_id: int = 0, 
                          resolution: tuple = (640, 480), 
                          fps: int = 30) -> Optional[str]:
        """
        Start a camera stream.
        
        Args:
            camera_id: Camera device ID
            resolution: Desired resolution (width, height)
            fps: Target frames per second
            
        Returns:
            Stream ID if successful, None otherwise
        """
        if camera_id not in self.available_cameras:
            logger.error(f"Camera {camera_id} not available")
            return None
        
        try:
            # Create camera stream
            stream = CameraStream(camera_id, resolution, fps)
            
            # Start stream
            if stream.start():
                self.streams[stream.stream_id] = stream
                logger.info(f"Started camera {camera_id} with stream ID: {stream.stream_id}")
                return stream.stream_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to start camera {camera_id}: {e}")
            return None
    
    async def stop_camera(self, stream_id: str) -> bool:
        """
        Stop a camera stream.
        
        Args:
            stream_id: Stream ID from start_camera()
            
        Returns:
            True if successfully stopped
        """
        if stream_id not in self.streams:
            return False
        
        try:
            stream = self.streams[stream_id]
            stream.stop()
            del self.streams[stream_id]
            logger.info(f"Stopped camera stream: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop camera {stream_id}: {e}")
            return False
    
    async def get_frame(self, stream_id: str) -> Optional[np.ndarray]:
        """
        Get the latest frame from a camera stream.
        
        Args:
            stream_id: Stream ID from start_camera()
            
        Returns:
            Latest frame as numpy array, or None if not available
        """
        if stream_id not in self.streams:
            return None
        
        try:
            stream = self.streams[stream_id]
            frame_data = stream.get_frame()
            
            if frame_data:
                return frame_data['frame']
            return None
            
        except Exception as e:
            logger.error(f"Failed to get frame from {stream_id}: {e}")
            return None
    
    async def get_frame_with_metadata(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """
        Get frame with metadata.
        
        Args:
            stream_id: Stream ID from start_camera()
            
        Returns:
            Frame data with metadata, or None if not available
        """
        if stream_id not in self.streams:
            return None
        
        try:
            stream = self.streams[stream_id]
            return stream.get_frame()
            
        except Exception as e:
            logger.error(f"Failed to get frame with metadata from {stream_id}: {e}")
            return None
    
    async def list_active_streams(self) -> List[Dict[str, Any]]:
        """List all active camera streams with statistics."""
        streams_info = []
        
        for stream_id, stream in self.streams.items():
            streams_info.append(stream.get_stats())
        
        return streams_info
    
    async def get_stream_stats(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific stream."""
        if stream_id not in self.streams:
            return None
        
        return self.streams[stream_id].get_stats()
    
    def get_available_cameras(self) -> List[int]:
        """Get list of available camera IDs."""
        return self.available_cameras.copy()
    
    async def cleanup(self) -> None:
        """Stop all camera streams and cleanup."""
        stream_ids = list(self.streams.keys())
        
        for stream_id in stream_ids:
            await self.stop_camera(stream_id)
        
        logger.info("Camera manager cleanup complete")
    
    async def capture_image(self, 
                           camera_id: int = 0, 
                           resolution: tuple = (640, 480)) -> Optional[np.ndarray]:
        """
        Capture a single image from camera.
        
        Args:
            camera_id: Camera device ID
            resolution: Desired resolution
            
        Returns:
            Captured image as numpy array, or None if failed
        """
        if camera_id not in self.available_cameras:
            logger.error(f"Camera {camera_id} not available")
            return None
        
        try:
            # Temporary capture
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                logger.error(f"Cannot open camera {camera_id}")
                return None
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            
            # Allow camera to warm up
            for _ in range(5):
                ret, frame = cap.read()
                if not ret:
                    continue
            
            # Capture final image
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                logger.info(f"Captured image from camera {camera_id}: {frame.shape}")
                return frame
            else:
                logger.error(f"Failed to capture image from camera {camera_id}")
                return None
                
        except Exception as e:
            logger.error(f"Image capture failed: {e}")
            return None
    
    async def record_video(self, 
                          stream_id: str,
                          output_path: str,
                          duration_seconds: int = 10,
                          codec: str = 'mp4v') -> bool:
        """
        Record video from a camera stream.
        
        Args:
            stream_id: Stream ID from start_camera()
            output_path: Path to save video file
            duration_seconds: Recording duration in seconds
            codec: Video codec to use
            
        Returns:
            True if successfully recorded
        """
        if stream_id not in self.streams:
            return False
        
        try:
            stream = self.streams[stream_id]
            
            # Get stream properties
            stats = stream.get_stats()
            resolution = stats['resolution']
            fps = stats['target_fps']
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
            
            start_time = datetime.now()
            frames_recorded = 0
            
            while (datetime.now() - start_time).seconds < duration_seconds:
                frame_data = stream.get_frame()
                
                if frame_data:
                    out.write(frame_data['frame'])
                    frames_recorded += 1
                
                await asyncio.sleep(1.0 / fps)
            
            out.release()
            
            logger.info(f"Recorded {frames_recorded} frames to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Video recording failed: {e}")
            return False


class MockCamera:
    """Mock camera for testing purposes."""
    
    def __init__(self, width: int = 640, height: int = 480):
        """Initialize mock camera."""
        self.width = width
        self.height = height
        self.frame_count = 0
    
    def generate_frame(self) -> np.ndarray:
        """Generate a mock frame with some pattern."""
        # Create a simple pattern with frame counter
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add some pattern
        cv2.rectangle(frame, (50, 50), (self.width-50, self.height-50), (100, 100, 100), 2)
        
        # Add frame counter text
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (10, self.height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        self.frame_count += 1
        return frame


if __name__ == "__main__":
    # Example usage
    async def test_camera():
        manager = CameraManager()
        
        # List available cameras
        cameras = manager.get_available_cameras()
        print(f"Available cameras: {cameras}")
        
        if cameras:
            # Start first available camera
            stream_id = await manager.start_camera(cameras[0])
            
            if stream_id:
                print(f"Started camera stream: {stream_id}")
                
                # Capture some frames
                for i in range(10):
                    frame = await manager.get_frame(stream_id)
                    if frame is not None:
                        print(f"Frame {i}: {frame.shape}")
                    
                    await asyncio.sleep(0.1)
                
                # Get statistics
                stats = await manager.get_stream_stats(stream_id)
                print(f"Stream stats: {stats}")
                
                # Stop camera
                await manager.stop_camera(stream_id)
                print("Camera stopped")
        
        else:
            print("No cameras available, testing with mock camera")
            mock = MockCamera()
            frame = mock.generate_frame()
            print(f"Mock frame: {frame.shape}")
        
        await manager.cleanup()
    
    asyncio.run(test_camera())