"""
FastAPI backend for license plate detection and recognition system.
"""
import os
import io
import time
import json
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.detector import LicensePlateDetector, VideoProcessor
from database.models import DatabaseManager, Detection


# Pydantic models for API requests/responses
class DetectionResult(BaseModel):
    id: Optional[int] = None
    uuid: Optional[str] = None
    license_plate_text: str
    detection_confidence: float
    text_confidence: float
    bbox: List[int]
    timestamp: str
    camera_source: str = "api"
    gps_coords: Optional[List[float]] = None
    metadata: dict = {}


class DetectionHistory(BaseModel):
    detections: List[DetectionResult]
    total_count: int
    page: int
    page_size: int


class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"


# Initialize FastAPI app
app = FastAPI(
    title="License Plate Detection API",
    description="Real-time license plate detection and recognition system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
detector = None
db_manager = None
video_processor = None

# Create directories
Path("uploads").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global detector, db_manager, video_processor
    
    print("Initializing License Plate Detection API...")
    
    # Initialize database
    db_manager = DatabaseManager()
    print("✓ Database initialized")
    
    # Initialize detector
    detector = LicensePlateDetector()
    print("✓ Detector initialized")
    
    # Initialize video processor
    video_processor = VideoProcessor(detector)
    print("✓ Video processor initialized")
    
    print("🚀 API ready!")


@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check endpoint."""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/detect/image", response_model=List[DetectionResult])
async def detect_in_image(
    file: UploadFile = File(...),
    save_detection: bool = Query(False, description="Save detection to database"),
    gps_lat: Optional[float] = Query(None, description="GPS latitude"),
    gps_lon: Optional[float] = Query(None, description="GPS longitude"),
    camera_source: str = Query("api_upload", description="Camera source identifier")
):
    """
    Detect license plates in uploaded image.
    
    Args:
        file: Image file to process
        save_detection: Whether to save results to database
        gps_lat: GPS latitude (optional)
        gps_lon: GPS longitude (optional)  
        camera_source: Source identifier
        
    Returns:
        List of detection results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Run detection
        start_time = time.time()
        results = detector.process_frame(image)
        processing_time = time.time() - start_time
        
        # Convert results to response format
        detection_results = []
        for result in results:
            # Add processing metadata
            result['processing_info'] = {
                'processing_time_ms': processing_time * 1000,
                'image_size': image.shape[:2],
                'filename': file.filename
            }
            
            # Create detection result
            detection_result = DetectionResult(
                license_plate_text=result['text'],
                detection_confidence=result['detection_confidence'],
                text_confidence=result.get('text_confidence', 0.0),
                bbox=result['bbox'],
                timestamp=datetime.now().isoformat(),
                camera_source=camera_source,
                gps_coords=[gps_lat, gps_lon] if gps_lat and gps_lon else None,
                metadata=result.get('processing_info', {})
            )
            
            # Save to database if requested
            if save_detection:
                # Save uploaded image
                image_filename = f"upload_{int(time.time())}_{file.filename}"
                image_path = Path("uploads") / image_filename
                
                with open(image_path, "wb") as f:
                    f.write(contents)
                
                # Store in database
                gps_coords = (gps_lat, gps_lon) if gps_lat and gps_lon else None
                db_detection = db_manager.store_detection(
                    detection_data=result,
                    image_path=str(image_path),
                    gps_coords=gps_coords,
                    camera_source=camera_source
                )
                
                detection_result.id = db_detection.id
                detection_result.uuid = db_detection.uuid
            
            detection_results.append(detection_result)
        
        return detection_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/api/detect/video")
async def detect_in_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    skip_frames: int = Query(1, description="Process every nth frame"),
    save_detections: bool = Query(False, description="Save detections to database"),
    camera_source: str = Query("api_video", description="Camera source identifier")
):
    """
    Process uploaded video for license plate detection.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Video file to process
        skip_frames: Process every nth frame for speed
        save_detections: Whether to save results to database
        camera_source: Source identifier
        
    Returns:
        Processing job information
    """
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Save uploaded video
        video_filename = f"video_{int(time.time())}_{file.filename}"
        video_path = Path("uploads") / video_filename
        
        contents = await file.read()
        with open(video_path, "wb") as f:
            f.write(contents)
        
        # Create output path for annotated video
        output_filename = f"annotated_{video_filename}"
        output_path = Path("results") / output_filename
        
        # Process video in background
        job_id = f"video_job_{int(time.time())}"
        
        background_tasks.add_task(
            process_video_background,
            str(video_path),
            str(output_path),
            skip_frames,
            save_detections,
            camera_source,
            job_id
        )
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Video processing started",
            "input_file": str(video_path),
            "output_file": str(output_path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


async def process_video_background(
    video_path: str,
    output_path: str,
    skip_frames: int,
    save_detections: bool,
    camera_source: str,
    job_id: str
):
    """Background task to process video."""
    try:
        # Process video
        detections = video_processor.process_video_file(
            video_path=video_path,
            output_path=output_path,
            skip_frames=skip_frames
        )
        
        # Save detections to database if requested
        if save_detections:
            for detection in detections:
                db_manager.store_detection(
                    detection_data=detection,
                    image_path=video_path,
                    camera_source=camera_source
                )
        
        print(f"Video processing completed for job {job_id}: {len(detections)} detections")
        
    except Exception as e:
        print(f"Video processing failed for job {job_id}: {e}")


@app.get("/api/history", response_model=DetectionHistory)
async def get_detection_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    filter_text: Optional[str] = Query(None, description="Filter by license plate text"),
    date_from: Optional[str] = Query(None, description="Filter from date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO format)")
):
    """
    Get detection history with pagination and filtering.
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        filter_text: Filter by license plate text
        date_from: Start date filter
        date_to: End date filter
        
    Returns:
        Paginated detection history
    """
    try:
        # Parse date filters
        date_from_dt = None
        date_to_dt = None
        
        if date_from:
            try:
                date_from_dt = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format")
        
        if date_to:
            try:
                date_to_dt = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format")
        
        # Get detections from database
        offset = (page - 1) * page_size
        detections = db_manager.get_detections(
            limit=page_size,
            offset=offset,
            filter_text=filter_text,
            date_from=date_from_dt,
            date_to=date_to_dt
        )
        
        # Convert to response format
        detection_results = []
        for detection in detections:
            result = DetectionResult(
                id=detection['id'],
                uuid=detection['uuid'],
                license_plate_text=detection['license_plate_text'],
                detection_confidence=detection['detection_confidence'],
                text_confidence=detection['text_confidence'],
                bbox=detection['bbox'],
                timestamp=detection['timestamp'],
                camera_source=detection['camera_source'],
                gps_coords=detection.get('gps_coords'),
                metadata=detection.get('metadata', {})
            )
            detection_results.append(result)
        
        # Get total count (simplified - in production, implement proper counting)
        total_count = len(detections) + offset  # Approximation
        
        return DetectionHistory(
            detections=detection_results,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.delete("/api/history/{detection_id}")
async def delete_detection(detection_id: int):
    """Delete a specific detection record."""
    try:
        success = db_manager.delete_detection(detection_id)
        if success:
            return {"message": "Detection deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Detection not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete detection: {str(e)}")


@app.get("/api/export")
async def export_detections(
    format: str = Query("json", description="Export format (json, csv)"),
    filter_text: Optional[str] = Query(None, description="Filter by license plate text"),
    date_from: Optional[str] = Query(None, description="Filter from date"),
    date_to: Optional[str] = Query(None, description="Filter to date")
):
    """Export detection data in various formats."""
    try:
        # Parse date filters
        date_from_dt = None
        date_to_dt = None
        
        if date_from:
            date_from_dt = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
        if date_to:
            date_to_dt = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
        
        # Get all matching detections
        detections = db_manager.get_detections(
            limit=10000,  # Large limit for export
            filter_text=filter_text,
            date_from=date_from_dt,
            date_to=date_to_dt
        )
        
        if format.lower() == "csv":
            # Export as CSV
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                'id', 'uuid', 'license_plate_text', 'detection_confidence', 
                'text_confidence', 'bbox', 'timestamp', 'camera_source', 'gps_coords'
            ])
            writer.writeheader()
            
            for detection in detections:
                row = detection.copy()
                row['bbox'] = str(row['bbox'])
                row['gps_coords'] = str(row.get('gps_coords', ''))
                writer.writerow(row)
            
            output.seek(0)
            filename = f"detections_export_{int(time.time())}.csv"
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        
        else:
            # Export as JSON
            filename = f"detections_export_{int(time.time())}.json"
            
            return JSONResponse(
                content={
                    "export_timestamp": datetime.now().isoformat(),
                    "total_records": len(detections),
                    "detections": detections
                },
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.post("/api/cleanup")
async def cleanup_old_data(days_old: int = Query(30, description="Delete data older than N days")):
    """Clean up old detection data."""
    try:
        deleted_count = db_manager.cleanup_old_detections(days_old)
        return {
            "message": f"Cleaned up {deleted_count} old detection records",
            "days_old": days_old
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# Serve static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)