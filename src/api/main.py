"""
FastAPI Backend for License Plate Recognition
============================================

RESTful API server providing license plate detection and recognition
services with real-time processing capabilities.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import io
from PIL import Image
import base64
import json
import asyncio
import logging
from datetime import datetime
import os
import uuid

from ..models.recognition_pipeline import LicensePlateRecognitionPipeline
from .database import DatabaseManager
from .storage import EncryptedStorage
from ..utils.camera import CameraManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="License Plate Recognition API",
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

# Global components (will be initialized on startup)
pipeline: Optional[LicensePlateRecognitionPipeline] = None
db_manager: Optional[DatabaseManager] = None
storage_manager: Optional[EncryptedStorage] = None
camera_manager: Optional[CameraManager] = None

# Configuration
CONFIG = {
    "model_path": os.environ.get("MODEL_PATH"),
    "detection_confidence": float(os.environ.get("DETECTION_CONFIDENCE", "0.5")),
    "ocr_confidence": float(os.environ.get("OCR_CONFIDENCE", "0.3")),
    "use_gpu": os.environ.get("USE_GPU", "true").lower() == "true",
    "database_url": os.environ.get("DATABASE_URL", "sqlite:///plates.db"),
    "storage_encryption_key": os.environ.get("STORAGE_KEY", "default-key-change-in-production"),
}


@app.on_event("startup")
async def startup_event():
    """Initialize application components."""
    global pipeline, db_manager, storage_manager, camera_manager
    
    logger.info("Starting License Plate Recognition API...")
    
    # Initialize recognition pipeline
    pipeline = LicensePlateRecognitionPipeline(
        detector_model_path=CONFIG["model_path"],
        detection_confidence=CONFIG["detection_confidence"],
        ocr_confidence=CONFIG["ocr_confidence"],
        use_gpu=CONFIG["use_gpu"]
    )
    
    # Initialize database
    db_manager = DatabaseManager(CONFIG["database_url"])
    await db_manager.initialize()
    
    # Initialize storage
    storage_manager = EncryptedStorage(CONFIG["storage_encryption_key"])
    
    # Initialize camera manager
    camera_manager = CameraManager()
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global camera_manager
    
    if camera_manager:
        await camera_manager.cleanup()
    
    logger.info("API shutdown complete")


def image_to_numpy(image_file: bytes) -> np.ndarray:
    """Convert uploaded image to numpy array."""
    try:
        # Open image with PIL
        pil_image = Image.open(io.BytesIO(image_file))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        np_image = np.array(pil_image)
        
        # Convert RGB to BGR (OpenCV format)
        bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        
        return bgr_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")


def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{image_base64}"


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "License Plate Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "/detect - Upload image for plate detection",
            "recognize": "/recognize - Upload image for full recognition",
            "camera": "/camera - Real-time camera processing",
            "history": "/history - Get detection history",
            "health": "/health - API health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline": pipeline is not None,
        "database": db_manager is not None,
        "storage": storage_manager is not None,
        "camera": camera_manager is not None
    }


@app.post("/detect")
async def detect_plates(file: UploadFile = File(...)):
    """
    Detect license plates in uploaded image.
    
    Returns bounding boxes and confidence scores.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Read and convert image
        image_bytes = await file.read()
        image = image_to_numpy(image_bytes)
        
        # Run detection only
        detections = pipeline.detector.detect(
            image, 
            confidence=CONFIG["detection_confidence"],
            return_crops=False
        )
        
        # Format response
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "detections": len(detections),
            "plates": [
                {
                    "id": i,
                    "bbox": det["bbox"],
                    "confidence": det["confidence"]
                }
                for i, det in enumerate(detections)
            ]
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize")
async def recognize_plates(
    file: UploadFile = File(...),
    save_result: bool = True,
    include_image: bool = False
):
    """
    Full license plate recognition on uploaded image.
    
    Detects plates and extracts text with OCR.
    """
    if not pipeline or not db_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Read and convert image
        image_bytes = await file.read()
        image = image_to_numpy(image_bytes)
        
        # Run full recognition
        results = pipeline.process_image(image, return_intermediate=True)
        
        # Process results for response
        response_plates = []
        for result in results:
            plate_data = {
                "id": result["plate_id"],
                "bbox": result["detection_bbox"],
                "detection_confidence": result["detection_confidence"],
                "text": result["text"],
                "text_confidence": result["text_confidence"],
                "valid_plate": result["valid_plate"],
                "timestamp": result["timestamp"]
            }
            
            # Add crop image as base64 if requested
            if include_image and "crop_image" in result and result["crop_image"] is not None:
                plate_data["crop_image"] = numpy_to_base64(result["crop_image"])
            
            response_plates.append(plate_data)
            
            # Save to database if requested and valid
            if save_result and result["valid_plate"] and result["text"]:
                try:
                    # Store crop image
                    image_id = None
                    if "crop_image" in result and result["crop_image"] is not None:
                        image_id = await storage_manager.store_image(
                            result["crop_image"], 
                            f"plate_{uuid.uuid4().hex}.jpg"
                        )
                    
                    # Save to database
                    await db_manager.save_detection(
                        plate_text=result["text"],
                        confidence=result["text_confidence"],
                        bbox=result["detection_bbox"],
                        image_path=image_id,
                        timestamp=datetime.fromisoformat(result["timestamp"])
                    )
                    
                    logger.info(f"Saved detection: {result['text']}")
                    
                except Exception as e:
                    logger.error(f"Failed to save detection: {e}")
        
        # Create annotated image
        annotated_image = None
        if include_image:
            annotated_img = pipeline.draw_results(image, results)
            annotated_image = numpy_to_base64(annotated_img)
        
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "total_plates": len(results),
            "valid_plates": sum(1 for r in results if r["valid_plate"]),
            "plates": response_plates
        }
        
        if annotated_image:
            response["annotated_image"] = annotated_image
        
        return response
        
    except Exception as e:
        logger.error(f"Recognition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history(
    limit: int = 100,
    offset: int = 0,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get detection history with optional filtering."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        # Parse date filters
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Get history from database
        detections = await db_manager.get_detections(
            limit=limit,
            offset=offset,
            search=search,
            start_date=start_dt,
            end_date=end_dt
        )
        
        total_count = await db_manager.count_detections(
            search=search,
            start_date=start_dt,
            end_date=end_dt
        )
        
        return {
            "success": True,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "detections": detections
        }
        
    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{detection_id}")
async def delete_detection(detection_id: str):
    """Delete a specific detection record."""
    if not db_manager or not storage_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Get detection info
        detection = await db_manager.get_detection_by_id(detection_id)
        if not detection:
            raise HTTPException(status_code=404, detail="Detection not found")
        
        # Delete associated image
        if detection.get("image_path"):
            await storage_manager.delete_image(detection["image_path"])
        
        # Delete from database
        await db_manager.delete_detection(detection_id)
        
        return {
            "success": True,
            "message": f"Detection {detection_id} deleted"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export")
async def export_history(
    format: str = "json",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Export detection history in specified format."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Get all matching detections
        detections = await db_manager.get_detections(
            limit=None,  # No limit for export
            start_date=start_dt,
            end_date=end_dt
        )
        
        if format.lower() == "csv":
            import pandas as pd
            df = pd.DataFrame(detections)
            csv_data = df.to_csv(index=False)
            return JSONResponse(
                content={"data": csv_data, "format": "csv"},
                headers={"Content-Type": "application/json"}
            )
        else:
            return {
                "success": True,
                "format": "json",
                "data": detections,
                "count": len(detections)
            }
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/camera")
async def camera_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time camera processing."""
    if not pipeline or not camera_manager:
        await websocket.close(code=4001, reason="Services not initialized")
        return
    
    await websocket.accept()
    logger.info("Camera WebSocket connected")
    
    try:
        # Start camera
        camera_id = await camera_manager.start_camera(0)
        
        while True:
            # Get frame from camera
            frame = await camera_manager.get_frame(camera_id)
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            
            # Process frame
            results = pipeline.process_image(frame)
            
            # Draw results
            annotated_frame = pipeline.draw_results(frame, results)
            
            # Convert to base64 and send
            frame_b64 = numpy_to_base64(annotated_frame)
            
            response = {
                "timestamp": datetime.now().isoformat(),
                "frame": frame_b64,
                "detections": len(results),
                "valid_plates": sum(1 for r in results if r["valid_plate"]),
                "plates": [
                    {
                        "text": r["text"],
                        "confidence": r["text_confidence"],
                        "bbox": r["detection_bbox"],
                        "valid": r["valid_plate"]
                    }
                    for r in results
                ]
            }
            
            await websocket.send_json(response)
            await asyncio.sleep(0.1)  # Limit frame rate
            
    except Exception as e:
        logger.error(f"Camera WebSocket error: {e}")
    finally:
        if camera_manager and 'camera_id' in locals():
            await camera_manager.stop_camera(camera_id)
        logger.info("Camera WebSocket disconnected")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )