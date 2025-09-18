"""
Database Manager for License Plate Recognition System
=====================================================

Handles persistent storage of detection results, user data, and system logs
with SQLite backend and SQLAlchemy ORM.
"""

import asyncio
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from sqlalchemy.dialects.sqlite import JSON
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json
import uuid
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class Detection(Base):
    """Database model for license plate detections."""
    
    __tablename__ = "detections"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    plate_text = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    detection_confidence = Column(Float, default=0.0)
    bbox = Column(JSON)  # Bounding box coordinates [x1, y1, x2, y2]
    image_path = Column(String)  # Path to stored crop image
    full_image_path = Column(String)  # Path to full image (optional)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    location_lat = Column(Float)  # GPS latitude
    location_lon = Column(Float)  # GPS longitude
    location_address = Column(String)  # Human-readable address
    device_id = Column(String)  # Device that made the detection
    user_id = Column(String)  # User who made the detection (optional)
    verified = Column(Boolean, default=False)  # Manual verification status
    notes = Column(Text)  # Additional notes
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemLog(Base):
    """Database model for system logs."""
    
    __tablename__ = "system_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    level = Column(String, nullable=False, index=True)  # INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    module = Column(String)  # Module that generated the log
    function = Column(String)  # Function that generated the log
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    extra_data = Column(JSON)  # Additional structured data


class UserSession(Base):
    """Database model for user sessions (for privacy tracking)."""
    
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String, nullable=False, index=True)
    user_agent = Column(String)
    ip_address = Column(String)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    detection_count = Column(Integer, default=0)
    consent_given = Column(Boolean, default=False)
    data_retention_days = Column(Integer, default=30)


class DatabaseManager:
    """
    Manages database operations for the license plate recognition system.
    """
    
    def __init__(self, database_url: str = "sqlite:///plates.db"):
        """
        Initialize database manager.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
    
    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        try:
            # Create engine
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {}
            )
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            logger.info(f"Database initialized: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a database session."""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()
    
    async def save_detection(self,
                           plate_text: str,
                           confidence: float,
                           bbox: List[int],
                           image_path: Optional[str] = None,
                           full_image_path: Optional[str] = None,
                           detection_confidence: float = 0.0,
                           location_lat: Optional[float] = None,
                           location_lon: Optional[float] = None,
                           location_address: Optional[str] = None,
                           device_id: Optional[str] = None,
                           user_id: Optional[str] = None,
                           timestamp: Optional[datetime] = None) -> str:
        """
        Save a license plate detection to the database.
        
        Args:
            plate_text: Extracted license plate text
            confidence: OCR confidence score
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            image_path: Path to stored crop image
            full_image_path: Path to full source image
            detection_confidence: Detection model confidence
            location_lat: GPS latitude
            location_lon: GPS longitude
            location_address: Human-readable address
            device_id: Device identifier
            user_id: User identifier
            timestamp: Detection timestamp (defaults to now)
            
        Returns:
            Detection ID
        """
        try:
            detection = Detection(
                plate_text=plate_text,
                confidence=confidence,
                detection_confidence=detection_confidence,
                bbox=bbox,
                image_path=image_path,
                full_image_path=full_image_path,
                timestamp=timestamp or datetime.utcnow(),
                location_lat=location_lat,
                location_lon=location_lon,
                location_address=location_address,
                device_id=device_id,
                user_id=user_id
            )
            
            with self.get_session() as session:
                session.add(detection)
                session.commit()
                session.refresh(detection)
                
                logger.info(f"Saved detection: {detection.id} - {plate_text}")
                return detection.id
                
        except Exception as e:
            logger.error(f"Failed to save detection: {e}")
            raise
    
    async def get_detections(self,
                           limit: Optional[int] = 100,
                           offset: int = 0,
                           search: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           device_id: Optional[str] = None,
                           user_id: Optional[str] = None,
                           verified_only: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve detection records with filtering.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            search: Text search in plate_text
            start_date: Filter by minimum timestamp
            end_date: Filter by maximum timestamp
            device_id: Filter by device ID
            user_id: Filter by user ID
            verified_only: Only return verified detections
            
        Returns:
            List of detection records as dictionaries
        """
        try:
            with self.get_session() as session:
                query = session.query(Detection).order_by(Detection.timestamp.desc())
                
                # Apply filters
                if search:
                    query = query.filter(Detection.plate_text.contains(search.upper()))
                
                if start_date:
                    query = query.filter(Detection.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(Detection.timestamp <= end_date)
                
                if device_id:
                    query = query.filter(Detection.device_id == device_id)
                
                if user_id:
                    query = query.filter(Detection.user_id == user_id)
                
                if verified_only:
                    query = query.filter(Detection.verified == True)
                
                # Apply pagination
                if limit:
                    query = query.limit(limit)
                if offset:
                    query = query.offset(offset)
                
                detections = query.all()
                
                # Convert to dictionaries
                result = []
                for detection in detections:
                    result.append({
                        "id": detection.id,
                        "plate_text": detection.plate_text,
                        "confidence": detection.confidence,
                        "detection_confidence": detection.detection_confidence,
                        "bbox": detection.bbox,
                        "image_path": detection.image_path,
                        "full_image_path": detection.full_image_path,
                        "timestamp": detection.timestamp.isoformat(),
                        "location_lat": detection.location_lat,
                        "location_lon": detection.location_lon,
                        "location_address": detection.location_address,
                        "device_id": detection.device_id,
                        "user_id": detection.user_id,
                        "verified": detection.verified,
                        "notes": detection.notes,
                        "created_at": detection.created_at.isoformat(),
                        "updated_at": detection.updated_at.isoformat()
                    })
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to retrieve detections: {e}")
            raise
    
    async def count_detections(self,
                             search: Optional[str] = None,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             device_id: Optional[str] = None,
                             user_id: Optional[str] = None,
                             verified_only: bool = False) -> int:
        """Count detections with the same filters as get_detections."""
        try:
            with self.get_session() as session:
                query = session.query(Detection)
                
                # Apply same filters as get_detections
                if search:
                    query = query.filter(Detection.plate_text.contains(search.upper()))
                
                if start_date:
                    query = query.filter(Detection.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(Detection.timestamp <= end_date)
                
                if device_id:
                    query = query.filter(Detection.device_id == device_id)
                
                if user_id:
                    query = query.filter(Detection.user_id == user_id)
                
                if verified_only:
                    query = query.filter(Detection.verified == True)
                
                return query.count()
                
        except Exception as e:
            logger.error(f"Failed to count detections: {e}")
            return 0
    
    async def get_detection_by_id(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific detection by ID."""
        try:
            with self.get_session() as session:
                detection = session.query(Detection).filter(Detection.id == detection_id).first()
                
                if not detection:
                    return None
                
                return {
                    "id": detection.id,
                    "plate_text": detection.plate_text,
                    "confidence": detection.confidence,
                    "detection_confidence": detection.detection_confidence,
                    "bbox": detection.bbox,
                    "image_path": detection.image_path,
                    "full_image_path": detection.full_image_path,
                    "timestamp": detection.timestamp.isoformat(),
                    "location_lat": detection.location_lat,
                    "location_lon": detection.location_lon,
                    "location_address": detection.location_address,
                    "device_id": detection.device_id,
                    "user_id": detection.user_id,
                    "verified": detection.verified,
                    "notes": detection.notes,
                    "created_at": detection.created_at.isoformat(),
                    "updated_at": detection.updated_at.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get detection {detection_id}: {e}")
            return None
    
    async def update_detection(self,
                             detection_id: str,
                             verified: Optional[bool] = None,
                             notes: Optional[str] = None,
                             location_address: Optional[str] = None) -> bool:
        """Update a detection record."""
        try:
            with self.get_session() as session:
                detection = session.query(Detection).filter(Detection.id == detection_id).first()
                
                if not detection:
                    return False
                
                if verified is not None:
                    detection.verified = verified
                
                if notes is not None:
                    detection.notes = notes
                
                if location_address is not None:
                    detection.location_address = location_address
                
                detection.updated_at = datetime.utcnow()
                
                session.commit()
                logger.info(f"Updated detection: {detection_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update detection {detection_id}: {e}")
            return False
    
    async def delete_detection(self, detection_id: str) -> bool:
        """Delete a detection record."""
        try:
            with self.get_session() as session:
                detection = session.query(Detection).filter(Detection.id == detection_id).first()
                
                if not detection:
                    return False
                
                session.delete(detection)
                session.commit()
                
                logger.info(f"Deleted detection: {detection_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete detection {detection_id}: {e}")
            return False
    
    async def cleanup_old_records(self, days_old: int = 30) -> int:
        """
        Clean up old detection records based on retention policy.
        
        Args:
            days_old: Delete records older than this many days
            
        Returns:
            Number of deleted records
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            with self.get_session() as session:
                # Find old records
                old_detections = session.query(Detection).filter(
                    Detection.created_at < cutoff_date
                ).all()
                
                deleted_count = len(old_detections)
                
                # Delete old records
                for detection in old_detections:
                    session.delete(detection)
                
                session.commit()
                
                logger.info(f"Cleaned up {deleted_count} old detection records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return 0
    
    async def log_event(self,
                       level: str,
                       message: str,
                       module: Optional[str] = None,
                       function: Optional[str] = None,
                       extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log a system event."""
        try:
            log_entry = SystemLog(
                level=level.upper(),
                message=message,
                module=module,
                function=function,
                extra_data=extra_data
            )
            
            with self.get_session() as session:
                session.add(log_entry)
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            with self.get_session() as session:
                # Total detections
                total_detections = session.query(Detection).count()
                
                # Verified detections
                verified_detections = session.query(Detection).filter(
                    Detection.verified == True
                ).count()
                
                # Recent detections (last 24 hours)
                recent_cutoff = datetime.utcnow() - timedelta(days=1)
                recent_detections = session.query(Detection).filter(
                    Detection.timestamp >= recent_cutoff
                ).count()
                
                # Unique plates
                unique_plates = session.query(Detection.plate_text).distinct().count()
                
                # Average confidence
                avg_confidence = session.query(func.avg(Detection.confidence)).scalar() or 0.0
                
                return {
                    "total_detections": total_detections,
                    "verified_detections": verified_detections,
                    "recent_detections_24h": recent_detections,
                    "unique_plates": unique_plates,
                    "average_confidence": float(avg_confidence),
                    "verification_rate": verified_detections / max(total_detections, 1)
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    async def test_database():
        db = DatabaseManager()
        await db.initialize()
        
        # Save a test detection
        detection_id = await db.save_detection(
            plate_text="ABC123",
            confidence=0.95,
            bbox=[100, 200, 300, 250]
        )
        
        print(f"Saved detection: {detection_id}")
        
        # Retrieve detections
        detections = await db.get_detections(limit=10)
        print(f"Found {len(detections)} detections")
        
        # Get statistics
        stats = await db.get_statistics()
        print(f"Statistics: {stats}")
    
    asyncio.run(test_database())