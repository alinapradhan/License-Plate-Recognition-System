"""
Database models for license plate detection history and user management.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import uuid
from cryptography.fernet import Fernet
import base64
import json
from pathlib import Path
import os

Base = declarative_base()


class Detection(Base):
    """Model for storing license plate detection results."""
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Detection information
    image_path = Column(String(500))
    license_plate_text = Column(String(50))  # Encrypted
    detection_confidence = Column(Float)
    text_confidence = Column(Float)
    
    # Bounding box coordinates
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer) 
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    gps_latitude = Column(Float, nullable=True)  # Encrypted
    gps_longitude = Column(Float, nullable=True)  # Encrypted
    camera_source = Column(String(100), default="unknown")
    
    # Encrypted image data (optional, for privacy)
    image_data = Column(LargeBinary, nullable=True)
    
    # Additional metadata as JSON
    extra_data = Column(Text, nullable=True)  # Encrypted JSON string


class User(Base):
    """Model for user authentication and preferences."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    username = Column(String(100), unique=True, index=True)
    email = Column(String(200), unique=True, index=True)
    hashed_password = Column(String(200))
    
    # Privacy settings
    encrypt_data = Column(Boolean, default=True)
    auto_delete_days = Column(Integer, default=30)  # Auto-delete detections after N days
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    is_active = Column(Boolean, default=True)


class EncryptionManager:
    """Manage encryption/decryption of sensitive data."""
    
    def __init__(self, key_file: str = "data/.encryption_key"):
        self.key_file = Path(key_file)
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self._key = self._load_or_create_key()
        self._cipher = Fernet(self._key)
    
    def _load_or_create_key(self) -> bytes:
        """Load existing key or create new one."""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            # Create new key
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(self.key_file, 0o600)
            return key
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        if not data:
            return data
        encrypted_bytes = self._cipher.encrypt(data.encode('utf-8'))
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        if not encrypted_data:
            return encrypted_data
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_bytes = self._cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception:
            return encrypted_data  # Return as-is if decryption fails
    
    def encrypt_json(self, data: dict) -> str:
        """Encrypt JSON data."""
        json_str = json.dumps(data)
        return self.encrypt(json_str)
    
    def decrypt_json(self, encrypted_data: str) -> dict:
        """Decrypt JSON data."""
        try:
            json_str = self.decrypt(encrypted_data)
            return json.loads(json_str) if json_str else {}
        except Exception:
            return {}


class DatabaseManager:
    """Manage database operations with encryption support."""
    
    def __init__(self, db_url: str = "sqlite:///data/license_plate_app.db"):
        self.db_url = db_url
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.encryption_manager = EncryptionManager()
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_db(self):
        """Get database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def store_detection(self, 
                       detection_data: dict, 
                       image_path: str = None,
                       image_data: bytes = None,
                       gps_coords: tuple = None,
                       camera_source: str = "unknown",
                       encrypt_sensitive_data: bool = True) -> Detection:
        """
        Store detection result in database.
        
        Args:
            detection_data: Detection result dictionary
            image_path: Path to original image
            image_data: Binary image data (optional)
            gps_coords: (latitude, longitude) tuple
            camera_source: Source camera identifier
            encrypt_sensitive_data: Whether to encrypt sensitive fields
        
        Returns:
            Created Detection object
        """
        db = next(self.get_db())
        
        try:
            # Extract data from detection result
            bbox = detection_data.get('bbox', [0, 0, 0, 0])
            license_text = detection_data.get('text', '')
            
            # Prepare detection object
            detection = Detection(
                image_path=image_path,
                detection_confidence=detection_data.get('detection_confidence', 0.0),
                text_confidence=detection_data.get('text_confidence', 0.0),
                bbox_x1=bbox[0],
                bbox_y1=bbox[1], 
                bbox_x2=bbox[2],
                bbox_y2=bbox[3],
                camera_source=camera_source,
                image_data=image_data
            )
            
            # Encrypt sensitive data if requested
            if encrypt_sensitive_data:
                detection.license_plate_text = self.encryption_manager.encrypt(license_text)
                
                if gps_coords:
                    detection.gps_latitude = float(self.encryption_manager.encrypt(str(gps_coords[0])))
                    detection.gps_longitude = float(self.encryption_manager.encrypt(str(gps_coords[1])))
                
                # Store additional metadata as encrypted JSON
                metadata = {
                    'raw_results': detection_data.get('raw_results', {}),
                    'processing_info': detection_data.get('processing_info', {})
                }
                detection.extra_data = self.encryption_manager.encrypt_json(metadata)
            else:
                detection.license_plate_text = license_text
                if gps_coords:
                    detection.gps_latitude = gps_coords[0]
                    detection.gps_longitude = gps_coords[1]
                
                metadata = {
                    'raw_results': detection_data.get('raw_results', {}),
                    'processing_info': detection_data.get('processing_info', {})
                }
                detection.extra_data = json.dumps(metadata)
            
            # Add to database
            db.add(detection)
            db.commit()
            db.refresh(detection)
            
            return detection
            
        finally:
            db.close()
    
    def get_detections(self, 
                      limit: int = 100,
                      offset: int = 0,
                      decrypt_data: bool = True,
                      filter_text: str = None,
                      date_from: datetime = None,
                      date_to: datetime = None) -> list:
        """
        Retrieve detection records from database.
        
        Args:
            limit: Maximum number of records
            offset: Number of records to skip
            decrypt_data: Whether to decrypt sensitive data
            filter_text: Filter by license plate text
            date_from: Filter by start date
            date_to: Filter by end date
            
        Returns:
            List of detection dictionaries
        """
        db = next(self.get_db())
        
        try:
            query = db.query(Detection).order_by(Detection.timestamp.desc())
            
            # Apply filters
            if date_from:
                query = query.filter(Detection.timestamp >= date_from)
            if date_to:
                query = query.filter(Detection.timestamp <= date_to)
            
            # Get results
            detections = query.offset(offset).limit(limit).all()
            
            # Convert to dictionaries and decrypt if needed
            results = []
            for detection in detections:
                result = {
                    'id': detection.id,
                    'uuid': detection.uuid,
                    'image_path': detection.image_path,
                    'detection_confidence': detection.detection_confidence,
                    'text_confidence': detection.text_confidence,
                    'bbox': [detection.bbox_x1, detection.bbox_y1, 
                            detection.bbox_x2, detection.bbox_y2],
                    'timestamp': detection.timestamp.isoformat(),
                    'camera_source': detection.camera_source
                }
                
                if decrypt_data:
                    result['license_plate_text'] = self.encryption_manager.decrypt(detection.license_plate_text)
                    
                    if detection.gps_latitude and detection.gps_longitude:
                        try:
                            result['gps_coords'] = [
                                float(self.encryption_manager.decrypt(str(detection.gps_latitude))),
                                float(self.encryption_manager.decrypt(str(detection.gps_longitude)))
                            ]
                        except:
                            result['gps_coords'] = [detection.gps_latitude, detection.gps_longitude]
                    
                    result['metadata'] = self.encryption_manager.decrypt_json(detection.extra_data)
                else:
                    result['license_plate_text'] = detection.license_plate_text
                    if detection.gps_latitude and detection.gps_longitude:
                        result['gps_coords'] = [detection.gps_latitude, detection.gps_longitude]
                    try:
                        result['metadata'] = json.loads(detection.extra_data) if detection.extra_data else {}
                    except:
                        result['metadata'] = {}
                
                # Apply text filter after decryption
                if filter_text and filter_text.lower() not in result['license_plate_text'].lower():
                    continue
                
                results.append(result)
            
            return results
            
        finally:
            db.close()
    
    def delete_detection(self, detection_id: int) -> bool:
        """Delete a detection record."""
        db = next(self.get_db())
        
        try:
            detection = db.query(Detection).filter(Detection.id == detection_id).first()
            if detection:
                # Delete associated image file if exists
                if detection.image_path and Path(detection.image_path).exists():
                    Path(detection.image_path).unlink()
                
                db.delete(detection)
                db.commit()
                return True
            return False
        finally:
            db.close()
    
    def cleanup_old_detections(self, days_old: int = 30) -> int:
        """Delete detections older than specified days."""
        from datetime import timedelta
        
        db = next(self.get_db())
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            old_detections = db.query(Detection).filter(Detection.timestamp < cutoff_date).all()
            
            count = 0
            for detection in old_detections:
                # Delete associated image file if exists
                if detection.image_path and Path(detection.image_path).exists():
                    try:
                        Path(detection.image_path).unlink()
                    except:
                        pass
                
                db.delete(detection)
                count += 1
            
            db.commit()
            return count
            
        finally:
            db.close()


# Global database manager instance
db_manager = DatabaseManager()


if __name__ == "__main__":
    # Test the database setup
    print("Setting up database...")
    
    # Test encryption
    encryption_manager = EncryptionManager()
    
    test_text = "ABC-1234"
    encrypted = encryption_manager.encrypt(test_text)
    decrypted = encryption_manager.decrypt(encrypted)
    
    print(f"Original: {test_text}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")
    print(f"Encryption working: {test_text == decrypted}")
    
    print("Database setup complete!")