"""
Encrypted Storage System for License Plate Recognition
======================================================

Provides secure, encrypted storage for images and sensitive data
with automatic key management and privacy features.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import hashlib
import base64
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import aiofiles
import aiofiles.os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EncryptedStorage:
    """
    Secure storage system with encryption for sensitive data.
    """
    
    def __init__(self, 
                 encryption_key: str,
                 storage_base_path: str = "data/encrypted_storage",
                 max_retention_days: int = 30):
        """
        Initialize encrypted storage.
        
        Args:
            encryption_key: Base key for encryption (should be from secure source)
            storage_base_path: Base directory for encrypted files
            max_retention_days: Maximum data retention period
        """
        self.storage_base_path = Path(storage_base_path)
        self.max_retention_days = max_retention_days
        self.metadata_file = self.storage_base_path / "metadata.enc"
        
        # Initialize encryption
        self._setup_encryption(encryption_key)
        
        # Create storage directories
        self._ensure_directories()
        
        # Load or create metadata
        self.metadata = {}
        asyncio.create_task(self._load_metadata())
    
    def _setup_encryption(self, base_key: str) -> None:
        """Setup encryption with derived key."""
        # Derive a proper encryption key from the base key
        salt = b"license_plate_storage"  # In production, use random salt per installation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(base_key.encode()))
        self.cipher = Fernet(key)
        logger.info("Encryption initialized")
    
    def _ensure_directories(self) -> None:
        """Create necessary storage directories."""
        directories = [
            self.storage_base_path,
            self.storage_base_path / "images",
            self.storage_base_path / "backups",
            self.storage_base_path / "temp"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def _load_metadata(self) -> None:
        """Load metadata from encrypted file."""
        try:
            if await aiofiles.os.path.exists(self.metadata_file):
                async with aiofiles.open(self.metadata_file, 'rb') as f:
                    encrypted_data = await f.read()
                
                decrypted_data = self.cipher.decrypt(encrypted_data)
                self.metadata = json.loads(decrypted_data.decode())
                logger.info(f"Loaded metadata for {len(self.metadata)} files")
            else:
                self.metadata = {}
                logger.info("No existing metadata found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.metadata = {}
    
    async def _save_metadata(self) -> None:
        """Save metadata to encrypted file."""
        try:
            metadata_json = json.dumps(self.metadata, indent=2)
            encrypted_data = self.cipher.encrypt(metadata_json.encode())
            
            async with aiofiles.open(self.metadata_file, 'wb') as f:
                await f.write(encrypted_data)
                
            logger.debug("Metadata saved")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _generate_secure_filename(self, original_name: str) -> str:
        """Generate a secure, unique filename."""
        # Create hash of original name + timestamp + random UUID
        timestamp = datetime.utcnow().isoformat()
        unique_id = str(uuid.uuid4())
        
        hash_input = f"{original_name}_{timestamp}_{unique_id}".encode()
        file_hash = hashlib.sha256(hash_input).hexdigest()[:16]
        
        # Extract extension from original name
        extension = Path(original_name).suffix or ".bin"
        
        return f"{file_hash}{extension}.enc"
    
    async def store_image(self, 
                         image: np.ndarray, 
                         original_name: str,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store an image with encryption.
        
        Args:
            image: Image as numpy array (BGR format)
            original_name: Original filename for reference
            metadata: Additional metadata to store
            
        Returns:
            Unique file ID for retrieval
        """
        try:
            # Generate secure filename
            secure_filename = self._generate_secure_filename(original_name)
            file_path = self.storage_base_path / "images" / secure_filename
            
            # Encode image to bytes
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_bytes = buffer.tobytes()
            
            # Encrypt image data
            encrypted_data = self.cipher.encrypt(image_bytes)
            
            # Save encrypted file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(encrypted_data)
            
            # Generate file ID
            file_id = str(uuid.uuid4())
            
            # Store metadata
            self.metadata[file_id] = {
                "filename": secure_filename,
                "original_name": original_name,
                "file_path": str(file_path.relative_to(self.storage_base_path)),
                "file_type": "image",
                "created_at": datetime.utcnow().isoformat(),
                "size_bytes": len(encrypted_data),
                "image_shape": image.shape,
                "metadata": metadata or {}
            }
            
            await self._save_metadata()
            
            logger.info(f"Stored image: {file_id} ({original_name})")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to store image: {e}")
            raise
    
    async def retrieve_image(self, file_id: str) -> Optional[np.ndarray]:
        """
        Retrieve and decrypt an image.
        
        Args:
            file_id: Unique file ID from store_image()
            
        Returns:
            Image as numpy array, or None if not found
        """
        try:
            if file_id not in self.metadata:
                logger.warning(f"File not found: {file_id}")
                return None
            
            file_info = self.metadata[file_id]
            file_path = self.storage_base_path / file_info["file_path"]
            
            # Check if file exists
            if not await aiofiles.os.path.exists(file_path):
                logger.warning(f"Physical file not found: {file_path}")
                return None
            
            # Read and decrypt file
            async with aiofiles.open(file_path, 'rb') as f:
                encrypted_data = await f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            
            # Decode image
            image_array = np.frombuffer(decrypted_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            logger.debug(f"Retrieved image: {file_id}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to retrieve image {file_id}: {e}")
            return None
    
    async def store_data(self, 
                        data: Dict[str, Any], 
                        data_type: str = "json",
                        retention_days: Optional[int] = None) -> str:
        """
        Store arbitrary data with encryption.
        
        Args:
            data: Data to store (must be JSON serializable)
            data_type: Type of data for metadata
            retention_days: Custom retention period (uses default if None)
            
        Returns:
            Unique file ID for retrieval
        """
        try:
            # Serialize data
            data_json = json.dumps(data, default=str)
            data_bytes = data_json.encode()
            
            # Encrypt data
            encrypted_data = self.cipher.encrypt(data_bytes)
            
            # Generate secure filename
            secure_filename = self._generate_secure_filename(f"{data_type}_data.json")
            file_path = self.storage_base_path / secure_filename
            
            # Save encrypted file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(encrypted_data)
            
            # Generate file ID
            file_id = str(uuid.uuid4())
            
            # Store metadata
            retention = retention_days or self.max_retention_days
            expiry_date = datetime.utcnow() + timedelta(days=retention)
            
            self.metadata[file_id] = {
                "filename": secure_filename,
                "original_name": f"{data_type}_data.json",
                "file_path": str(file_path.relative_to(self.storage_base_path)),
                "file_type": data_type,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": expiry_date.isoformat(),
                "size_bytes": len(encrypted_data),
                "metadata": {"retention_days": retention}
            }
            
            await self._save_metadata()
            
            logger.info(f"Stored data: {file_id} (type: {data_type})")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            raise
    
    async def retrieve_data(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt stored data.
        
        Args:
            file_id: Unique file ID from store_data()
            
        Returns:
            Decrypted data as dictionary, or None if not found
        """
        try:
            if file_id not in self.metadata:
                return None
            
            file_info = self.metadata[file_id]
            file_path = self.storage_base_path / file_info["file_path"]
            
            # Check if file exists and hasn't expired
            if not await aiofiles.os.path.exists(file_path):
                return None
            
            if "expires_at" in file_info:
                expiry = datetime.fromisoformat(file_info["expires_at"])
                if datetime.utcnow() > expiry:
                    logger.info(f"File expired, deleting: {file_id}")
                    await self.delete_file(file_id)
                    return None
            
            # Read and decrypt file
            async with aiofiles.open(file_path, 'rb') as f:
                encrypted_data = await f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode())
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve data {file_id}: {e}")
            return None
    
    async def delete_file(self, file_id: str) -> bool:
        """
        Delete a stored file.
        
        Args:
            file_id: Unique file ID to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            if file_id not in self.metadata:
                return False
            
            file_info = self.metadata[file_id]
            file_path = self.storage_base_path / file_info["file_path"]
            
            # Delete physical file
            if await aiofiles.os.path.exists(file_path):
                await aiofiles.os.remove(file_path)
            
            # Remove from metadata
            del self.metadata[file_id]
            await self._save_metadata()
            
            logger.info(f"Deleted file: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    async def list_files(self, 
                        file_type: Optional[str] = None,
                        include_expired: bool = False) -> List[Dict[str, Any]]:
        """
        List stored files with metadata.
        
        Args:
            file_type: Filter by file type (optional)
            include_expired: Whether to include expired files
            
        Returns:
            List of file metadata
        """
        try:
            files = []
            current_time = datetime.utcnow()
            
            for file_id, file_info in self.metadata.items():
                # Filter by type
                if file_type and file_info.get("file_type") != file_type:
                    continue
                
                # Check expiry
                if not include_expired and "expires_at" in file_info:
                    expiry = datetime.fromisoformat(file_info["expires_at"])
                    if current_time > expiry:
                        continue
                
                files.append({
                    "file_id": file_id,
                    **file_info
                })
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    async def cleanup_expired_files(self) -> int:
        """
        Clean up expired files.
        
        Returns:
            Number of files cleaned up
        """
        try:
            current_time = datetime.utcnow()
            expired_files = []
            
            for file_id, file_info in self.metadata.items():
                if "expires_at" in file_info:
                    expiry = datetime.fromisoformat(file_info["expires_at"])
                    if current_time > expiry:
                        expired_files.append(file_id)
            
            # Delete expired files
            deleted_count = 0
            for file_id in expired_files:
                if await self.delete_file(file_id):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} expired files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired files: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            total_files = len(self.metadata)
            total_size = sum(info.get("size_bytes", 0) for info in self.metadata.values())
            
            file_types = {}
            expired_count = 0
            current_time = datetime.utcnow()
            
            for file_info in self.metadata.values():
                file_type = file_info.get("file_type", "unknown")
                file_types[file_type] = file_types.get(file_type, 0) + 1
                
                if "expires_at" in file_info:
                    expiry = datetime.fromisoformat(file_info["expires_at"])
                    if current_time > expiry:
                        expired_count += 1
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "expired_files": expired_count,
                "storage_path": str(self.storage_base_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    async def export_file(self, file_id: str, output_path: str) -> bool:
        """
        Export a file to unencrypted format.
        
        Args:
            file_id: File ID to export
            output_path: Path to save unencrypted file
            
        Returns:
            True if successfully exported
        """
        try:
            if file_id not in self.metadata:
                return False
            
            file_info = self.metadata[file_id]
            
            if file_info["file_type"] == "image":
                # Export image
                image = await self.retrieve_image(file_id)
                if image is not None:
                    cv2.imwrite(output_path, image)
                    logger.info(f"Exported image to: {output_path}")
                    return True
            else:
                # Export data
                data = await self.retrieve_data(file_id)
                if data is not None:
                    async with aiofiles.open(output_path, 'w') as f:
                        await f.write(json.dumps(data, indent=2))
                    logger.info(f"Exported data to: {output_path}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to export file {file_id}: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    async def test_storage():
        storage = EncryptedStorage("test-key-123")
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        # Store image
        file_id = await storage.store_image(test_image, "test_plate.jpg")
        print(f"Stored image with ID: {file_id}")
        
        # Retrieve image
        retrieved_image = await storage.retrieve_image(file_id)
        print(f"Retrieved image shape: {retrieved_image.shape if retrieved_image is not None else 'None'}")
        
        # Get stats
        stats = await storage.get_storage_stats()
        print(f"Storage stats: {stats}")
        
        # Cleanup
        await storage.delete_file(file_id)
        print("Cleaned up test file")
    
    asyncio.run(test_storage())