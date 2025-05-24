import cv2
import numpy as np
from typing import Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraInput:
    """Handles webcam input and frame capture"""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera connection
        
        Args:
            camera_id: Camera device ID (default: 0 for primary camera)
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_initialized = False
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to camera"""
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Get actual dimensions (camera might not support requested size)
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Log if dimensions differ from requested
            if self.actual_width != self.width or self.actual_height != self.height:
                logger.warning(
                    f"Camera using {self.actual_width}x{self.actual_height} "
                    f"instead of requested {self.width}x{self.height}"
                )
                self.width = self.actual_width
                self.height = self.actual_height
            
            self.is_initialized = True
            logger.info(f"Camera {self.camera_id} connected successfully at {self.width}x{self.height}")
            
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            self.is_initialized = False
            raise
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture and return current frame
        
        Returns:
            numpy array of frame in RGB format, or None if capture fails
        """
        if not self.is_connected():
            logger.warning("Camera not connected")
            return None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Failed to capture frame")
                return None
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame_rgb
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None

    def get_frame_bgr(self) -> Optional[np.ndarray]:
        """
        Capture and return current frame in BGR format (for OpenCV display)
        
        Returns:
            numpy array of frame in BGR format, or None if capture fails
        """
        if not self.is_connected():
            return None
        
        try:
            ret, frame = self.cap.read()
            return frame if ret else None
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None

    def is_connected(self) -> bool:
        """
        Check if camera is connected and operational
        
        Returns:
            True if camera is connected, False otherwise
        """
        return self.is_initialized and self.cap is not None and self.cap.isOpened()
    
    def release(self) -> None:
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")
        self.is_initialized = False
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Get current frame dimensions
        
        Returns:
            Tuple of (width, height)
        """
        return (self.width, self.height)
    
    def get_fps(self) -> float:
        """
        Get camera FPS
        
        Returns:
            Frames per second
        """
        if self.is_connected():
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0
    
    def set_camera_property(self, prop_id: int, value: float) -> bool:
        """
        Set camera property
        
        Args:
            prop_id: OpenCV property ID (e.g., cv2.CAP_PROP_BRIGHTNESS)
            value: Property value
            
        Returns:
            True if successful, False otherwise
        """
        if self.is_connected():
            return self.cap.set(prop_id, value)
        return False
    
    def get_camera_property(self, prop_id: int) -> float:
        """
        Get camera property value
        
        Args:
            prop_id: OpenCV property ID
            
        Returns:
            Property value
        """
        if self.is_connected():
            return self.cap.get(prop_id)
        return -1.0
    
    def list_available_cameras(self, max_cameras: int = 10) -> list:
        """
        List available camera devices
        
        Args:
            max_cameras: Maximum number of cameras to check
            
        Returns:
            List of available camera IDs
        """
        available_cameras = []
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        return available_cameras
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()