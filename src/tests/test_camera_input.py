import unittest
import numpy as np
import cv2
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.camera.camera_input import CameraInput

class TestCameraInput(unittest.TestCase):
    """Unit tests for CameraInput class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.camera = None
    
    def tearDown(self):
        """Clean up after tests"""
        if self.camera is not None:
            self.camera.release()
    
    def test_camera_initialization(self):
        """Test camera initialization"""
        try:
            self.camera = CameraInput()
            self.assertTrue(self.camera.is_connected())
            self.assertIsNotNone(self.camera.cap)
        except RuntimeError:
            self.skipTest("No camera available")
    
    def test_frame_capture(self):
        """Test frame capture"""
        try:
            self.camera = CameraInput()
            frame = self.camera.get_frame()
            
            self.assertIsNotNone(frame)
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(len(frame.shape), 3)  # Should be 3D (height, width, channels)
            self.assertEqual(frame.shape[2], 3)  # Should have 3 channels (RGB)
            
        except RuntimeError:
            self.skipTest("No camera available")
    
    def test_frame_dimensions(self):
        """Test frame dimensions"""
        try:
            width, height = 640, 480
            self.camera = CameraInput(width=width, height=height)
            
            frame = self.camera.get_frame()
            if frame is not None:
                actual_height, actual_width = frame.shape[:2]
                
                # Camera might not support exact dimensions
                self.assertGreater(actual_width, 0)
                self.assertGreater(actual_height, 0)
                
                # Check if get_frame_dimensions returns actual dimensions
                reported_dims = self.camera.get_frame_dimensions()
                self.assertEqual(reported_dims[0], actual_width)
                self.assertEqual(reported_dims[1], actual_height)
                
        except RuntimeError:
            self.skipTest("No camera available")
    
    def test_bgr_rgb_conversion(self):
        """Test BGR to RGB conversion"""
        try:
            self.camera = CameraInput()
            
            frame_rgb = self.camera.get_frame()
            frame_bgr = self.camera.get_frame_bgr()
            
            if frame_rgb is not None and frame_bgr is not None:
                # Check that RGB and BGR are different (unless image is grayscale)
                if not np.array_equal(frame_rgb[:,:,0], frame_rgb[:,:,2]):
                    self.assertFalse(np.array_equal(frame_rgb, frame_bgr))
                
                # Check conversion is correct
                converted = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                np.testing.assert_array_equal(frame_rgb, converted)
                
        except RuntimeError:
            self.skipTest("No camera available")
    
    def test_camera_release(self):
        """Test camera release"""
        try:
            self.camera = CameraInput()
            self.assertTrue(self.camera.is_connected())
            
            self.camera.release()
            self.assertFalse(self.camera.is_connected())
            
            # Should be able to get None frame after release
            frame = self.camera.get_frame()
            self.assertIsNone(frame)
            
        except RuntimeError:
            self.skipTest("No camera available")
    
    def test_context_manager(self):
        """Test context manager functionality"""
        try:
            with CameraInput() as camera:
                self.assertTrue(camera.is_connected())
                frame = camera.get_frame()
                self.assertIsNotNone(frame)
            
            # Camera should be released after context
            self.assertFalse(camera.is_connected())
            
        except RuntimeError:
            self.skipTest("No camera available")
    
    def test_multiple_cameras(self):
        """Test listing available cameras"""
        temp_camera = CameraInput()
        available = temp_camera.list_available_cameras()
        temp_camera.release()
        
        self.assertIsInstance(available, list)
        # Should at least find camera 0 if any camera exists
        if len(available) > 0:
            self.assertIn(0, available)

if __name__ == '__main__':
    unittest.main()