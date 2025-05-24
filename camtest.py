"""
Test script for camera module
Displays camera feed with various information overlays
Press 'q' to quit, 's' to save a screenshot, 'space' to pause
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime

# Add src directory to path so we can import our module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera.camera_input import CameraInput

def draw_info_overlay(frame: np.ndarray, camera: CameraInput, fps: float) -> np.ndarray:
    """Draw information overlay on frame"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    
    # Draw background rectangles for text
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.rectangle(overlay, (width - 200, 10), (width - 10, 60), (0, 0, 0), -1)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Camera info
    cv2.putText(frame, f"Camera ID: {camera.camera_id}", (20, 35), 
                font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Resolution: {width}x{height}", (20, 60), 
                font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 85), 
                font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Camera FPS: {camera.get_fps():.2f}", (20, 110), 
                font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Controls
    cv2.putText(frame, "Q: Quit", (width - 180, 35), 
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "S: Screenshot", (width - 180, 55), 
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def draw_crosshair(frame: np.ndarray) -> np.ndarray:
    """Draw crosshair at center of frame"""
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Draw crosshair
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), 1)
    
    return frame

def test_camera_feed():
    """Test camera input with live display"""
    print("Camera Input Test")
    print("-" * 50)
    
    # Check for available cameras
    print("Checking for available cameras...")
    temp_camera = CameraInput()
    available_cameras = temp_camera.list_available_cameras()
    temp_camera.release()
    
    if not available_cameras:
        print("No cameras found!")
        return
    
    print(f"Found {len(available_cameras)} camera(s): {available_cameras}")
    
    # Select camera
    camera_id = 0
    if len(available_cameras) > 1:
        print(f"Using camera {camera_id}. Change camera_id in code to use a different camera.")
    
    # Initialize camera
    try:
        camera = CameraInput(camera_id=camera_id, width=640, height=480)
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return
    
    print(f"Camera initialized at {camera.get_frame_dimensions()}")
    print("\nControls:")
    print("  Q     - Quit")
    print("  S     - Save screenshot")
    print("  SPACE - Pause/Resume")
    print("  B/V   - Adjust brightness (-/+)")
    print("  C/X   - Adjust contrast (-/+)")
    
    # Create window
    window_name = "Camera Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # FPS calculation variables
    fps = 0.0
    frame_count = 0
    start_time = cv2.getTickCount()
    
    # State variables
    paused = False
    saved_frame = None
    screenshot_count = 0
    
    try:
        while True:
            # Get frame
            if not paused:
                frame = camera.get_frame_bgr()
                if frame is None:
                    print("Failed to get frame")
                    break
                saved_frame = frame.copy()
            else:
                frame = saved_frame.copy()
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = cv2.getTickCount()
                fps = 30 / ((end_time - start_time) / cv2.getTickFrequency())
                start_time = end_time
            
            # Draw overlays
            frame = draw_info_overlay(frame, camera, fps)
            frame = draw_crosshair(frame)
            
            if paused:
                cv2.putText(frame, "PAUSED", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camera_screenshot_{timestamp}_{screenshot_count}.png"
                cv2.imwrite(filename, saved_frame)
                print(f"Screenshot saved: {filename}")
                screenshot_count += 1
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('b'):
                # Decrease brightness
                current = camera.get_camera_property(cv2.CAP_PROP_BRIGHTNESS)
                camera.set_camera_property(cv2.CAP_PROP_BRIGHTNESS, current - 10)
                print(f"Brightness: {camera.get_camera_property(cv2.CAP_PROP_BRIGHTNESS)}")
            elif key == ord('v'):
                # Increase brightness
                current = camera.get_camera_property(cv2.CAP_PROP_BRIGHTNESS)
                camera.set_camera_property(cv2.CAP_PROP_BRIGHTNESS, current + 10)
                print(f"Brightness: {camera.get_camera_property(cv2.CAP_PROP_BRIGHTNESS)}")
            elif key == ord('c'):
                # Decrease contrast
                current = camera.get_camera_property(cv2.CAP_PROP_CONTRAST)
                camera.set_camera_property(cv2.CAP_PROP_CONTRAST, current - 10)
                print(f"Contrast: {camera.get_camera_property(cv2.CAP_PROP_CONTRAST)}")
            elif key == ord('x'):
                # Increase contrast
                current = camera.get_camera_property(cv2.CAP_PROP_CONTRAST)
                camera.set_camera_property(cv2.CAP_PROP_CONTRAST, current + 10)
                print(f"Contrast: {camera.get_camera_property(cv2.CAP_PROP_CONTRAST)}")
    
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        print("\nCamera test completed")

def test_camera_properties():
    """Test camera properties and capabilities"""
    print("\nCamera Properties Test")
    print("-" * 50)
    
    with CameraInput() as camera:
        print(f"Connected: {camera.is_connected()}")
        print(f"Dimensions: {camera.get_frame_dimensions()}")
        print(f"FPS: {camera.get_fps()}")
        
        # Test frame capture
        frame = camera.get_frame()
        if frame is not None:
            print(f"Frame shape: {frame.shape}")
            print(f"Frame dtype: {frame.dtype}")
            print(f"Frame range: [{frame.min()}, {frame.max()}]")
        else:
            print("Failed to capture frame")
        
        # Test camera properties
        print("\nCamera Properties:")
        properties = {
            "Brightness": cv2.CAP_PROP_BRIGHTNESS,
            "Contrast": cv2.CAP_PROP_CONTRAST,
            "Saturation": cv2.CAP_PROP_SATURATION,
            "Hue": cv2.CAP_PROP_HUE,
            "Gain": cv2.CAP_PROP_GAIN,
            "Exposure": cv2.CAP_PROP_EXPOSURE,
        }
        
        for name, prop_id in properties.items():
            value = camera.get_camera_property(prop_id)
            print(f"  {name}: {value}")

if __name__ == "__main__":
    # Run tests
    test_camera_properties()
    print("\n" + "="*50 + "\n")
    test_camera_feed()