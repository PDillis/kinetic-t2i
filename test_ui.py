"""
Test script for UI interface with real camera and hand tracking
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
import time
import threading
import mediapipe as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ui.interface import MainInterface
from src.camera.camera_input import CameraInput
from src.hand_tracking.hand_detector import HandDetector

def test_ui_with_real_camera():
    """Test the UI interface with real camera and hand tracking"""
    print("Testing UI Interface with Real Camera and Hand Tracking")
    print("-" * 50)
    print("Press Ctrl+C to quit")
    
    # Initialize components
    camera = None
    hand_detector = None
    ui = None
    
    try:
        # Initialize camera
        print("Initializing camera...")
        camera = CameraInput(width=640, height=480)
        if not camera.is_connected():
            print("Error: Camera not connected!")
            return
        print("Camera initialized successfully")
        
        # Initialize hand detector
        print("Initializing hand detector...")
        hand_detector = HandDetector(max_num_hands=1)
        print("Hand detector initialized successfully")
        
        # Initialize UI
        print("Initializing UI...")
        ui = MainInterface(window_width=1280, window_height=720)
        
        # MediaPipe drawing utilities for comparison
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        
        # Set up callbacks
        def on_test_mode_changed(mode):
            ui.update_status(f"Test mode: {mode}")
        
        def on_brightness_changed(value):
            # Apply to camera if supported
            if camera.is_connected():
                camera.set_camera_property(cv2.CAP_PROP_BRIGHTNESS, value)
        
        def on_contrast_changed(value):
            # Apply to camera if supported
            if camera.is_connected():
                camera.set_camera_property(cv2.CAP_PROP_CONTRAST, value)
        
        ui.set_test_mode_callback(on_test_mode_changed)
        ui.set_brightness_callback(on_brightness_changed)
        ui.set_contrast_callback(on_contrast_changed)
        
        # Start update thread
        running = True
        frame_count = 0
        generation_counter = 0
        
        def update_loop():
            nonlocal frame_count, generation_counter
            
            while running:
                try:
                    # Get camera frame
                    frame = camera.get_frame()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    
                    # Convert to BGR for OpenCV processing
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Detect hands
                    hand_data = hand_detector.detect_hands(frame)
                    
                    # Create custom skeleton overlay
                    skeleton_overlay = np.zeros_like(frame_bgr)
                    mediapipe_overlay = np.zeros_like(frame_bgr)
                    
                    if hand_data['detected']:
                        # Get display settings
                        settings = ui.get_display_settings()
                        
                        # Draw custom skeleton
                        skeleton_overlay = hand_detector.draw_skeleton(
                            skeleton_overlay,
                            hand_data['skeleton'],
                            draw_connections=True,
                            draw_landmarks=True
                        )
                        
                        # Draw MediaPipe visualization for comparison
                        if hand_data['landmarks']:
                            mp_drawing.draw_landmarks(
                                mediapipe_overlay,
                                hand_data['landmarks'],
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                            )
                        
                        # Draw hand center if enabled
                        if settings['show_hand_center']:
                            center = hand_data['center']
                            cv2.circle(skeleton_overlay, center, 10, (0, 255, 255), -1)
                            cv2.circle(skeleton_overlay, center, 12, (0, 0, 0), 2)
                            
                            # Add text with center coordinates
                            cv2.putText(skeleton_overlay, f"Center: {center}", 
                                      (center[0] + 15, center[1] + 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Draw finger tips
                        if settings.get('show_finger_tips', False):
                            finger_tips = hand_detector.get_finger_tip_positions(hand_data['skeleton'])
                            for finger_name, pos in finger_tips.items():
                                cv2.circle(skeleton_overlay, pos, 6, (255, 255, 0), -1)
                                cv2.putText(skeleton_overlay, finger_name[:1].upper(), 
                                          (pos[0] - 5, pos[1] + 20),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    # Update camera display with both overlays
                    ui.update_camera_display(frame_bgr, skeleton_overlay, mediapipe_overlay)
                    
                    # Update status
                    if hand_data['detected']:
                        ui.update_status(f"Hand detected at {hand_data['center']} | "
                                       f"Handedness: {hand_data['handedness']}")
                    else:
                        ui.update_status("No hand detected")
                    
                    # Generate dummy image periodically (every 2 seconds)
                    if frame_count % 60 == 0:
                        # Create a gradient test image
                        gen_img = Image.new('RGB', (512, 512))
                        pixels = gen_img.load()
                        for i in range(512):
                            for j in range(512):
                                r = int((i / 512) * 255)
                                g = int((j / 512) * 255)
                                b = int((generation_counter * 50) % 255)
                                pixels[i, j] = (r, g, b)
                        
                        info = f"Test generation {generation_counter}"
                        ui.update_generated_image(gen_img, info)
                        generation_counter += 1
                    
                    frame_count += 1
                    
                    # Small delay to control frame rate
                    time.sleep(1/30)  # Target 30 FPS
                    
                except Exception as e:
                    print(f"Error in update loop: {e}")
                    time.sleep(0.1)
        
        # Start update thread
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        
        print("\nUI started. Showing real camera feed with hand tracking.")
        print("Use the controls to adjust display settings.")
        
        # Run UI
        ui.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        running = False
        
        # Cleanup
        if camera:
            camera.release()
        if hand_detector:
            hand_detector.release()
        
        print("Test completed")

def test_hand_features():
    """Quick test to verify hand tracking features"""
    print("\nTesting hand tracking features...")
    
    camera = CameraInput()
    detector = HandDetector()
    
    print("Checking hand detection for 5 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 5:
        frame = camera.get_frame()
        if frame is None:
            continue
        
        hand_data = detector.detect_hands(frame)
        
        if hand_data['detected']:
            print(f"\rHand detected - Center: {hand_data['center']}, "
                  f"Handedness: {hand_data['handedness']}", end='')
        else:
            print("\rNo hand detected", end='')
        
        time.sleep(0.1)
    
    print("\n")
    camera.release()
    detector.release()

if __name__ == "__main__":
    # Run quick feature test first
    # test_hand_features()
    
    # Then run main UI test
    test_ui_with_real_camera()