"""
Test script for hand tracking module
Displays camera feed with colored hand skeleton overlay
Press 'q' to quit, 's' to save screenshot, 'space' to toggle skeleton
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera.camera_input import CameraInput
from src.hand_tracking.hand_detector import HandDetector

def draw_info_overlay(frame: np.ndarray, hand_data: Dict, fps: float) -> np.ndarray:
    """Draw information overlay on frame"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay for info
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # FPS and detection status
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 35), 
                font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    if hand_data['detected']:
        cv2.putText(frame, "Hand Detected", (20, 60), 
                    font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        center = hand_data['center']
        cv2.putText(frame, f"Center: ({center[0]}, {center[1]})", (20, 85), 
                    font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Hand: {hand_data['handedness']}", (20, 110), 
                    font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Draw center point
        cv2.circle(frame, center, 8, (0, 255, 255), -1)
        cv2.circle(frame, center, 9, (0, 0, 0), 1)
    else:
        cv2.putText(frame, "No Hand Detected", (20, 60), 
                    font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    
    # Controls
    cv2.putText(frame, "Q: Quit | S: Screenshot | Space: Toggle Skeleton", 
                (20, 135), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def draw_finger_labels(frame: np.ndarray, hand_data: Dict) -> np.ndarray:
    """Draw labels for each finger tip"""
    if not hand_data['detected'] or len(hand_data['skeleton']) < 21:
        return frame
    
    skeleton = hand_data['skeleton']
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Finger tip indices and labels
    finger_tips = [
        (4, "Thumb"),
        (8, "Index"),
        (12, "Middle"),
        (16, "Ring"),
        (20, "Pinky")
    ]
    
    for idx, label in finger_tips:
        if idx < len(skeleton):
            pos = skeleton[idx]
            # Offset label position slightly above fingertip
            label_pos = (pos[0] - 20, pos[1] - 10)
            cv2.putText(frame, label, label_pos, font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def draw_color_legend(frame: np.ndarray) -> np.ndarray:
    """Draw color legend for fingers"""
    height, width = frame.shape[:2]
    
    # Create legend background
    overlay = frame.copy()
    legend_x = width - 150
    cv2.rectangle(overlay, (legend_x, 10), (width - 10, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Finger Colors:", (legend_x + 10, 30), 
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    colors_list = [
        ("Palm", HandDetector.COLORS['palm']),
        ("Thumb", HandDetector.COLORS['thumb']),
        ("Index", HandDetector.COLORS['index']),
        ("Middle", HandDetector.COLORS['middle']),
        ("Ring", HandDetector.COLORS['ring']),
        ("Pinky", HandDetector.COLORS['pinky'])
    ]
    
    y_offset = 50
    for name, color in colors_list:
        cv2.circle(frame, (legend_x + 20, y_offset), 6, color, -1)
        cv2.putText(frame, name, (legend_x + 35, y_offset + 5), 
                    font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 20
    
    return frame

def test_hand_tracking():
    """Test hand tracking with camera feed"""
    print("Hand Tracking Test")
    print("-" * 50)
    print("Controls:")
    print("  Q     - Quit")
    print("  S     - Save screenshot")
    print("  SPACE - Toggle skeleton display")
    print("  L     - Toggle finger labels")
    print("  C     - Toggle color legend")
    print("  D     - Toggle connections")
    print("  P     - Toggle landmark points")
    
    # Initialize components
    try:
        camera = CameraInput(width=640, height=480)
        hand_detector = HandDetector(max_num_hands=1)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    # Create window
    window_name = "Hand Tracking Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # State variables
    show_skeleton = True
    show_labels = False
    show_legend = True
    show_connections = True
    show_landmarks = True
    screenshot_count = 0
    
    # FPS calculation
    fps = 0.0
    frame_count = 0
    start_time = cv2.getTickCount()
    
    try:
        while True:
            # Get frame
            frame = camera.get_frame()
            if frame is None:
                print("Failed to get frame")
                break
            
            # Convert to BGR for display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect hands
            hand_data = hand_detector.detect_hands(frame)
            
            # Draw skeleton if enabled and hand detected
            if show_skeleton and hand_data['detected']:
                frame_bgr = hand_detector.draw_skeleton(
                    frame_bgr, 
                    hand_data['skeleton'],
                    draw_connections=show_connections,
                    draw_landmarks=show_landmarks
                )
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = cv2.getTickCount()
                fps = 30 / ((end_time - start_time) / cv2.getTickFrequency())
                start_time = end_time
            
            # Draw overlays
            frame_bgr = draw_info_overlay(frame_bgr, hand_data, fps)
            
            if show_labels and hand_data['detected']:
                frame_bgr = draw_finger_labels(frame_bgr, hand_data)
            
            if show_legend:
                frame_bgr = draw_color_legend(frame_bgr)
            
            # Display frame
            cv2.imshow(window_name, frame_bgr)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"hand_tracking_screenshot_{timestamp}_{screenshot_count}.png"
                cv2.imwrite(filename, frame_bgr)
                print(f"Screenshot saved: {filename}")
                screenshot_count += 1
            elif key == ord(' '):
                show_skeleton = not show_skeleton
                print(f"Skeleton display: {'ON' if show_skeleton else 'OFF'}")
            elif key == ord('l'):
                show_labels = not show_labels
                print(f"Finger labels: {'ON' if show_labels else 'OFF'}")
            elif key == ord('c'):
                show_legend = not show_legend
                print(f"Color legend: {'ON' if show_legend else 'OFF'}")
            elif key == ord('d'):
                show_connections = not show_connections
                print(f"Connections: {'ON' if show_connections else 'OFF'}")
            elif key == ord('p'):
                show_landmarks = not show_landmarks
                print(f"Landmark points: {'ON' if show_landmarks else 'OFF'}")
    
    finally:
        # Cleanup
        camera.release()
        hand_detector.release()
        cv2.destroyAllWindows()
        print("\nHand tracking test completed")

def test_gesture_features():
    """Test gesture feature extraction"""
    print("\nGesture Feature Test")
    print("-" * 50)
    
    camera = CameraInput()
    hand_detector = HandDetector()
    
    print("Show your hand to the camera...")
    
    for i in range(100):  # Test for 100 frames
        frame = camera.get_frame()
        if frame is None:
            continue
        
        hand_data = hand_detector.detect_hands(frame)
        
        if hand_data['detected'] and i % 10 == 0:  # Print every 10 frames
            features = hand_detector.get_hand_gesture_features(hand_data['skeleton'])
            
            print(f"\nFrame {i}:")
            print(f"  Palm center: {features.get('palm_center', 'N/A')}")
            print(f"  Hand span: {features.get('hand_span', 'N/A'):.2f}" if 'hand_span' in features else "  Hand span: N/A")
            
            if 'finger_distances' in features:
                print("  Finger distances from palm:")
                for finger, distance in features['finger_distances'].items():
                    print(f"    {finger}: {distance:.2f}")
    
    camera.release()
    hand_detector.release()

if __name__ == "__main__":
    # Run main test
    test_hand_tracking()
    
    # Optionally run gesture feature test
    # test_gesture_features()