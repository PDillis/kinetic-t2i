import mediapipe as mp
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandDetector:
    """Detect and track hand landmarks using MediaPipe"""
    
    # MediaPipe hand landmark indices for each finger
    THUMB_INDICES = [1, 2, 3, 4]
    INDEX_INDICES = [5, 6, 7, 8]
    MIDDLE_INDICES = [9, 10, 11, 12]
    RING_INDICES = [13, 14, 15, 16]
    PINKY_INDICES = [17, 18, 19, 20]
    PALM_INDICES = [0, 1, 5, 9, 13, 17]  # Wrist and base of each finger
    
    # Finger colors (hex converted to BGR for OpenCV)
    COLORS = {
        'palm': (0, 76, 242),        # #F24C00 -> BGR
        'thumb': (255, 141, 133),    # #FF858D -> BGR
        'index': (212, 212, 255),    # #FFD4D4 -> BGR
        'middle': (229, 159, 255),   # #FF9FE5 -> BGR
        'ring': (170, 80, 43),       # #2B50AA -> BGR
        'pinky': (39, 39, 39),       # #272727 -> BGR
    }
    
    # Connection pairs for drawing skeleton
    HAND_CONNECTIONS = [
        # Palm connections
        (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),     # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        # Palm base connections
        (5, 9), (9, 13), (13, 17)
    ]

    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.7):
        """
        Initialize MediaPipe hand detector
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands: Optional[Any] = None
        self._initialize_detector()
    
    def _initialize_detector(self) -> None:
        """Initialize MediaPipe hands solution"""
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=0.5
            )
            logger.info("Hand detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize hand detector: {e}")
            raise
    
    def detect_hands(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect hands in frame and return landmarks
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Dictionary containing:
                - 'detected': bool indicating if hands were found
                - 'landmarks': list of hand landmarks
                - 'center': tuple of (x, y) for primary hand center
                - 'skeleton': list of keypoint coordinates
                - 'normalized_landmarks': normalized landmark positions
        """
        if self.hands is None:
            logger.error("Hand detector not initialized")
            return self._empty_result()
        
        try:
            # Process the frame
            results = self.hands.process(frame)
            
            if results.multi_hand_landmarks:
                # Get the first hand (primary hand)
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract pixel coordinates
                height, width = frame.shape[:2]
                skeleton = []
                normalized_landmarks = []
                
                for landmark in hand_landmarks.landmark:
                    # Pixel coordinates
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    skeleton.append((x, y))
                    
                    # Normalized coordinates
                    normalized_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                # Calculate hand center
                center = self.get_hand_center(skeleton)
                
                return {
                    'detected': True,
                    'landmarks': hand_landmarks,
                    'center': center,
                    'skeleton': skeleton,
                    'normalized_landmarks': normalized_landmarks,
                    'handedness': results.multi_handedness[0].classification[0].label if results.multi_handedness else 'Unknown'
                }
            else:
                return self._empty_result()
                
        except Exception as e:
            logger.error(f"Error detecting hands: {e}")
            return self._empty_result()

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result dictionary"""
        return {
            'detected': False,
            'landmarks': None,
            'center': (0, 0),
            'skeleton': [],
            'normalized_landmarks': [],
            'handedness': 'Unknown'
        }
    
    def get_hand_center(self, skeleton: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Calculate center point of hand from skeleton points
        
        Args:
            skeleton: List of (x, y) coordinates for each keypoint
            
        Returns:
            Tuple of (x, y) coordinates for hand center
        """
        if not skeleton:
            return (0, 0)
        
        # Calculate center as average of all points
        x_coords = [pt[0] for pt in skeleton]
        y_coords = [pt[1] for pt in skeleton]
        
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        return (center_x, center_y)
    
    def get_palm_center(self, skeleton: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Calculate center of palm (using wrist and finger bases)
        
        Args:
            skeleton: List of (x, y) coordinates for each keypoint
            
        Returns:
            Tuple of (x, y) coordinates for palm center
        """
        if len(skeleton) < 21:
            return (0, 0)
        
        palm_points = [skeleton[i] for i in self.PALM_INDICES]
        x_coords = [pt[0] for pt in palm_points]
        y_coords = [pt[1] for pt in palm_points]
        
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        return (center_x, center_y)

    def get_finger_tip_positions(self, skeleton: List[Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
        """
        Get positions of all finger tips
        
        Args:
            skeleton: List of (x, y) coordinates for each keypoint
            
        Returns:
            Dictionary mapping finger names to tip positions
        """
        if len(skeleton) < 21:
            return {}
        
        return {
            'thumb': skeleton[4],
            'index': skeleton[8],
            'middle': skeleton[12],
            'ring': skeleton[16],
            'pinky': skeleton[20]
        }

    def get_hand_skeleton(self, landmarks: Any, frame_width: int = 640, frame_height: int = 480) -> List[Tuple[int, int]]:
        """
        Extract hand skeleton coordinates from landmarks
        
        Args:
            landmarks: MediaPipe hand landmarks object
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels
            
        Returns:
            List of (x, y) coordinates for each keypoint
        """
        if landmarks is None:
            return []
        
        skeleton = []
        
        # Extract coordinates for each landmark
        for landmark in landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            skeleton.append((x, y))
        
        return skeleton
    
    def draw_skeleton(self, frame: np.ndarray, skeleton: List[Tuple[int, int]], 
                     draw_connections: bool = True, draw_landmarks: bool = True) -> np.ndarray:
        """
        Draw hand skeleton overlay on frame with custom colors
        
        Args:
            frame: RGB image as numpy array
            skeleton: List of (x, y) coordinates for each keypoint
            draw_connections: Whether to draw connections between points
            draw_landmarks: Whether to draw landmark points
            
        Returns:
            Frame with skeleton overlay
        """
        if len(skeleton) < 21:
            return frame
        
        frame_copy = frame.copy()
        
        # Draw connections first (so points appear on top)
        if draw_connections:
            # Draw palm connections
            for connection in self.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(skeleton) and end_idx < len(skeleton):
                    start_point = skeleton[start_idx]
                    end_point = skeleton[end_idx]
                    
                    # Determine color based on which finger the connection belongs to
                    color = self._get_connection_color(start_idx, end_idx)
                    
                    cv2.line(frame_copy, start_point, end_point, color, 2, cv2.LINE_AA)
        
        # Draw landmarks
        if draw_landmarks:
            # Draw palm points
            for idx in self.PALM_INDICES:
                if idx < len(skeleton):
                    cv2.circle(frame_copy, skeleton[idx], 5, self.COLORS['palm'], -1, cv2.LINE_AA)
                    cv2.circle(frame_copy, skeleton[idx], 6, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Draw finger points
            finger_groups = [
                ('thumb', self.THUMB_INDICES),
                ('index', self.INDEX_INDICES),
                ('middle', self.MIDDLE_INDICES),
                ('ring', self.RING_INDICES),
                ('pinky', self.PINKY_INDICES)
            ]
            
            for finger_name, indices in finger_groups:
                color = self.COLORS[finger_name]
                for idx in indices:
                    if idx < len(skeleton):
                        # Larger circles for fingertips
                        size = 6 if idx in [4, 8, 12, 16, 20] else 4
                        cv2.circle(frame_copy, skeleton[idx], size, color, -1, cv2.LINE_AA)
                        cv2.circle(frame_copy, skeleton[idx], size + 1, (0, 0, 0), 1, cv2.LINE_AA)
        
        return frame_copy

    def _get_connection_color(self, start_idx: int, end_idx: int) -> Tuple[int, int, int]:
        """
        Get color for a connection based on which finger it belongs to
        
        Args:
            start_idx: Start point index
            end_idx: End point index
            
        Returns:
            BGR color tuple
        """
        # Check which finger the connection belongs to
        if start_idx in self.THUMB_INDICES or end_idx in self.THUMB_INDICES:
            return self.COLORS['thumb']
        elif start_idx in self.INDEX_INDICES or end_idx in self.INDEX_INDICES:
            return self.COLORS['index']
        elif start_idx in self.MIDDLE_INDICES or end_idx in self.MIDDLE_INDICES:
            return self.COLORS['middle']
        elif start_idx in self.RING_INDICES or end_idx in self.RING_INDICES:
            return self.COLORS['ring']
        elif start_idx in self.PINKY_INDICES or end_idx in self.PINKY_INDICES:
            return self.COLORS['pinky']
        else:
            return self.COLORS['palm']

    def get_hand_gesture_features(self, skeleton: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Extract gesture-related features from hand skeleton
        
        Args:
            skeleton: List of (x, y) coordinates for each keypoint
            
        Returns:
            Dictionary of gesture features
        """
        if len(skeleton) < 21:
            return {}
        
        features = {}
        
        # Calculate distances between fingertips and palm center
        palm_center = self.get_palm_center(skeleton)
        finger_tips = self.get_finger_tip_positions(skeleton)
        
        features['palm_center'] = palm_center
        features['finger_tips'] = finger_tips
        
        # Calculate finger distances from palm
        features['finger_distances'] = {}
        for finger, tip_pos in finger_tips.items():
            distance = np.sqrt((tip_pos[0] - palm_center[0])**2 + 
                             (tip_pos[1] - palm_center[1])**2)
            features['finger_distances'][finger] = distance
        
        # Calculate hand span (distance between thumb and pinky)
        if 'thumb' in finger_tips and 'pinky' in finger_tips:
            span = np.sqrt((finger_tips['thumb'][0] - finger_tips['pinky'][0])**2 + 
                          (finger_tips['thumb'][1] - finger_tips['pinky'][1])**2)
            features['hand_span'] = span
        
        return features

    def release(self) -> None:
        """Release MediaPipe resources"""
        if self.hands is not None:
            self.hands.close()
            self.hands = None
            logger.info("Hand detector released")