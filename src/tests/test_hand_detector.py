import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.hand_tracking.hand_detector import HandDetector

class TestHandDetector(unittest.TestCase):
    """Unit tests for HandDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = HandDetector()
    
    def tearDown(self):
        """Clean up after tests"""
        self.detector.release()
    
    def test_initialization(self):
        """Test hand detector initialization"""
        self.assertIsNotNone(self.detector.hands)
        self.assertEqual(self.detector.max_num_hands, 1)
        self.assertEqual(self.detector.min_detection_confidence, 0.7)
    
    def test_empty_frame_detection(self):
        """Test detection on empty frame"""
        # Create black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect_hands(frame)
        
        self.assertFalse(result['detected'])
        self.assertEqual(result['center'], (0, 0))
        self.assertEqual(len(result['skeleton']), 0)
    
    def test_color_definitions(self):
        """Test that all colors are defined correctly"""
        required_colors = ['palm', 'thumb', 'index', 'middle', 'ring', 'pinky']
        
        for color_name in required_colors:
            self.assertIn(color_name, HandDetector.COLORS)
            color = HandDetector.COLORS[color_name]
            self.assertEqual(len(color), 3)  # BGR tuple
            
            # Check values are in valid range
            for channel in color:
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)
    
    def test_hand_center_calculation(self):
        """Test hand center calculation"""
        # Mock skeleton points
        skeleton = [(100, 100), (200, 200), (150, 150)]
        center = self.detector.get_hand_center(skeleton)
        
        self.assertEqual(center, (150, 150))
    
    def test_palm_center_calculation(self):
        """Test palm center calculation"""
        # Create mock 21-point skeleton
        skeleton = [(i*10, i*10) for i in range(21)]
        palm_center = self.detector.get_palm_center(skeleton)
        
        # Should calculate center from palm indices
        self.assertIsInstance(palm_center, tuple)
        self.assertEqual(len(palm_center), 2)
    
    def test_finger_tip_positions(self):
        """Test finger tip position extraction"""
        # Create mock 21-point skeleton
        skeleton = [(i, i) for i in range(21)]
        tips = self.detector.get_finger_tip_positions(skeleton)
        
        expected_tips = {
            'thumb': (4, 4),
            'index': (8, 8),
            'middle': (12, 12),
            'ring': (16, 16),
            'pinky': (20, 20)
        }
        
        self.assertEqual(tips, expected_tips)
    
    def test_skeleton_drawing(self):
        """Test skeleton drawing doesn't crash"""
        # Create test frame and skeleton
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        skeleton = [(i*20, i*20) for i in range(21)]
        
        # Should not raise exception
        result_frame = self.detector.draw_skeleton(frame, skeleton)
        
        self.assertEqual(result_frame.shape, frame.shape)
        # Check that something was drawn (frame should be different)
        self.assertFalse(np.array_equal(result_frame, frame))
    
    def test_gesture_features(self):
        """Test gesture feature extraction"""
        # Create mock skeleton with known positions
        skeleton = []
        for i in range(21):
            x = 100 + (i % 5) * 50
            y = 100 + (i // 5) * 50
            skeleton.append((x, y))
        
        features = self.detector.get_hand_gesture_features(skeleton)
        
        self.assertIn('palm_center', features)
        self.assertIn('finger_tips', features)
        self.assertIn('finger_distances', features)
        
        # Check finger distances are calculated
        if 'finger_distances' in features:
            self.assertEqual(len(features['finger_distances']), 5)
            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                self.assertIn(finger, features['finger_distances'])
                self.assertGreater(features['finger_distances'][finger], 0)

if __name__ == '__main__':
    unittest.main()