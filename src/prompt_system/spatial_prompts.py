import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2
from src.ui.color_utils import get_prompt_color_bgr

@dataclass(frozen=True)
class PromptZone:
    """Represents a prompt zone in 2D space"""
    x: int
    y: int
    prompt: str
    radius: float = 100.0

class SpatialPromptManager:
    """Manage spatial positioning and blending of prompts"""
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize spatial prompt manager
        
        Args:
            frame_width: Width of coordinate space
            frame_height: Height of coordinate space
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.prompt_zones: List[PromptZone] = []
    
    def add_prompt(self, x: int, y: int, prompt: str, radius: float = 100) -> None:
        """
        Add a prompt at specific coordinates
        
        Args:
            x: X coordinate
            y: Y coordinate
            prompt: Text prompt
            radius: Influence radius
        """
        self.prompt_zones.append(PromptZone(x=x, y=y, prompt=prompt, radius=radius))
    
    def get_active_prompts(self, hand_x: int, hand_y: int) -> Dict[str, float]:
        """
        Get prompts that influence given position (used only for precomputation)
        
        Args:
            hand_x: X coordinate of hand
            hand_y: Y coordinate of hand
            
        Returns:
            Dict mapping prompts to their influence weights
        """
        weights = self.calculate_blend_weights((hand_x, hand_y))
        return {zone.prompt: weight for zone, weight in weights.items()}
    
    def calculate_blend_weights(self, hand_pos: Tuple[int, int]) -> Dict[PromptZone, float]:
        """
        Calculate blending weights for ALL prompts based on distance (for entire grid interpolation)
        NOTE: This is only used during precomputation phase, not at runtime!
        
        Args:
            hand_pos: Tuple of (x, y) coordinates
            
        Returns:
            Dict mapping ALL prompt zones to their weights (normalized to sum to 1)
        """
        weights = {}
        hx, hy = hand_pos

        # Calculate weights for ALL zones based on distance
        for zone in self.prompt_zones:
            dist = np.sqrt((hx - zone.x)**2 + (hy - zone.y)**2)
            
            # Use inverse distance weighting with falloff
            # The radius is just for visualization - all prompts contribute everywhere
            if dist < 1.0:  # Avoid division by zero at exact center
                weight = 1.0
            else:
                # Inverse distance with some falloff control
                weight = 1.0 / (1.0 + (dist / 100.0)**2)  # Adjust 100.0 to control falloff
            
            weights[zone] = weight

        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            for zone in weights:
                weights[zone] /= total

        return weights
        
    def get_blended_prompt(self, hand_pos: Tuple[int, int]) -> str:
        """
        Get single blended prompt string (only used for debugging/precomputation)
        NOTE: At runtime, use the precomputed grid instead!
        
        Args:
            hand_pos: Tuple of (x, y) coordinates
            
        Returns:
            Blended prompt string with weights
        """
        weights = self.calculate_blend_weights(hand_pos)
        if not weights:
            return "no active prompts"
        
        # Create weighted prompt string
        weighted_parts = []
        for zone, weight in weights.items():
            if weight > 0.01:  # Only include significant weights
                weighted_parts.append(f"({zone.prompt}:{weight:.2f})")
        
        return " ".join(weighted_parts)
    
    def visualize_prompt_zones(self, frame: np.ndarray, show_full_zones: bool = True, show_text: bool = False) -> np.ndarray:
        """
        Draw prompt zones on frame with gradient visualization
        
        Args:
            frame: BGR image as numpy array (from OpenCV camera)
            show_full_zones: If True, show gradient zones. If False, show only centers.
            show_text: If True, show prompt text labels
            
        Returns:
            Frame with prompt zone overlay (still in BGR)
        """
        overlay = frame.copy()
        
        if show_full_zones:
            # Create gradient overlays for each zone
            overlay = self._create_gradient_overlay(overlay)
        
        # Always draw center points
        for i, zone in enumerate(self.prompt_zones):
            color = get_prompt_color_bgr(i)
            
            # Draw center point (the actual prompt anchor)
            cv2.circle(overlay, (zone.x, zone.y), 8, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (zone.x, zone.y), 10, (0, 0, 0), 2, cv2.LINE_AA)  # Black outline
            
            # Draw text if enabled
            if show_text:
                # text = zone.prompt[:25] + "..." if len(zone.prompt) > 25 else zone.prompt
                text = zone.prompt
                
                # Calculate text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3 if len(zone.prompt) > 25 else 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Text position (above the center)
                text_x = zone.x - text_width // 2
                text_y = zone.y - 20
                
                # Ensure text stays within frame bounds
                text_x = max(5, min(text_x, frame.shape[1] - text_width - 5))
                text_y = max(text_height + 5, min(text_y, frame.shape[0] - 5))
                
                # Draw text background rectangle
                bg_x1 = text_x - 3
                bg_y1 = text_y - text_height - 3
                bg_x2 = text_x + text_width + 3
                bg_y2 = text_y + baseline + 3
                
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
                
                # Draw text
                cv2.putText(overlay, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return overlay

    def _create_gradient_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Create gradient overlay showing prompt influence"""
        height, width = frame.shape[:2]
        
        # Create influence map for each prompt
        influence_maps = []
        
        for i, zone in enumerate(self.prompt_zones):
            # Create coordinate grids
            y_coords, x_coords = np.ogrid[:height, :width]
            
            # Calculate distance from zone center
            distances = np.sqrt((x_coords - zone.x)**2 + (y_coords - zone.y)**2)
            
            # Create Gaussian-like influence (same formula as calculate_blend_weights)
            influence = np.where(distances < 1.0, 1.0, 1.0 / (1.0 + (distances / 200.0)**2))
            
            # Normalize to 0-1 range
            influence = influence / influence.max() if influence.max() > 0 else influence
            
            # Get color for this zone
            color = get_prompt_color_bgr(i)
            
            # Create colored influence map
            colored_influence = np.zeros((height, width, 3), dtype=np.uint8)
            for c in range(3):
                colored_influence[:, :, c] = (influence * color[c]).astype(np.uint8)
            
            influence_maps.append((colored_influence, influence))
        
        # Blend all influence maps
        if influence_maps:
            # Normalize influences so they sum to 1 at each pixel
            total_influence = np.sum([inf_map[1] for inf_map in influence_maps], axis=0)
            total_influence = np.where(total_influence == 0, 1, total_influence)  # Avoid division by zero
            
            # Create blended overlay
            blended_overlay = np.zeros_like(frame)
            for colored_influence, influence in influence_maps:
                normalized_influence = influence / total_influence
                for c in range(3):
                    blended_overlay[:, :, c] += (colored_influence[:, :, c] * normalized_influence).astype(np.uint8)
            
            # Apply alpha blending with original frame
            alpha = 0.6  # Transparency of gradient overlay
            overlay = cv2.addWeighted(frame, 1 - alpha, blended_overlay, alpha, 0)
            return overlay
        
        return frame
    
    def get_zone_info(self, hand_pos: Tuple[int, int]) -> str:
        """
        Get information string about active zones at hand position
        NOTE: Only used for debugging/precomputation. At runtime, info comes from grid.
        
        Args:
            hand_pos: Tuple of (x, y) coordinates
            
        Returns:
            Information string about active zones
        """
        weights = self.calculate_blend_weights(hand_pos)
        if not weights:
            return "No active zones"
        
        info_parts = []
        for zone, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.01:
                info_parts.append(f"{zone.prompt[:20]}: {weight:.1%}")
        
        return " | ".join(info_parts)
    
    def is_position_in_any_zone(self, hand_pos: Tuple[int, int]) -> bool:
        """
        Check if position is within any prompt zone
        NOTE: Only used for debugging/precomputation. At runtime, check grid instead.
        
        Args:
            hand_pos: Tuple of (x, y) coordinates
            
        Returns:
            True if position is within at least one zone
        """
        weights = self.calculate_blend_weights(hand_pos)
        return len(weights) > 0
    
    def get_nearest_zone(self, hand_pos: Tuple[int, int]) -> Optional[PromptZone]:
        """
        Get the nearest prompt zone to a position
        
        Args:
            hand_pos: Tuple of (x, y) coordinates
            
        Returns:
            Nearest PromptZone or None if no zones exist
        """
        if not self.prompt_zones:
            return None
        
        hx, hy = hand_pos
        min_distance = float('inf')
        nearest_zone = None
        
        for zone in self.prompt_zones:
            dist = np.sqrt((hx - zone.x)**2 + (hy - zone.y)**2)
            if dist < min_distance:
                min_distance = dist
                nearest_zone = zone
        
        return nearest_zone
    
    def clear_zones(self) -> None:
        """Clear all prompt zones"""
        self.prompt_zones.clear()
    
    def remove_zone(self, index: int) -> bool:
        """
        Remove a prompt zone by index
        
        Args:
            index: Index of zone to remove
            
        Returns:
            True if zone was removed, False if index invalid
        """
        if 0 <= index < len(self.prompt_zones):
            self.prompt_zones.pop(index)
            return True
        return False