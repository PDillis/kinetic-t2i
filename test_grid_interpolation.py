#!/usr/bin/env python3
"""
Debug script to test grid interpolation without running the full app
"""

import sys
import os
import yaml
import numpy as np

# Add src to path
sys.path.append('src')

from prompt_system.spatial_prompts import SpatialPromptManager

def test_weight_interpolation():
    """Test the weight calculation for interpolation"""
    
    print("=== Testing Weight Interpolation ===")
    
    # Load config to get prompt positions
    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create prompt manager
    cam_cfg = config["camera"]
    width, height = cam_cfg["width"], cam_cfg["height"]
    manager = SpatialPromptManager(width, height)
    
    # Add prompts from config
    for p in config["prompts"]["default_prompts"]:
        manager.add_prompt(p["x"], p["y"], p["prompt"], p.get("radius", 100))
    
    print(f"Frame size: {width}x{height}")
    print(f"Prompt zones:")
    for i, zone in enumerate(manager.prompt_zones):
        print(f"  {i}: '{zone.prompt}' at ({zone.x}, {zone.y})")
    print()
    
    # Test specific positions
    test_positions = [
        (160, 120),   # At first prompt
        (480, 120),   # At second prompt  
        (320, 120),   # Halfway between first two
        (320, 240),   # Center of image
        (50, 50),     # Corner
    ]
    
    print("Weight distribution at test positions:")
    print(f"{'Position':<15} | ", end="")
    for zone in manager.prompt_zones:
        print(f"{zone.prompt[:12]:<15} | ", end="")
    print("Sum")
    print("-" * (15 + 17 * len(manager.prompt_zones) + 5))
    
    for pos in test_positions:
        weights = manager.calculate_blend_weights(pos)
        total = sum(weights.values())
        
        print(f"{str(pos):<15} | ", end="")
        for zone in manager.prompt_zones:
            weight = weights.get(zone, 0.0)
            print(f"{weight:<15.3f} | ", end="")
        print(f"{total:.3f}")
    
    print()
    
    # Test interpolation along a line
    print("Interpolation along horizontal line (y=120):")
    print(f"{'X':<8} | ", end="")
    for zone in manager.prompt_zones:
        print(f"{zone.prompt[:8]:<10} | ", end="")
    print()
    print("-" * (8 + 12 * len(manager.prompt_zones)))
    
    for x in range(100, 500, 60):
        weights = manager.calculate_blend_weights((x, 120))
        print(f"{x:<8} | ", end="")
        for zone in manager.prompt_zones:
            weight = weights.get(zone, 0.0)
            print(f"{weight:<10.3f} | ", end="")
        print()

def test_grid_sampling():
    """Test how the grid would sample the space"""
    
    print("\n=== Testing Grid Sampling ===")
    
    # Simulate grid computation
    width, height = 640, 480
    grid_resolution = 8  # Small for testing
    
    x_step = width // grid_resolution
    y_step = height // grid_resolution
    
    print(f"Grid: {grid_resolution}x{grid_resolution}")
    print(f"Step size: {x_step}x{y_step}")
    print(f"Grid positions:")
    
    for grid_y in range(grid_resolution):
        for grid_x in range(grid_resolution):
            frame_x = grid_x * x_step + x_step // 2
            frame_y = grid_y * y_step + y_step // 2
            print(f"({grid_x},{grid_y}) -> ({frame_x},{frame_y})")
            
            if grid_x == 2 and grid_y == 2:  # Test center position
                print(f"  Center position would be at frame coordinates ({frame_x}, {frame_y})")

if __name__ == "__main__":
    test_weight_interpolation()
    test_grid_sampling()
    
    print("\nTo visualize this properly, you could plot the weight distributions.")
    print("The key is that ALL prompts should contribute to ALL positions with distance-based weights.")