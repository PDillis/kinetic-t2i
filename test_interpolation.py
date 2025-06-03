import numpy as np
import matplotlib.pyplot as plt
from src.prompt_system.spatial_prompts import SpatialPromptManager

def test_interpolation_visualization():
    """Test and visualize the interpolation behavior"""
    
    # Create a small test grid
    width, height = 640, 480
    manager = SpatialPromptManager(width, height)
    
    # Add some test prompts (matching your config)
    manager.add_prompt(160, 120, "landscape", 80)
    manager.add_prompt(480, 120, "digital art", 80)
    manager.add_prompt(320, 360, "cityscape", 80)
    manager.add_prompt(160, 360, "underwater", 80)
    manager.add_prompt(480, 360, "space", 80)
    
    # Create a visualization grid
    grid_size = 32
    x_step = width // grid_size
    y_step = height // grid_size
    
    # Create weight maps for each prompt
    weight_maps = {}
    for i, zone in enumerate(manager.prompt_zones):
        weight_maps[zone.prompt] = np.zeros((grid_size, grid_size))
    
    # Calculate weights for each grid position
    for grid_x in range(grid_size):
        for grid_y in range(grid_size):
            frame_x = grid_x * x_step + x_step // 2
            frame_y = grid_y * y_step + y_step // 2
            
            weights = manager.calculate_blend_weights((frame_x, frame_y))
            
            for zone, weight in weights.items():
                weight_maps[zone.prompt][grid_y, grid_x] = weight  # Note: y,x for image coordinates
    
    # Plot the weight maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (prompt, weight_map) in enumerate(weight_maps.items()):
        if i < len(axes):
            im = axes[i].imshow(weight_map, cmap='viridis', origin='upper')
            axes[i].set_title(f'"{prompt}" influence')
            axes[i].set_xlabel('X (grid)')
            axes[i].set_ylabel('Y (grid)')
            plt.colorbar(im, ax=axes[i])
            
            # Mark the prompt location
            prompt_zone = None
            for zone in manager.prompt_zones:
                if zone.prompt == prompt:
                    prompt_zone = zone
                    break
            
            if prompt_zone:
                grid_x = prompt_zone.x // x_step
                grid_y = prompt_zone.y // y_step
                axes[i].plot(grid_x, grid_y, 'r*', markersize=15, label='Prompt location')
                axes[i].legend()
    
    # Hide unused subplot
    if len(weight_maps) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('interpolation_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test a specific line through the space
    print("\nTesting interpolation along a horizontal line:")
    y_test = 240  # Middle of the image
    
    print(f"{'X Position':<12} | ", end="")
    for zone in manager.prompt_zones:
        print(f"{zone.prompt[:10]:<12} | ", end="")
    print()
    print("-" * (12 + 14 * len(manager.prompt_zones)))
    
    for x in range(0, width, 80):
        weights = manager.calculate_blend_weights((x, y_test))
        print(f"{x:<12} | ", end="")
        for zone in manager.prompt_zones:
            weight = weights.get(zone, 0.0)
            print(f"{weight:<12.3f} | ", end="")
        print()

def test_simple_2x2_case():
    """Test a simple 2x2 case like your example"""
    print("\n=== Testing Simple 2x2 Case ===")
    
    # Create simple 2x2 grid
    manager = SpatialPromptManager(4, 4)
    manager.add_prompt(1, 1, "prompt_A")  # Top-left-ish
    manager.add_prompt(3, 3, "prompt_B")  # Bottom-right-ish
    
    print("Grid positions and weights:")
    print("(weights should interpolate smoothly between prompt_A and prompt_B)")
    print()
    
    for y in range(4):
        for x in range(4):
            weights = manager.calculate_blend_weights((x, y))
            weight_A = weights.get(manager.prompt_zones[0], 0.0)
            weight_B = weights.get(manager.prompt_zones[1], 0.0)
            print(f"({x},{y}): A={weight_A:.2f}, B={weight_B:.2f}")
        print()

if __name__ == "__main__":
    print("Testing interpolation behavior...")
    test_simple_2x2_case()
    
    # Uncomment to see full visualization (requires matplotlib)
    test_interpolation_visualization()
    
    print("\nIf the weights look good, the grid computation should work correctly!")