import numpy as np
import torch
from typing import Dict, Union, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import time
from pathlib import Path

# Taken from: https://github.com/NVlabs/stylegan3
def format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)

@dataclass
class GridPoint:
    """A single point in the pre-computed grid"""
    x: int
    y: int
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    blended_prompt: str = ""  # For debugging/display
    generated_image: Optional[Any] = None  # Store PIL Image

class PrecomputedGrid:
    """Pre-computed grid of interpolated prompt embeddings"""
    
    def __init__(self, width: int, height: int, grid_resolution: int = 32):
        """
        Initialize pre-computed grid
        
        Args:
            width: Frame width
            height: Frame height  
            grid_resolution: Number of grid points per dimension (32x32 = 1024 points)
        """
        self.width = width
        self.height = height
        self.grid_resolution = grid_resolution
        
        # Calculate grid spacing
        self.x_step = width // grid_resolution
        self.y_step = height // grid_resolution
        
        # Storage for pre-computed embeddings
        self.grid: Dict[Tuple[int, int], GridPoint] = {}
        self.is_computed = False
    
    def pregenerate_all_images(self, t2i_model, save_images: bool = True) -> None:
        """Pre-generate all images for the grid"""
        if not self.is_computed:
            raise RuntimeError("Grid not computed yet!")
        
        print(f"Pre-generating {len(self.grid)} images...")
        start_time = time.time()
        
        for i, ((grid_x, grid_y), grid_point) in enumerate(self.grid.items()):
            if grid_point.prompt_embeds is not None:
                # Generate image
                image = t2i_model.generate_with_embeddings({
                    'prompt_embeds': grid_point.prompt_embeds,
                    'pooled_prompt_embeds': grid_point.pooled_prompt_embeds
                })
                
                # Store in grid point
                grid_point.generated_image = image
            
            # Progress
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                progress = (i + 1) / len(self.grid)
                eta = elapsed / progress - elapsed
                eta = format_time(eta)
                print(f"Progress: {i+1}/{len(self.grid)} ({progress*100:.1f}%) ETA: {eta}")
        
        elapsed = time.time() - start_time
        print(f"Pre-generation complete! {len(self.grid)} images in {elapsed:.2f}s")

    def get_pregenerated_image(self, x: int, y: int) -> Optional[Any]:
        """Get pre-generated image for position"""
        if not self.is_computed:
            return None
        
        grid_x = min(max(x // self.x_step, 0), self.grid_resolution - 1)
        grid_y = min(max(y // self.y_step, 0), self.grid_resolution - 1)
        
        grid_point = self.grid.get((grid_x, grid_y))
        return grid_point.generated_image if grid_point else None

    def compute_grid(self, prompt_manager, t2i_model) -> None:
        """
        Pre-compute all grid points with interpolated embeddings using SLERP
        
        Args:
            prompt_manager: SpatialPromptManager instance
            t2i_model: Text-to-image model for encoding prompts
        """
        print(f"Computing {self.grid_resolution}x{self.grid_resolution} grid...")
        
        # First, encode all unique prompts
        unique_prompts = list(set(pz.prompt for pz in prompt_manager.prompt_zones))
        print(f"Encoding {len(unique_prompts)} unique prompts...")
        
        # Get embeddings for all prompts at once
        all_embeddings = t2i_model.encode_prompts(unique_prompts)
        prompt_to_embeddings = {}
        
        for i, prompt in enumerate(unique_prompts):
            prompt_to_embeddings[prompt] = {
                'prompt_embeds': all_embeddings['prompt_embeds'][i] if 'prompt_embeds' in all_embeddings else None,
                'pooled_prompt_embeds': all_embeddings['pooled_prompt_embeds'][i] if 'pooled_prompt_embeds' in all_embeddings else None,
            }
        
        # Now compute grid points
        total_points = self.grid_resolution * self.grid_resolution
        computed_points = 0
        
        start_time = time.time()
        
        print(f"Sample weight calculation at center:")
        center_x, center_y = self.width // 2, self.height // 2
        sample_weights = prompt_manager.calculate_blend_weights((center_x, center_y))
        for zone, weight in sample_weights.items():
            print(f"  {zone.prompt}: {weight:.3f}")
        print()
        
        for grid_x in range(self.grid_resolution):
            for grid_y in range(self.grid_resolution):
                # Convert grid coordinates to frame coordinates
                frame_x = grid_x * self.x_step + self.x_step // 2
                frame_y = grid_y * self.y_step + self.y_step // 2
                
                # Get blend weights for this position (ALL prompts contribute)
                weights_dict = prompt_manager.calculate_blend_weights((frame_x, frame_y))
                
                if not weights_dict:
                    print(f"Warning: No weights at position ({frame_x}, {frame_y})")
                    continue
                
                # Get embeddings and weights for ALL prompts (not just significant ones)
                embeddings_list = []
                weights_list = []
                prompt_parts = []
                
                # Process ALL zones in the same order
                for zone in prompt_manager.prompt_zones:
                    weight = weights_dict.get(zone, 0.0)
                    zone_embeddings = prompt_to_embeddings[zone.prompt]
                    
                    if zone_embeddings['prompt_embeds'] is not None:
                        embeddings_list.append(zone_embeddings)
                        weights_list.append(weight)  # Include ALL weights, even small ones
                        # if weight > 0.05:  # Only show significant weights in info string
                        prompt_parts.append(f"{zone.prompt[:15]}:{weight:.2f}")
                
                if embeddings_list and len(embeddings_list) > 0:
                    # Create proper embedding dictionaries for blending
                    embeddings_dict = {
                        'prompt_embeds': [e['prompt_embeds'] for e in embeddings_list if e['prompt_embeds'] is not None],
                        'pooled_prompt_embeds': [e['pooled_prompt_embeds'] for e in embeddings_list if e['pooled_prompt_embeds'] is not None]
                    }
                    
                    # Blend embeddings using SLERP with ALL weights
                    blended = t2i_model.blend_prompt_embeddings(embeddings_dict, weights_list)
                    blended_prompt = " + ".join(prompt_parts) if prompt_parts else f"blend({len(weights_list)} prompts)"
                    
                    # Store in grid
                    self.grid[(grid_x, grid_y)] = GridPoint(
                        x=frame_x,
                        y=frame_y,
                        prompt_embeds=blended.get('prompt_embeds') if isinstance(blended, dict) else blended,
                        pooled_prompt_embeds=blended.get('pooled_prompt_embeds') if isinstance(blended, dict) else None,
                        blended_prompt=blended_prompt
                    )
                else:
                    print(f"Error: No valid embeddings at grid position ({grid_x}, {grid_y})")
                    continue
                
                computed_points += 1
                
                # Progress update
                if computed_points % 100 == 0:
                    elapsed = time.time() - start_time
                    progress = computed_points / total_points
                    eta = elapsed / progress - elapsed if progress > 0 else 0
                    eta = format_time(eta)
                    print(f"Progress: {computed_points}/{total_points} ({progress*100:.1f}%) ETA: {eta}")
        
        elapsed = time.time() - start_time
        print(f"Grid computation complete! {total_points} points in {elapsed:.2f}s")
        self.is_computed = True

        # Clear GPU memory after computation
        print("Clearing GPU memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Also clear the original embeddings from t2i_model if it has a cache
        if hasattr(t2i_model, 'clear_cache'):
            t2i_model.clear_cache()
    
    def get_embeddings_for_position(self, x: int, y: int, device='cuda') -> Dict[str, Any]:
        """
        Get pre-computed embeddings for a frame position
        
        Args:
            x: X coordinate in frame
            y: Y coordinate in frame
            device: Device to move tensors to ('cuda' or 'cpu')
            
        Returns:
            Dictionary containing prompt embeddings and info
        """
        if not self.is_computed:
            raise RuntimeError("Grid not computed yet!")
        
        # Convert frame coordinates to grid coordinates
        grid_x = min(max(x // self.x_step, 0), self.grid_resolution - 1)
        grid_y = min(max(y // self.y_step, 0), self.grid_resolution - 1)
        
        grid_point = self.grid.get((grid_x, grid_y))
        if grid_point is None:
            return {
                'prompt_embeds': None,
                'pooled_prompt_embeds': None,
                'info': 'no data'
            }
        
        # Move tensors to requested device only when needed
        prompt_embeds = grid_point.prompt_embeds.to(device) if grid_point.prompt_embeds is not None else None
        pooled_embeds = grid_point.pooled_prompt_embeds.to(device) if grid_point.pooled_prompt_embeds is not None else None
        
        return {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_embeds,
            'info': grid_point.blended_prompt
        }
    
    def save_grid(self, filepath: str) -> None:
        if not self.is_computed:
            raise RuntimeError("Grid not computed yet!")
        
        # Move all tensors to CPU before saving
        cpu_grid = {}
        for key, point in self.grid.items():
            cpu_grid[key] = GridPoint(
                x=point.x,
                y=point.y,
                prompt_embeds=point.prompt_embeds.cpu() if point.prompt_embeds is not None else None,
                pooled_prompt_embeds=point.pooled_prompt_embeds.cpu() if point.pooled_prompt_embeds is not None else None,
                blended_prompt=point.blended_prompt,
                generated_image=point.generated_image
            )
        
        data = {
            'width': self.width,
            'height': self.height,
            'grid_resolution': self.grid_resolution,
            'grid': cpu_grid,
            'has_pregenerated_images': any(point.generated_image is not None for point in self.grid.values())
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        # Replace the original grid with CPU version and clear GPU memory
        self.grid = cpu_grid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Grid saved to {filepath} (with images: {data['has_pregenerated_images']})")
        
    def load_grid(self, filepath: str) -> bool:
        """Load pre-computed grid from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Verify compatibility
            if (data['width'] != self.width or 
                data['height'] != self.height or 
                data['grid_resolution'] != self.grid_resolution):
                print("Grid dimensions don't match, recomputation needed")
                return False
            
            self.grid = data['grid']
            self.is_computed = True
            has_images = data.get('has_pregenerated_images', False)
            print(f"Grid loaded from {filepath} (with images: {has_images})")
            return True
            
        except (FileNotFoundError, pickle.PickleError) as e:
            print(f"Failed to load grid: {e}")
            return False