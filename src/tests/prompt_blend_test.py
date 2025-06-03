from typing import Dict, Any, Tuple, List
import torch
import numpy as np
from PIL import Image

from src.prompt_system.precomputed_grid import PrecomputedGrid

class PromptBlendTest:
    """Test mode for prompt blending based on hand position using pre-computed grid"""
    
    def __init__(self, prompt_manager, t2i_model, use_precomputed=True, pregenerate_images=True, grid_resolution=32):
        """
        Initialize prompt blend test
        
        Args:
            prompt_manager: SpatialPromptManager instance
            t2i_model: Text-to-image model instance
            use_precomputed: Whether to use pre-computed grid (recommended)
            grid_resolution: Grid resolution for pre-computation
        """
        self.prompt_manager = prompt_manager
        self.t2i_model = t2i_model
        self.use_precomputed = use_precomputed
        self.pregenerate_images = pregenerate_images
        
        if use_precomputed:
            self._setup_precomputed_grid(grid_resolution, pregenerate_images)
        else:
            self._setup_realtime_mode()
    
    def _setup_precomputed_grid(self, grid_resolution, pregenerate_images=False):
        """Setup pre-computed grid mode"""
        print("Setting up pre-computed grid...")
        
        frame_width = self.prompt_manager.frame_width
        frame_height = self.prompt_manager.frame_height
        
        self.grid = PrecomputedGrid(frame_width, frame_height, grid_resolution)
        
        # Try to load existing grid first
        grid_file = f"cache/grid_{frame_width}x{frame_height}_{grid_resolution}.pkl"
        if not self.grid.load_grid(grid_file):
            print("Computing new grid (this may take a minute)...")
            self.grid.compute_grid(self.prompt_manager, self.t2i_model)
            # Create cache directory if it doesn't exist
            import os
            os.makedirs("cache", exist_ok=True)
            self.grid.save_grid(grid_file)
        
        # Check if we need to pre-generate images
        has_images = any(point.generated_image is not None for point in self.grid.grid.values())
        if pregenerate_images and not has_images:
            if self.t2i_model is None:
                raise RuntimeError("Cannot pre-generate images without t2i_model")
            print("Pre-generating images...")
            self.grid.pregenerate_all_images(self.t2i_model)
            self.grid.save_grid(grid_file)
        elif has_images:
            print("Using existing pre-generated images")
    
        print("Pre-computed grid ready!")
    
    def _setup_realtime_mode(self):
        """Setup real-time mode (fallback, slower)"""
        print("Setting up real-time mode...")
        # Extract prompt strings from manager
        self.prompt_zones = self.prompt_manager.prompt_zones
        self.prompts = [pz.prompt for pz in self.prompt_zones]
        
        # Encode all prompts once
        self.embeddings = self.t2i_model.encode_prompts(self.prompts)
        print("Real-time mode ready!")
    
    def process_hand_position(self, hand_pos: Tuple[int, int]) -> Dict[str, Any]:
        """
        Process hand position and return generation parameters
        
        Args:
            hand_pos: Tuple of (x, y) coordinates
            
        Returns:
            Dict containing embeddings and metadata
        """
        if self.use_precomputed:
            return self._process_with_grid(hand_pos)
        else:
            return self._process_realtime(hand_pos)
    
    def _process_with_grid(self, hand_pos: Tuple[int, int]) -> Dict[str, Any]:
        """Process using pre-computed grid (fast)"""
        device = 'cuda' if self.t2i_model and hasattr(self.t2i_model, 'device') else 'cpu'
        return self.grid.get_embeddings_for_position(hand_pos[0], hand_pos[1], device)
    
    def _process_realtime(self, hand_pos: Tuple[int, int]) -> Dict[str, Any]:
        """Process in real-time (slower, fallback)"""
        weights_dict = self.prompt_manager.calculate_blend_weights(hand_pos)
        
        if not weights_dict:
            return {"prompt_embeds": None, "info": "No active prompts"}
        
        # Get prompt weights in the correct order
        weights = [weights_dict.get(pz, 0.0) for pz in self.prompt_zones]

        # Create proper embeddings dict for blending
        embeddings_dict = {
            'prompt_embeds': [self.embeddings['prompt_embeds'][i] for i in range(len(self.prompts))],
            'pooled_prompt_embeds': [self.embeddings['pooled_prompt_embeds'][i] for i in range(len(self.prompts))] if 'pooled_prompt_embeds' in self.embeddings else []
        }

        # Blend embeddings
        blended = self.t2i_model.blend_prompt_embeddings(embeddings_dict, weights)
        
        return {
            "prompt_embeds": blended.get('prompt_embeds'),
            "pooled_prompt_embeds": blended.get('pooled_prompt_embeds'),
            "weights": weights,
            "info": ", ".join([
                f"{pz.prompt[:30]}: {weights_dict[pz]:.2f}" 
                for pz in weights_dict
                if weights_dict[pz] > 0.05
            ])
        }


    def generate_image(self, hand_data: Dict[str, Any]) -> Image.Image:
        """Generate image (use pre-generated if available)"""
        if hasattr(self, 'grid'):
            # Try to get pre-generated image first
            if 'hand_pos' in hand_data:
                pregenerated = self.grid.get_pregenerated_image(*hand_data['hand_pos'])
                if pregenerated is not None:
                    return pregenerated
        
        # Only fallback to real-time if model is available
        if self.t2i_model is None:
            return Image.new('RGB', (512, 512), color='green')
        
        # Fallback to real-time generation
        if hand_data.get("prompt_embeds") is None:
            return Image.new('RGB', (512, 512), color='gray')
        
        embeddings = {
            'prompt_embeds': hand_data["prompt_embeds"],
            'pooled_prompt_embeds': hand_data.get("pooled_prompt_embeds")
        }
        
        return self.t2i_model.generate_with_embeddings(embeddings)