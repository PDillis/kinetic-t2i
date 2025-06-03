import torch
import torch.nn as nn
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from PIL import Image
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
import gc
from dataclasses import dataclass
import time

from .base_model import BaseT2IModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for image generation"""
    num_inference_steps: int = 1  # SDXL-Turbo default
    guidance_scale: float = 0.0  # No CFG for turbo
    height: int = 512
    width: int = 512
    seed: Optional[int] = None

class StableDiffusionModel(BaseT2IModel):
    """SDXL-Turbo implementation of T2I model"""

    def __init__(self, 
                 model_name: str = "stabilityai/sdxl-turbo",
                 use_fp16: bool = True,
                 compile_unet: bool = False):
        """
        Initialize SDXL-Turbo model
        
        Args:
            model_name: HuggingFace model identifier
            use_fp16: Use fp16 precision for faster inference
            compile_unet: Compile UNet with torch.compile for speed (requires PyTorch 2.0+)
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.compile_unet = compile_unet and hasattr(torch, 'compile')
        self.pipe: Optional[DiffusionPipeline] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Cache for prompt embeddings
        self.embedding_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.cache_size = 100
        
        # Layer corruption state (for testing)
        self.corrupted_layers: Dict[int, float] = {}
        
        logger.info(f"Initialized SDXL-Turbo model for device: {self.device}")
    
    def load_model(self) -> None:
        """Load SDXL-Turbo pipeline"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load pipeline
            if self.use_fp16:
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
            else:
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    use_safetensors=True,
                )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Optimize VAE for speed
            self.pipe.upcast_vae()
            
            # Compile UNet for faster inference (PyTorch 2.0+)
            if self.compile_unet and hasattr(torch, 'compile'):
                logger.info("Compiling UNet for faster inference...")
                try:
                    self.pipe.unet.to(memory_format=torch.channels_last)
                    self.pipe.vae.to(memory_format=torch.channels_last)

                    self.pipe.unet = torch.compile(
                        self.pipe.unet, 
                        mode="default", 
                        fullgraph=False,  # Allow graph breaks
                        dynamic=True,  # Handle dynamic shapes better
                    )
                    logger.info("UNet compiled successfully")

                    self.pipe.vae.decode = torch.compile(
                        self.pipe.vae.decode,
                        mode="default",
                        fullgraph=False,
                        dynamic=True,
                    )

                except Exception as e:
                    logger.warning(f"Failed to compile UNet: {e}")
            
            # Enable memory efficient attention if available
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            elif hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            
            # Store references to model components
            self.text_encoder = self.pipe.text_encoder
            self.text_encoder_2 = self.pipe.text_encoder_2
            self.unet = self.pipe.unet
            self.vae = self.pipe.vae
            
            # Set up for fast inference
            self.pipe.set_progress_bar_config(disable=True)
            
            logger.info(f"Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    
    def generate_image(self, prompt: str, num_inference_steps: int = 1, **kwargs) -> Image.Image:
        """
        Generate image from text prompt
        
        Args:
            prompt: Text description
            num_inference_steps: Number of denoising steps (1 for turbo)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated PIL Image
        """
        if self.pipe is None:
            logger.error("Model not loaded. Call load_model() first.")
            return Image.new('RGB', (512, 512), color='black')
        
        try:
            # Set up generation config
            config = GenerationConfig(num_inference_steps=num_inference_steps)
            
            # Override with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Set seed if provided
            generator = None
            if config.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(config.seed)
            
            # Generate image
            logger.info(f"Generating image with prompt: '{prompt[:50]}...'")
            start_time = time.time()
            
            output = self.pipe(
                prompt=prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                height=config.height,
                width=config.width,
                generator=generator,
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Image generated in {generation_time:.2f}s")
            
            return output.images[0]
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "compile" in str(e).lower():
                logger.warning(f"Compilation error detected: {e}")
                logger.warning("Falling back to non-compiled UNet")
                
                # Restore original UNet
                if hasattr(self.pipe.unet, "_orig_mod"):
                    self.pipe.unet = self.pipe.unet._orig_mod
                    
                # Try again without compilation
                try:
                    output = self.pipe(
                        prompt=prompt,
                        num_inference_steps=config.num_inference_steps,
                        guidance_scale=config.guidance_scale,
                        height=config.height,
                        width=config.width,
                        generator=generator,
                    )
                    return output.images[0]
                except Exception as e2:
                    logger.error(f"Error even without compilation: {e2}")
                    return Image.new('RGB', (config.width, config.height), color='red')
            else:
                logger.error(f"Error generating image: {e}")
                return Image.new('RGB', (config.width, config.height), color='red')
    
    def encode_prompts(self, prompts: List[str]) -> Dict[str, List[torch.Tensor]]:
        """Encode multiple prompts to embeddings"""
        if self.pipe is None:
            logger.error("Model not loaded")
            return {}
        
        all_embeddings = {
            'prompt_embeds': [],
            'pooled_prompt_embeds': [],
            'negative_prompt_embeds': [],
            'negative_pooled_prompt_embeds': []
        }
        
        try:
            for prompt in prompts:
                cache_key = f"prompt_{hash(prompt)}"
                
                if cache_key not in self.embedding_cache:
                    # Encode prompt
                    result = self.pipe.encode_prompt(
                        prompt=prompt,
                        prompt_2=prompt,
                        device=self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    
                    # SDXL might return different formats, handle accordingly
                    if isinstance(result, tuple) and len(result) >= 2:
                        prompt_embeds = result[0]
                        negative_prompt_embeds = result[1] if len(result) > 1 else None
                        pooled_prompt_embeds = result[2] if len(result) > 2 else None
                        negative_pooled_prompt_embeds = result[3] if len(result) > 3 else None
                    else:
                        # If format is unexpected, generate embeddings manually
                        prompt_embeds = self.pipe._encode_prompt(
                            prompt=prompt,
                            device=self.device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                        )
                        negative_prompt_embeds = None
                        pooled_prompt_embeds = None
                        negative_pooled_prompt_embeds = None
                    
                    # Cache the embeddings
                    self.embedding_cache[cache_key] = {
                        'prompt_embeds': prompt_embeds,
                        'pooled_prompt_embeds': pooled_prompt_embeds,
                        'negative_prompt_embeds': negative_prompt_embeds,
                        'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds,
                    }
                    
                    # Manage cache size
                    if len(self.embedding_cache) > self.cache_size:
                        oldest_key = list(self.embedding_cache.keys())[0]
                        del self.embedding_cache[oldest_key]
                
                # Add to results - filter out None values
                cached = self.embedding_cache[cache_key]
                for key in all_embeddings.keys():
                    value = cached.get(key)
                    if value is not None:
                        all_embeddings[key].append(value)
            
            # Remove keys with empty lists
            all_embeddings = {k: v for k, v in all_embeddings.items() if v}
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error encoding prompts: {e}")
            return {}
    
    def slerp(self, v0: torch.Tensor, v1: torch.Tensor, t: float, dot_threshold: float = 0.9995) -> torch.Tensor:
        """
        Spherical linear interpolation between two vectors.
        
        Args:
            v0: Start vector
            v1: End vector  
            t: Interpolation parameter (0 to 1)
            dot_threshold: Threshold for using linear interpolation to avoid numerical issues
            
        Returns:
            Interpolated vector
        """
        # Handle different tensor shapes by flattening and reshaping
        original_shape = v0.shape
        v0_flat = v0.flatten()
        v1_flat = v1.flatten()
        
        # Normalize
        v0_norm = v0_flat / v0_flat.norm()
        v1_norm = v1_flat / v1_flat.norm()
        
        # Compute angle between vectors
        dot = torch.dot(v0_norm, v1_norm)
        
        # Clamp to avoid numerical issues
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # If vectors are nearly parallel, use linear interpolation
        if torch.abs(dot) > dot_threshold:
            result = (1 - t) * v0_flat + t * v1_flat
        else:
            # Calculate angle
            theta = torch.acos(dot)
            
            # Perform SLERP
            sin_theta = torch.sin(theta)
            s0 = torch.sin((1 - t) * theta) / sin_theta
            s1 = torch.sin(t * theta) / sin_theta
            
            result = s0 * v0_flat + s1 * v1_flat
        
        return result.reshape(original_shape)
    
    def blend_prompt_embeddings(self, embeddings: Dict[str, List[torch.Tensor]], 
                           weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Blend multiple prompt embeddings with weights using SLERP.
        
        Args:
            embeddings: Dictionary of embedding lists from encode_prompts
            weights: List of weights for each embedding (should sum to 1.0)
            
        Returns:
            Dictionary of blended embeddings
        """
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        blended = {}
        
        try:
            for key, embedding_list in embeddings.items():
                if not embedding_list:  # Skip empty lists
                    continue
                
                # For single embedding, return as is
                if len(embedding_list) == 1:
                    blended[key] = embedding_list[0]
                    continue
                
                # For two embeddings, simple SLERP
                if len(embedding_list) == 2:
                    blended[key] = self.slerp(
                        embedding_list[0], 
                        embedding_list[1], 
                        weights[1]  # t parameter is the weight of the second vector
                    )
                    continue
                
                # For multiple embeddings, use recursive SLERP
                # This performs pairwise SLERP based on weights
                result = embedding_list[0]
                accumulated_weight = weights[0]
                
                for i in range(1, len(embedding_list)):
                    if accumulated_weight + weights[i] > 0:
                        # Calculate interpolation parameter
                        t = weights[i] / (accumulated_weight + weights[i])
                        # SLERP between accumulated result and current embedding
                        result = self.slerp(result, embedding_list[i], t)
                        accumulated_weight += weights[i]
                
                blended[key] = result
            
            return blended
            
        except Exception as e:
            logger.error(f"Error blending embeddings: {e}")
            return {}
    
    def generate_with_embeddings(self, prompt_embeds: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                                **kwargs) -> Image.Image:
        """
        Generate image from pre-computed embeddings
        
        Args:
            prompt_embeds: Pre-computed prompt embeddings (tensor or dict)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated PIL Image
        """
        if self.pipe is None:
            logger.error("Model not loaded")
            return Image.new('RGB', (512, 512), color='black')
        
        try:
            # Handle both tensor and dict inputs
            if isinstance(prompt_embeds, torch.Tensor):
                embeddings = {
                    'prompt_embeds': prompt_embeds,
                    'pooled_prompt_embeds': None,
                }
            else:
                embeddings = prompt_embeds
            
            # Set up generation config
            config = GenerationConfig()
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Generate with embeddings
            output = self.pipe(
                prompt_embeds=embeddings.get('prompt_embeds'),
                pooled_prompt_embeds=embeddings.get('pooled_prompt_embeds'),
                negative_prompt_embeds=embeddings.get('negative_prompt_embeds'),
                negative_pooled_prompt_embeds=embeddings.get('negative_pooled_prompt_embeds'),
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                height=config.height,
                width=config.width,
            )
            
            return output.images[0]
            
        except Exception as e:
            logger.error(f"Error generating from embeddings: {e}")
            return Image.new('RGB', (512, 512), color='blue')

    def generate_with_blended_prompts(self, prompts: List[str], weights: List[float], 
                                     **kwargs) -> Image.Image:
        """
        Generate image from blended prompts
        
        Args:
            prompts: List of prompts to blend
            weights: Weights for each prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated PIL Image
        """
        # Encode all prompts
        embeddings = self.encode_prompts(prompts)
        
        if not embeddings:
            return Image.new('RGB', (512, 512), color='black')
        
        # Blend embeddings
        blended = self.blend_prompt_embeddings(embeddings, weights)
        
        # Generate image
        return self.generate_with_embeddings(blended, **kwargs)
    
    def corrupt_layers(self, layer_indices: List[int], corruption_strength: float) -> None:
        """
        Apply corruption to specific model layers (for testing)
        
        Args:
            layer_indices: List of layer indices to corrupt
            corruption_strength: Strength of corruption (0-1)
        """
        if self.unet is None:
            logger.error("Model not loaded")
            return
        
        try:
            # Store corruption settings
            for idx in layer_indices:
                self.corrupted_layers[idx] = corruption_strength
            
            # Apply corruption to UNet blocks
            blocks = []
            if hasattr(self.unet, 'down_blocks'):
                blocks.extend(self.unet.down_blocks)
            if hasattr(self.unet, 'up_blocks'):
                blocks.extend(self.unet.up_blocks)
            
            for idx, block in enumerate(blocks):
                if idx in self.corrupted_layers:
                    strength = self.corrupted_layers[idx]
                    
                    # Add noise to attention weights
                    for param in block.parameters():
                        if param.requires_grad:
                            noise = torch.randn_like(param) * strength
                            param.data += noise
            
            logger.info(f"Applied corruption to layers: {layer_indices}")
            
        except Exception as e:
            logger.error(f"Error corrupting layers: {e}")
    
    def reset_corruption(self) -> None:
        """Reset all layer corruptions"""
        if self.unet is None:
            return
        
        # Clear corruption dictionary
        self.corrupted_layers.clear()
        
        # Reload model weights (safest way to reset)
        logger.info("Resetting model to remove corruptions...")
        self.load_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.pipe is None:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "use_fp16": self.use_fp16,
            "compiled": self.compile_unet and hasattr(self.unet, "_orig_mod"),
            "cache_size": len(self.embedding_cache),
            "corrupted_layers": list(self.corrupted_layers.keys()),
        }
        
        return info
    
    def clear_cache(self) -> None:
        """Clear embedding cache to free memory"""
        self.embedding_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cache cleared")