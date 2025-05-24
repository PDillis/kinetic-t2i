from abc import ABC, abstractmethod
from PIL import Image
import torch
from typing import Optional, List, Dict, Any

class BaseT2IModel(ABC):
    """Abstract base class for text-to-image models"""
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the text-to-image model"""
        pass
    
    @abstractmethod
    def generate_image(self, prompt: str, **kwargs) -> Image.Image:
        """
        Generate image from text prompt
        
        Args:
            prompt: Text description of desired image
            **kwargs: Additional generation parameters
            
        Returns:
            Generated PIL Image
        """
        pass
    
    @abstractmethod
    def generate_with_embeddings(self, prompt_embeds: torch.Tensor, **kwargs) -> Image.Image:
        """
        Generate image from prompt embeddings
        
        Args:
            prompt_embeds: Pre-computed prompt embeddings
            **kwargs: Additional generation parameters
            
        Returns:
            Generated PIL Image
        """
        pass