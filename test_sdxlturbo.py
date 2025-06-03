"""
Test script for SDXL-Turbo model
Tests basic generation, prompt blending, and performance
"""

import sys
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.t2i_backend.stable_diffusion_model import StableDiffusionModel

def test_basic_generation():
    """Test basic image generation"""
    print("Test 1: Basic Image Generation with SDXL-Turbo")
    print("-" * 50)
    
    # Initialize model
    model = StableDiffusionModel(
        model_name="stabilityai/sdxl-turbo",
        use_fp16=True,
        compile_unet=False  # First run will be slow due to compilation
    )
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    model.load_model()
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    
    # Test prompts
    test_prompts = [
        "a serene mountain landscape at sunset",
        "abstract colorful geometric patterns",
        "a futuristic city with flying cars",
        "underwater coral reef with tropical fish",
    ]
    
    # Generate images
    images = []
    generation_times = []
    
    # First generation will be slow if UNet is compiled
    print("\nFirst generation (may be slow due to compilation)...")
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nGenerating image {i+1}/{len(test_prompts)}: '{prompt}'")
        start_time = time.time()
        
        image = model.generate_image(
            prompt,
            num_inference_steps=1,  # SDXL-Turbo single step
            height=512,
            width=512,
            seed=42  # Fixed seed for reproducibility
        )
        
        generation_time = time.time() - start_time
        generation_times.append(generation_time)
        images.append(image)
        
        print(f"Generated in {generation_time:.2f}s")
        
        # Save individual image
        image.save(f"test_sdxl_output_{i+1}.png")
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, (image, prompt) in enumerate(zip(images, test_prompts)):
        axes[i].imshow(image)
        axes[i].set_title(f"{prompt[:30]}...\n({generation_times[i]:.2f}s)")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_sdxl_basic_generation.png")
    plt.show()
    
    print(f"\nAverage generation time: {np.mean(generation_times):.2f}s")
    print(f"Average after first (compiled): {np.mean(generation_times[1:]):.2f}s")
    print("Basic generation test completed!")
    
    return model

def test_speed_comparison(model: StableDiffusionModel):
    """Test generation speed over multiple runs"""
    print("\n\nTest: Speed Comparison")
    print("-" * 50)
    
    prompt = "a beautiful landscape with mountains and lakes"
    num_runs = 10
    
    times = []
    for i in range(num_runs):
        start_time = time.time()
        image = model.generate_image(prompt, seed=42)
        generation_time = time.time() - start_time
        times.append(generation_time)
        print(f"Run {i+1}: {generation_time:.3f}s")
    
    print(f"\nMin time: {min(times):.3f}s")
    print(f"Max time: {max(times):.3f}s")
    print(f"Average time: {np.mean(times):.3f}s")
    print(f"Std dev: {np.std(times):.3f}s")

def test_prompt_blending(model: StableDiffusionModel):
    """Test prompt blending functionality"""
    print("\n\nTest 2: Prompt Blending")
    print("-" * 50)
    
    # Define prompts to blend
    prompts = [
        "a peaceful forest with tall trees",
        "a vibrant sunset with orange and pink colors",
        "abstract digital art with geometric shapes"
    ]
    
    # Test different weight combinations
    weight_combinations = [
        [1.0, 0.0, 0.0],  # Only first prompt
        [0.0, 1.0, 0.0],  # Only second prompt
        [0.0, 0.0, 1.0],  # Only third prompt
        [0.5, 0.5, 0.0],  # Mix of first two
        [0.33, 0.33, 0.34],  # Equal mix of all three
        [0.7, 0.2, 0.1],  # Mostly first prompt
    ]
    
    images = []
    
    for i, weights in enumerate(weight_combinations):
        print(f"\nBlending with weights: {weights}")
        
        image = model.generate_with_blended_prompts(
            prompts=prompts,
            weights=weights,
            num_inference_steps=1,
            height=512,
            width=512,
            seed=42
        )
        
        images.append(image)
        image.save(f"test_blend_{i+1}.png")
    
    # Display blending results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (image, weights) in enumerate(zip(images, weight_combinations)):
        axes[i].imshow(image)
        axes[i].set_title(f"Weights: {weights}")
        axes[i].axis('off')
    
    plt.suptitle("Prompt Blending Results")
    plt.tight_layout()
    plt.savefig("test_prompt_blending.png")
    plt.show()
    
    print("Prompt blending test completed!")

def test_embedding_cache(model: StableDiffusionModel):
    """Test embedding cache performance"""
    print("\n\nTest 3: Embedding Cache Performance")
    print("-" * 50)
    
    prompt = "a beautiful landscape"
    
    # First generation (no cache)
    print("First generation (no cache)...")
    start_time = time.time()
    image1 = model.generate_image(prompt, seed=42)
    time1 = time.time() - start_time
    
    # Second generation (with cache)
    print("Second generation (with cache)...")
    start_time = time.time()
    image2 = model.generate_image(prompt, seed=42)
    time2 = time.time() - start_time
    
    print(f"\nFirst generation: {time1:.2f}s")
    print(f"Second generation: {time2:.2f}s")
    print(f"Speed improvement: {(time1 - time2) / time1 * 100:.1f}%")
    
    # Check cache
    model_info = model.get_model_info()
    print(f"Cache size: {model_info['cache_size']}")

def test_different_resolutions(model: StableDiffusionModel):
    """Test generation at different resolutions"""
    print("\n\nTest 4: Different Resolutions")
    print("-" * 50)
    
    prompt = "a majestic eagle soaring through clouds"
    resolutions = [
        (256, 256),
        (512, 512),
        (768, 768),
        (512, 768),  # Portrait
        (768, 512),  # Landscape
    ]
    
    images = []
    times = []
    
    for width, height in resolutions:
        print(f"\nGenerating at {width}x{height}...")
        start_time = time.time()
        
        try:
            image = model.generate_image(
                prompt,
                width=width,
                height=height,
                num_inference_steps=1,
                seed=42
            )
            generation_time = time.time() - start_time
            
            images.append(image)
            times.append(generation_time)
            
            print(f"Generated in {generation_time:.2f}s")
            
        except Exception as e:
            print(f"Failed to generate at {width}x{height}: {e}")
            images.append(Image.new('RGB', (width, height), color='red'))
            times.append(0)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, ((width, height), image, gen_time) in enumerate(zip(resolutions, images, times)):
        if i < len(axes):
            axes[i].imshow(image)
            axes[i].set_title(f"{width}x{height}\n({gen_time:.2f}s)")
            axes[i].axis('off')
    
    # Hide extra subplot
    if len(resolutions) < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle("Different Resolution Tests")
    plt.tight_layout()
    plt.savefig("test_resolutions.png")
    plt.show()

def test_layer_corruption(model: StableDiffusionModel):
    """Test layer corruption functionality"""
    print("\n\nTest 5: Layer Corruption (for testing purposes)")
    print("-" * 50)
    
    prompt = "a crystal clear mountain lake"
    
    # Generate normal image
    print("Generating normal image...")
    normal_image = model.generate_image(prompt, seed=42)
    
    # Test different corruption levels
    corruption_tests = [
        ([0, 1], 0.1),  # Light corruption on first layers
        ([5, 6, 7], 0.3),  # Medium corruption on middle layers
        ([10, 11, 12], 0.5),  # Strong corruption on later layers
    ]
    
    corrupted_images = []
    
    for layers, strength in corruption_tests:
        print(f"\nCorrupting layers {layers} with strength {strength}")
        
        # Apply corruption
        model.corrupt_layers(layers, strength)
        
        # Generate corrupted image
        corrupted_image = model.generate_image(prompt, seed=42)
        corrupted_images.append(corrupted_image)
        
        # Reset corruption
        model.reset_corruption()
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    axes[0].imshow(normal_image)
    axes[0].set_title("Normal")
    axes[0].axis('off')
    
    for i, ((layers, strength), image) in enumerate(zip(corruption_tests, corrupted_images)):
        axes[i+1].imshow(image)
        axes[i+1].set_title(f"Layers {layers}, Strength {strength}")
        axes[i+1].axis('off')
    
    plt.suptitle("Layer Corruption Effects")
    plt.tight_layout()
    plt.savefig("test_layer_corruption.png")
    plt.show()

def main():
    """Run all tests"""
    print("SDXL-Turbo Model Test Suite")
    print("=" * 50)
    
    # Run tests
    model = test_basic_generation()
    test_speed_comparison(model)
    
    # Test prompt blending (reuse from previous implementation)
    test_prompt_blending(model)
    # test_embedding_cache(model)
    # test_different_resolutions(model)
    # test_layer_corruption(model)
    print("\nAll tests completed!")
    
    # Clean up
    print("\n\nCleaning up...")
    model.clear_cache()
    
if __name__ == "__main__":
    main()