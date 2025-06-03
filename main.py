import asyncio
import threading
from typing import Optional
import yaml
import queue
import time
import cv2
import numpy as np

from src.camera.camera_input import CameraInput
from src.hand_tracking.hand_detector import HandDetector
from src.t2i_backend.stable_diffusion_model import StableDiffusionModel
from src.prompt_system.spatial_prompts import SpatialPromptManager
from src.ui.interface import MainInterface
from src.tests.prompt_blend_test import PromptBlendTest

class HandT2IApplication:
    """Main application controller with optimized grid-based generation"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize application
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.running = False
        
        # Initialize components
        self.camera: Optional[CameraInput] = None
        self.hand_detector: Optional[HandDetector] = None
        self.t2i_model: Optional[StableDiffusionModel] = None
        self.prompt_manager: Optional[SpatialPromptManager] = None
        self.ui: Optional[MainInterface] = None
        
        # Test modes
        self.current_test = None
        self.tests = {}
        
        # Performance optimizations
        self.last_grid_position = None  # Track last grid position
        self.generation_queue = queue.Queue(maxsize=1)  # Only queue one generation at a time
        self.generation_thread = None
        self.last_generation_time = 0
        self.min_generation_interval = 0.1  # Minimum 100ms between generations
        
        # Fixed seed for consistent results
        self.fixed_seed = 42
        
        # Shutdown flag
        self.shutdown_called = False
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def initialize_components(self) -> None:
        """Initialize all application components"""
        print("Initializing camera...")
        cam_cfg = self.config["camera"]
        self.camera = CameraInput(cam_cfg["device_id"], cam_cfg["width"], cam_cfg["height"])
        
        print("Initializing hand detector...")
        hand_cfg = self.config["hand_tracking"]
        self.hand_detector = HandDetector(hand_cfg["max_hands"], hand_cfg["detection_confidence"])
        
        print("Setting up prompt zones...")
        width, height = cam_cfg["width"], cam_cfg["height"]
        self.prompt_manager = SpatialPromptManager(width, height)

        for p in self.config["prompts"]["default_prompts"]:
            self.prompt_manager.add_prompt(p["x"], p["y"], p["prompt"], p.get("radius", 100))

        # Check if we need the model by looking at the grid file
        grid_resolution = self.config.get("performance", {}).get("grid_resolution", 32)
        grid_file = f"cache/grid_{width}x{height}_{grid_resolution}.pkl"
        
        need_model = True
        try:
            import pickle
            with open(grid_file, 'rb') as f:
                data = pickle.load(f)
            has_images = data.get('has_pregenerated_images', False)
            if has_images:
                print("Found pre-generated images, skipping model loading")
                need_model = False
        except:
            print("No pre-generated images found, will need model")
        
        # Only load model if needed
        if need_model:
            print("Initializing AI model...")
            model_cfg = self.config["t2i_model"]
            self.t2i_model = StableDiffusionModel(model_cfg["model_name"])
            self.t2i_model.load_model()
        else:
            self.t2i_model = None

        print("Initializing UI...")
        ui_cfg = self.config["ui"]
        self.ui = MainInterface(ui_cfg["window_width"], ui_cfg["window_height"])
        self.ui.set_prompt_zones(self.prompt_manager.prompt_zones)
        
        # Connect UI callbacks to actual functionality
        self.ui.set_display_option_callback(self._on_display_option_changed)
    
    def setup_tests(self) -> None:
        """Setup available test modes with precomputed grid"""
        print("Setting up test mode with precomputed grid...")
        # Check if we should pre-generate images
        pregenerate = self.config.get("performance", {}).get("pregenerate_images", False)
    
        self.tests["prompt_blend"] = PromptBlendTest(
            prompt_manager=self.prompt_manager,
            t2i_model=self.t2i_model,
            use_precomputed=True,  # Enable precomputed grid
            pregenerate_images=pregenerate,
            grid_resolution=self.config.get("performance", {}).get("grid_resolution", 32))    # 32x32 grid by default
        self.current_test = self.tests["prompt_blend"]
        print("Test mode ready!")
    
    def get_grid_position(self, hand_pos):
        """Get the grid position for a hand position"""
        if not hasattr(self.current_test, 'grid'):
            return None
        
        grid = self.current_test.grid
        grid_x = min(max(hand_pos[0] // grid.x_step, 0), grid.grid_resolution - 1)
        grid_y = min(max(hand_pos[1] // grid.y_step, 0), grid.grid_resolution - 1)
        
        return (grid_x, grid_y)
    
    def generation_worker(self):
        """Background worker for image generation"""
        print("Generation worker started")
        
        while self.running:
            try:
                # Get generation request from queue (blocking with timeout)
                generation_data = self.generation_queue.get(timeout=1.0)
                
                if generation_data is None:  # Shutdown signal
                    break
                
                hand_data, grid_pos = generation_data
                
                # Check minimum time interval
                current_time = time.time()
                if current_time - self.last_generation_time < self.min_generation_interval:
                    continue
                
                # Add fixed seed to hand_data for consistent generation
                hand_data['seed'] = self.fixed_seed
                
                # Generate image with fixed seed
                start_time = time.time()
                image = self.current_test.generate_image(hand_data)
                generation_time = time.time() - start_time
                
                # print(f"Generated image for grid position {grid_pos} in {generation_time:.3f}s")
                
                # Update UI
                info = hand_data.get("info", "")
                grid_info = f"Grid: {grid_pos} | {info}"
                self.ui.update_generated_image(image, grid_info)
                
                self.last_generation_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Generation error: {e}")
                continue
        
        print("Generation worker stopped")
    
    def main_loop(self) -> None:
        """Optimized main application processing loop"""
        # Get performance settings
        display_skip = self.config.get("performance", {}).get("frame_skip", 2)
        hand_skip = self.config.get("performance", {}).get("hand_skip", 1)
        
        frame_counter = 0
        hand_counter = 0
        
        print("Starting main loop...")
        
        # Keep track of last detected skeleton for persistent display
        last_skeleton = []
        
        while self.running:
            # Get camera frame in BGR format for OpenCV processing
            frame_bgr = self.camera.get_frame_bgr()
            if frame_bgr is None:
                continue
            
            # Process hand detection less frequently
            hand_result = None
            current_grid_pos = None
            
            if frame_counter % display_skip == 0:
                # Convert BGR to RGB for MediaPipe hand detection
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                hand_result = self.hand_detector.detect_hands(frame_rgb)
                
                # Update skeleton if hand detected
                if hand_result and hand_result["detected"]:
                    last_skeleton = hand_result.get("skeleton", [])
                    
                    # Check if we should process for generation
                    if self.current_test and hand_counter % hand_skip == 0:
                        # Use index finger tip instead of hand center
                        if len(last_skeleton) >= 21:  # Full hand detected
                            index_tip = last_skeleton[8]  # Index 8 is the index finger tip
                            current_grid_pos = self.get_grid_position(index_tip)
                            
                            # Only generate if we've moved to a new grid position
                            if (current_grid_pos is not None and 
                                current_grid_pos != self.last_grid_position):
                                
                                # Get precomputed data for this position
                                hand_data = self.current_test.process_hand_position(index_tip)
                                hand_data['hand_pos'] = index_tip
                                
                                # Queue generation (non-blocking)
                                try:
                                    # Clear any pending generation and add new one
                                    try:
                                        self.generation_queue.get_nowait()  # Remove old request
                                    except queue.Empty:
                                        pass
                                    
                                    self.generation_queue.put_nowait((hand_data, current_grid_pos))
                                    self.last_grid_position = current_grid_pos
                                    
                                except queue.Full:
                                    # Queue full, skip this generation
                                    pass
                
                hand_counter += 1
            
            # Create display overlay based on UI settings
            overlay = None
            
            # Show camera background if enabled
            try:
                if self.ui and self.ui.show_camera.get():
                    overlay = frame_bgr.copy()
                else:
                    # Create black background if camera disabled
                    overlay = np.zeros_like(frame_bgr)
            except:
                # Fallback if UI not ready
                overlay = frame_bgr.copy()
            
            # Always draw skeleton if we have one and skeleton is enabled
            if (last_skeleton and self.ui and 
                self.ui.show_skeleton.get()):
                overlay = self.hand_detector.draw_skeleton(
                    overlay, 
                    last_skeleton, 
                    draw_connections=self.ui.show_connections.get(),
                    draw_landmarks=self.ui.show_landmarks.get()
                )
            
            # Draw index fingertip if enabled and we have skeleton
            if (last_skeleton and self.ui and self.ui.show_hand_center.get()):
                # Use index finger tip instead of hand center
                if len(last_skeleton) >= 21:  # Full hand
                    index_tip_x, index_tip_y = last_skeleton[8]  # Index finger tip
                    
                    # Draw center point (large, visible)
                    cv2.circle(overlay, (index_tip_x, index_tip_y), 8, (0, 255, 255), -1)  # Yellow center
                    cv2.circle(overlay, (index_tip_x, index_tip_y), 10, (0, 0, 0), 2)      # Black outline
                    
                    # Draw crosshair
                    cv2.line(overlay, (index_tip_x-15, index_tip_y), (index_tip_x+15, index_tip_y), (0, 255, 255), 2)
                    cv2.line(overlay, (index_tip_x, index_tip_y-15), (index_tip_x, index_tip_y+15), (0, 255, 255), 2)
            
            # Add prompt zone overlays - always show centers, full zones if enabled
            show_full_zones = self.ui and self.ui.show_prompt_zones.get()
            show_text = self.ui and self.ui.show_prompt_text.get()
            overlay = self.prompt_manager.visualize_prompt_zones(overlay, show_full_zones, show_text)
            
            # Update camera display
            if self.ui:
                self.ui.update_camera_display(overlay)
            
            frame_counter += 1
    
    def run(self) -> None:
        """Start the optimized application"""
        self.running = True
        self.initialize_components()
        self.setup_tests()
        
        # Start background generation worker
        self.generation_thread = threading.Thread(target=self.generation_worker, daemon=True)
        self.generation_thread.start()
        
        # Start main processing thread
        processing_thread = threading.Thread(target=self.main_loop, daemon=True)
        processing_thread.start()
        
        # Run UI on main thread
        try:
            print("Starting UI...")
            self.ui.run()  # This blocks until UI is closed
        except KeyboardInterrupt:
            print("Keyboard interrupt received")
        except Exception as e:
            print(f"UI error: {e}")
        finally:
            # UI has been closed, now shutdown cleanly
            self.shutdown()
    
    def shutdown(self) -> None:
        """Gracefully shutdown application"""
        if self.shutdown_called:
            return  # Already shutting down
        
        self.shutdown_called = True
        print("Shutting down application...")
        self.running = False
        
        # Signal generation worker to stop
        try:
            self.generation_queue.put_nowait(None)
        except queue.Full:
            pass
        
        # Wait for generation thread
        if self.generation_thread and self.generation_thread.is_alive():
            print("Waiting for generation thread...")
            self.generation_thread.join(timeout=2.0)
            if self.generation_thread.is_alive():
                print("Warning: Generation thread did not stop cleanly")
        
        # Release resources (but not UI, it's already destroyed)
        try:
            if self.camera:
                self.camera.release()
                print("Camera released")
        except Exception as e:
            print(f"Error releasing camera: {e}")
        
        try:
            if self.hand_detector:
                self.hand_detector.release()
                print("Hand detector released")
        except Exception as e:
            print(f"Error releasing hand detector: {e}")
        
        # Don't try to destroy UI here - it's already been destroyed by the main thread
        print("Shutdown complete")
    
    def _on_display_option_changed(self):
        """Handle display option changes from UI"""
        # This gets called when checkboxes are toggled
        # No specific action needed as the main loop checks UI state each frame
        pass

def main():
    """Application entry point"""
    app = HandT2IApplication()
    try:
        app.run()
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        # Catch any remaining errors (including Tkinter errors) and show them nicely
        import traceback
        if "_tkinter.TclError" in str(e) and "destroy" in str(e):
            print("Application closed cleanly")
        else:
            print(f"Application error: {e}")
            traceback.print_exc()
    finally:
        # Ensure shutdown is called
        try:
            app.shutdown()
        except Exception as e:
            if "_tkinter.TclError" not in str(e):
                print(f"Shutdown error: {e}")

if __name__ == "__main__":
    main()