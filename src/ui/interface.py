import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import cv2
from typing import Optional, Callable, Dict, Any
import threading
import queue
import logging
from src.ui.color_utils import get_prompt_color_hex

# Configure CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

logger = logging.getLogger(__name__)

class MainInterface:
    """Main application GUI with split-screen layout"""
    
    def __init__(self, window_width: int = 1280, window_height: int = 720):
        """Initialize the user interface"""
        self.window_width = window_width
        self.window_height = window_height
        
        # Initialize main window
        self.root = ctk.CTk()
        self.root.title("Hand-Controlled Text-to-Image")
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.resizable(False, False)
        
        # State variables - update these to match hand test
        self.show_camera = tk.BooleanVar(value=True)
        self.show_skeleton = tk.BooleanVar(value=True)
        self.show_connections = tk.BooleanVar(value=True)
        self.show_landmarks = tk.BooleanVar(value=True)
        self.show_hand_center = tk.BooleanVar(value=True)
        self.show_prompt_zones = tk.BooleanVar(value=True)
        self.show_fps = tk.BooleanVar(value=True)
        self.show_prompt_text = tk.BooleanVar(value=False)
        self.current_test_mode = tk.StringVar(value="prompt_blend")
        
        # Camera controls
        self.brightness_var = tk.DoubleVar(value=0)
        self.contrast_var = tk.DoubleVar(value=0)
        self.blend_alpha_var = tk.DoubleVar(value=1.0)  # 1.0 = full overlay, 0.0 = camera only
        self.mediapipe_blend_var = tk.DoubleVar(value=0.0)  # 0.0 = custom only, 1.0 = mediapipe only
        
        # Display dimensions
        self.camera_display_width = 640
        self.camera_display_height = 480
        self.generated_display_width = 512
        self.generated_display_height = 512
        
        # Image placeholders
        self.camera_image_label: Optional[tk.Label] = None
        self.generated_image_label: Optional[tk.Label] = None
        
        # Callbacks
        self.test_mode_callback: Optional[Callable] = None
        self.brightness_callback: Optional[Callable] = None
        self.contrast_callback: Optional[Callable] = None
        
        # Thread-safe queues for image updates
        self.camera_queue = queue.Queue(maxsize=2)
        self.generated_queue = queue.Queue(maxsize=2)
        
        # FPS tracking
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = cv2.getTickCount()
        
        # Current images (for blending)
        self.current_camera_frame: Optional[np.ndarray] = None
        self.current_skeleton_overlay: Optional[np.ndarray] = None
        self.current_mediapipe_overlay: Optional[np.ndarray] = None
        
        self._setup_layout()
        self._start_update_thread()
    
    def _setup_layout(self) -> None:
        """Create UI layout with split screen and controls"""
        # Main container
        main_container = ctk.CTkFrame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side - Camera view and controls
        left_frame = ctk.CTkFrame(main_container)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Camera display
        camera_frame = ctk.CTkFrame(left_frame)
        camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.camera_image_label = ctk.CTkLabel(
            camera_frame, 
            text="Camera Feed",
            width=self.camera_display_width,
            height=self.camera_display_height
        )
        self.camera_image_label.pack()
        
        # Control panel
        control_frame = ctk.CTkFrame(left_frame)
        control_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Display controls
        display_controls = ctk.CTkFrame(control_frame)
        display_controls.pack(fill="x", pady=5)
        
        ctk.CTkLabel(display_controls, text="Display Options:", font=("Arial", 14, "bold")).pack(anchor="w")
        
        # Checkboxes in a grid
        checkbox_frame = ctk.CTkFrame(display_controls)
        checkbox_frame.pack(fill="x", pady=5)
        
        ctk.CTkCheckBox(
            checkbox_frame, 
            text="Show Camera", 
            variable=self.show_camera,
            command=self._on_display_option_changed
        ).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        
        ctk.CTkCheckBox(
            checkbox_frame, 
            text="Show Skeleton", 
            variable=self.show_skeleton,
            command=self._on_display_option_changed
        ).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        # ctk.CTkCheckBox(
        #     checkbox_frame, 
        #     text="Show Connections", 
        #     variable=self.show_connections,
        #     command=self._on_display_option_changed
        # ).grid(row=0, column=2, sticky="w", padx=5, pady=2)
        
        # ctk.CTkCheckBox(
        #     checkbox_frame, 
        #     text="Show Landmarks", 
        #     variable=self.show_landmarks,
        #     command=self._on_display_option_changed
        # ).grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        ctk.CTkCheckBox(
            checkbox_frame, 
            text="Show Index Tip", 
            variable=self.show_hand_center,  # Keep same variable
            command=self._on_display_option_changed
        ).grid(row=0, column=2, sticky="w", padx=5, pady=2)
        
        ctk.CTkCheckBox(
            checkbox_frame, 
            text="Show FPS", 
            variable=self.show_fps,
            command=self._on_display_option_changed
        ).grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        ctk.CTkCheckBox(
            checkbox_frame, 
            text="Show Prompts", 
            variable=self.show_prompt_text,
            command=self._on_display_option_changed
        ).grid(row=0, column=4, sticky="w", padx=5, pady=2)

        # Camera controls
        camera_controls = ctk.CTkFrame(control_frame)
        camera_controls.pack(fill="x", pady=5)
        
        ctk.CTkLabel(camera_controls, text="Camera Controls:", font=("Arial", 14, "bold")).pack(anchor="w")
        
        # Brightness slider
        brightness_frame = ctk.CTkFrame(camera_controls)
        brightness_frame.pack(fill="x", pady=2)
        
        ctk.CTkLabel(brightness_frame, text="Brightness:").pack(side="left", padx=5)
        self.brightness_slider = ctk.CTkSlider(
            brightness_frame,
            from_=-100,
            to=100,
            variable=self.brightness_var,
            command=self._on_brightness_changed
        )
        self.brightness_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.brightness_label = ctk.CTkLabel(brightness_frame, text="0")
        self.brightness_label.pack(side="left", padx=5)
        
        # Contrast slider
        contrast_frame = ctk.CTkFrame(camera_controls)
        contrast_frame.pack(fill="x", pady=2)
        
        ctk.CTkLabel(contrast_frame, text="Contrast:").pack(side="left", padx=5)
        self.contrast_slider = ctk.CTkSlider(
            contrast_frame,
            from_=-100,
            to=100,
            variable=self.contrast_var,
            command=self._on_contrast_changed
        )
        self.contrast_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.contrast_label = ctk.CTkLabel(contrast_frame, text="0")
        self.contrast_label.pack(side="left", padx=5)
        
        # Blend alpha slider
        # blend_frame = ctk.CTkFrame(camera_controls)
        # blend_frame.pack(fill="x", pady=2)

        # MediaPipe blend slider
        # mediapipe_frame = ctk.CTkFrame(camera_controls)
        # mediapipe_frame.pack(fill="x", pady=2)

        # ctk.CTkLabel(mediapipe_frame, text="MediaPipe Blend:").pack(side="left", padx=5)
        # self.mediapipe_slider = ctk.CTkSlider(
        #     mediapipe_frame,
        #     from_=0.0,
        #     to=1.0,
        #     variable=self.mediapipe_blend_var,
        #     command=self._on_mediapipe_blend_changed
        # )
        # self.mediapipe_slider.pack(side="left", fill="x", expand=True, padx=5)
        # self.mediapipe_label = ctk.CTkLabel(mediapipe_frame, text="0.0")
        # self.mediapipe_label.pack(side="left", padx=5)
        
        # Test mode selector
        # test_mode_frame = ctk.CTkFrame(control_frame)
        # test_mode_frame.pack(fill="x", pady=5)
        
        # ctk.CTkLabel(test_mode_frame, text="Test Mode:", font=("Arial", 14, "bold")).pack(anchor="w")
        
        # self.test_mode_menu = ctk.CTkOptionMenu(
        #     test_mode_frame,
        #     values=["prompt_blend", "layer_corruption"],
        #     variable=self.current_test_mode,
        #     command=self._on_test_mode_changed
        # )
        # self.test_mode_menu.pack(fill="x", pady=5)
        
        # Right side - Generated image
        right_frame = ctk.CTkFrame(main_container)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        # Generated image display
        generated_frame = ctk.CTkFrame(right_frame)
        generated_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(generated_frame, text="Generated Image", 
                    font=("Arial", 16, "bold")).pack(pady=5)
        
        self.generated_image_label = ctk.CTkLabel(
            generated_frame,
            text="",  # Start with empty text
            width=self.generated_display_width,
            height=self.generated_display_height
        )
        self.generated_image_label.pack()

        # Add state tracking for first image
        self.first_image_generated = False
        
        # Generation info
        self.generation_info_label = ctk.CTkLabel(
            generated_frame,
            text="",
            font=("Arial", 10)
        )
        self.generation_info_label.pack(pady=5)
        
        # Prompt weights bar graph
        weights_frame = ctk.CTkFrame(generated_frame)
        weights_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(weights_frame, text="Prompt Weights", font=("Arial", 12, "bold")).pack(pady=2)

        self.weights_canvas = tk.Canvas(weights_frame, height=60, bg='#212121', highlightthickness=0)
        self.weights_canvas.pack(fill="x", padx=5, pady=5)

        # Status bar
        self.status_bar = ctk.CTkLabel(
            self.root,
            text="Ready",
            anchor="w"
        )
        self.status_bar.pack(side="bottom", fill="x", padx=10, pady=5)
    
    def _start_update_thread(self) -> None:
        """Start thread for updating UI displays"""
        def update_displays():
            while True:
                try:
                    # Update camera display
                    if not self.camera_queue.empty():
                        image = self.camera_queue.get_nowait()
                        # Use thread-safe update
                        self.root.after_idle(lambda img=image: self._update_camera_label(img))
                    
                    # Update generated display
                    if not self.generated_queue.empty():
                        image = self.generated_queue.get_nowait()
                        # Use thread-safe update
                        self.root.after_idle(lambda img=image: self._update_generated_label(img))
                    
                except:
                    pass
                
                import time
                time.sleep(0.01)  # Small delay instead of after()
        
        thread = threading.Thread(target=update_displays, daemon=True)
        thread.start()

    def _update_camera_label(self, image):
        """Thread-safe camera label update"""
        try:
            self.camera_image_label.configure(image=image)
            self.camera_image_label.image = image
        except:
            pass

    def _update_generated_label(self, image):
        """Thread-safe generated label update"""
        try:
            self.generated_image_label.configure(image=image)
            self.generated_image_label.image = image
        except:
            pass
    
    def update_camera_display(self, frame: np.ndarray, skeleton_overlay: Optional[np.ndarray] = None, 
                            mediapipe_overlay: Optional[np.ndarray] = None) -> None:
        """
        Update camera feed display with optional skeleton overlay
        
        Args:
            frame: RGB image as numpy array (camera feed)
            skeleton_overlay: Optional custom skeleton overlay to blend
            mediapipe_overlay: Optional MediaPipe visualization to blend
        """
        # Store current frames
        self.current_camera_frame = frame.copy()
        if skeleton_overlay is not None:
            self.current_skeleton_overlay = skeleton_overlay.copy()
        if mediapipe_overlay is not None:
            self.current_mediapipe_overlay = mediapipe_overlay.copy()
        
        # Apply brightness and contrast
        adjusted_frame = self._apply_camera_adjustments(frame)
        
        # Start with adjusted camera frame or black if camera disabled
        if self.show_camera.get():
            display_frame = adjusted_frame.copy()
        else:
            display_frame = np.zeros_like(adjusted_frame)
        
        # Blend overlays if skeleton is enabled
        if self.show_skeleton.get():
            final_overlay = None
            
            # Determine which overlay to use
            if skeleton_overlay is not None and mediapipe_overlay is not None:
                # Blend custom skeleton with MediaPipe visualization
                final_overlay = self.blend_images(
                    skeleton_overlay, 
                    mediapipe_overlay, 
                    self.mediapipe_blend_var.get()
                )
            elif skeleton_overlay is not None:
                # Only custom skeleton available
                final_overlay = skeleton_overlay
            elif mediapipe_overlay is not None:
                # Only MediaPipe available
                final_overlay = mediapipe_overlay
            
            # Blend the final overlay with the display frame
            if final_overlay is not None:
                if self.show_camera.get():
                    display_frame = self.blend_images(display_frame, final_overlay, self.blend_alpha_var.get())
                else:
                    # If camera is off, just show the skeleton on black background
                    display_frame = final_overlay.copy()
        
        # Add FPS if enabled
        if self.show_fps.get():
            self._update_fps()
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert to PIL and resize
        image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        image = image.resize((self.camera_display_width, self.camera_display_height), Image.LANCZOS)
        
        # Convert to CTkImage
        photo = ctk.CTkImage(light_image=image, dark_image=image, size=(self.camera_display_width, self.camera_display_height))
        
        # Queue update
        try:
            self.camera_queue.put_nowait(photo)
        except queue.Full:
            pass
    
    def update_prompt_weights(self, weights_info: str) -> None:
        self.weights_canvas.delete("all")
        
        # Parse "prompt1:0.6 + prompt2:0.4" format
        weights_data = []
        if weights_info and ":" in weights_info:
            for part in weights_info.split(" + "):
                if ":" in part:
                    try:
                        name, weight = part.split(":", 1)
                        weights_data.append((name.strip("()"), float(weight)))
                    except ValueError:
                        continue
        
        if not weights_data:
            return
        
        canvas_width = self.weights_canvas.winfo_width()
        if canvas_width <= 1:
            return
        
        # Draw horizontal bar segments
        bar_width = canvas_width - 20
        current_x = 10
        
        for i, (name, weight) in enumerate(weights_data):
            # Normalize to full bar width with epsilon to avoid division by zero
            total_weight = sum(weight for _, weight in weights_data)
            segment_width = bar_width * (weight / (total_weight + 1e-6))
            
            # Find the actual prompt zone index for this prompt name
            color_index = 0  # fallback
            for j, zone in enumerate(getattr(self, 'prompt_zones', [])):
                if name.strip() in zone.prompt or zone.prompt in name.strip():
                    color_index = j
                    break

            color = get_prompt_color_hex(color_index)
            
            if segment_width > 2:
                self.weights_canvas.create_rectangle(
                    current_x, 10, current_x + segment_width, 50,
                    fill=color, outline="#FFFFFF", width=1
                )
                current_x += segment_width

    def update_generated_image(self, image: Image.Image, info: str = "") -> None:
        """Update generated image display"""
        # On first image, clear the placeholder text
        if not self.first_image_generated:
            self.generated_image_label.configure(text="")
            self.first_image_generated = True
        
        # Resize if needed
        if image.size != (self.generated_display_width, self.generated_display_height):
            image = image.resize((self.generated_display_width, self.generated_display_height), 
                            Image.LANCZOS)
        
        # Convert to PhotoImage
        photo = ctk.CTkImage(light_image=image, dark_image=image, size=(self.generated_display_width, self.generated_display_height))
        
        # Queue update
        try:
            self.generated_queue.put_nowait(photo)
        except queue.Full:
            pass
        
        # Update info label and weights
        if info:
            if " | " in info:
                grid_info, weights_info = info.split(" | ", 1)
                self.generation_info_label.configure(text=grid_info)
                self.update_prompt_weights(weights_info)
            else:
                self.generation_info_label.configure(text=info)
    
    def blend_images(self, img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
        """
        Blend two images together
        
        Args:
            img1: First image (camera)
            img2: Second image (overlay)
            alpha: Blend factor (0=img1 only, 1=img2 only)
            
        Returns:
            Blended image
        """
        return cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
    
    def _apply_camera_adjustments(self, frame: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast adjustments to frame"""
        adjusted = frame.copy()
        
        # Apply brightness (-100 to 100)
        brightness = self.brightness_var.get()
        if brightness != 0:
            alpha = (brightness + 100) / 100  # Convert to multiplier
            adjusted = cv2.convertScaleAbs(adjusted, alpha=alpha, beta=0)
        
        # Apply contrast (-100 to 100)
        contrast = self.contrast_var.get()
        if contrast != 0:
            adjusted = cv2.convertScaleAbs(adjusted, alpha=1, beta=contrast)
        
        return adjusted
    
    def _update_fps(self) -> None:
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = cv2.getTickCount()
            self.fps = 30 / ((current_time - self.last_fps_time) / cv2.getTickFrequency())
            self.last_fps_time = current_time
    
    def _on_display_option_changed(self) -> None:
        """Handle display option checkbox changes"""
        # Re-render current frame with new settings
        if self.current_camera_frame is not None:
            self.update_camera_display(self.current_camera_frame, self.current_skeleton_overlay)
        
    def _on_brightness_changed(self, value: float) -> None:
        """Handle brightness slider change"""
        self.brightness_label.configure(text=f"{int(value)}")
        # Apply brightness change immediately to current frame
        if self.current_camera_frame is not None:
            self.update_camera_display(self.current_camera_frame, self.current_skeleton_overlay)

    def _on_contrast_changed(self, value: float) -> None:
        """Handle contrast slider change"""
        self.contrast_label.configure(text=f"{int(value)}")
        # Apply contrast change immediately to current frame
        if self.current_camera_frame is not None:
            self.update_camera_display(self.current_camera_frame, self.current_skeleton_overlay)
    
    def _on_blend_changed(self, value: float) -> None:
        """Handle blend slider change"""
        self.blend_label.configure(text=f"{value:.2f}")
        # Re-render current frame with new blend
        if self.current_camera_frame is not None:
            self.update_camera_display(self.current_camera_frame, self.current_skeleton_overlay)
    
    def _on_mediapipe_blend_changed(self, value: float) -> None:
        """Handle MediaPipe blend slider change"""
        self.mediapipe_label.configure(text=f"{value:.2f}")
        # Re-render current frame with new blend
        if self.current_camera_frame is not None:
            self.update_camera_display(self.current_camera_frame, self.current_skeleton_overlay)

    def _on_test_mode_changed(self, mode: str) -> None:
        """Handle test mode change"""
        if self.test_mode_callback:
            self.test_mode_callback(mode)
    
    def set_prompt_zones(self, prompt_zones):
        self.prompt_zones = prompt_zones

    def set_test_mode_callback(self, callback: Callable) -> None:
        """Set callback for test mode changes"""
        self.test_mode_callback = callback
    
    def set_brightness_callback(self, callback: Callable) -> None:
        """Set callback for brightness changes"""
        self.brightness_callback = callback
    
    def set_contrast_callback(self, callback: Callable) -> None:
        """Set callback for contrast changes"""
        self.contrast_callback = callback
    
    def update_status(self, message: str) -> None:
        """Update status bar message"""
        self.status_bar.configure(text=message)
    
    def get_display_settings(self) -> Dict[str, bool]:
        """Get current display settings"""
        return {
            'show_camera': self.show_camera.get(),
            'show_skeleton': self.show_skeleton.get(),
            'show_connections': self.show_connections.get(),
            'show_landmarks': self.show_landmarks.get(),
            'show_hand_center': self.show_hand_center.get(),
            'show_prompt_zones': self.show_prompt_zones.get(),
            'show_fps': self.show_fps.get(),
            'show_prompt_text': self.show_prompt_text.get(),
        }
    
    def set_display_option_callback(self, callback):
        """Set callback for display option changes"""
        self.display_option_callback = callback

    def _on_display_option_changed(self) -> None:
        """Handle display option checkbox changes"""
        # Call the callback if it exists
        if hasattr(self, 'display_option_callback') and self.display_option_callback:
            self.display_option_callback()
        
        # Re-render current frame with new settings if we have one
        if self.current_camera_frame is not None:
            self.update_camera_display(self.current_camera_frame, self.current_skeleton_overlay)

    def run(self) -> None:
        """Start the GUI event loop"""
        self.root.mainloop()
    
    def destroy(self) -> None:
        """Clean up and close the GUI"""
        self.root.destroy()