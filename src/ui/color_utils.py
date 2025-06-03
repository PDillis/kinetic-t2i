"""
Color utilities for consistent prompt zone colors across the application
"""

# Shared color palette for prompt zones
# Each tuple is (BGR for OpenCV, Hex for UI)
PROMPT_COLORS = [
    ((100, 100, 255), "#FF6464"),  # Red
    ((100, 255, 100), "#64FF64"),  # Green  
    ((255, 100, 100), "#6464FF"),  # Blue
    ((100, 255, 255), "#FFFF64"),  # Yellow
    ((255, 100, 255), "#FF64FF"),  # Magenta
    ((255, 255, 100), "#64FFFF"),  # Cyan
]

def get_prompt_color_bgr(index: int) -> tuple:
    """Get BGR color for OpenCV drawing"""
    return PROMPT_COLORS[index % len(PROMPT_COLORS)][0]

def get_prompt_color_hex(index: int) -> str:
    """Get hex color for UI elements"""
    return PROMPT_COLORS[index % len(PROMPT_COLORS)][1]

def get_num_colors() -> int:
    """Get number of available colors"""
    return len(PROMPT_COLORS)