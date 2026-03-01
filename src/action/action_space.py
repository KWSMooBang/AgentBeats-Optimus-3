"""Action space definitions for Purple Agent

Purple Agent uses Joint Space format:
- Button space: 0-2303 (2,304 button combinations)
- Camera space: 0-120 (11x11 grid, 121 positions)
"""
from typing import Dict, Any


def noop_action() -> Dict[str, Any]:
    """
    Return no-op action
    
    Button Combinations:
        Index 0: all "none" combination (no buttons pressed)
        
    Camera Positions:
        Index 60: center position (0, 0)
        - 11x11 grid in row-major order
        - row: 0-10 (pitch bins)
        - col: 0-10 (yaw bins)
        - center: row=5, col=5 → index = 5*11 + 5 = 60
    
    Returns:
        dict: {"buttons": [0], "camera": [60]}
    """
    return {
        "buttons": [0],
        "camera": [60]
    }
