"""
Victor Prime Omni Model - A Unified Multi-Modal AGI Model

A single unified synthesis AGI frontier model that handles:
- Text generation and understanding
- Image generation and understanding
- Video generation and understanding
- Audio generation and understanding
- Voice synthesis and recognition
- Advanced reasoning capabilities

All without calling separate services.
"""

__version__ = "0.1.0"
__author__ = "MASSIVEMAGNETICS"

from .model import VictorPrimeModel
from .config import VictorPrimeConfig
from .pipeline import OmniPipeline

__all__ = [
    "VictorPrimeModel",
    "VictorPrimeConfig",
    "OmniPipeline",
]
