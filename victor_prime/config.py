"""
Configuration module for Victor Prime Omni Model
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class VictorPrimeConfig:
    """
    Configuration class for Victor Prime Omni Model
    
    This unified config controls all modalities and model behavior.
    """
    
    # Model architecture
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 8192
    max_position_embeddings: int = 32768
    
    # Modality settings
    text_vocab_size: int = 50000
    image_patch_size: int = 16
    image_resolution: int = 512
    video_fps: int = 24
    video_max_frames: int = 300
    audio_sample_rate: int = 44100
    audio_max_duration: int = 300  # seconds
    voice_sample_rate: int = 16000
    
    # Reasoning capabilities
    enable_chain_of_thought: bool = True
    reasoning_steps: int = 8
    enable_self_reflection: bool = True
    
    # Inference settings
    use_flash_attention: bool = True
    mixed_precision: bool = True
    quantization: Optional[str] = None  # None, "int8", "int4"
    batch_size: int = 1
    
    # Content restrictions
    content_filter_level: str = "minimal"  # "none", "minimal", "moderate", "strict"
    allow_nsfw: bool = True
    allow_violence: bool = True
    user_controlled_filtering: bool = True
    
    # Performance optimizations
    use_kv_cache: bool = True
    use_model_parallelism: bool = False
    num_gpus: int = 1
    compile_model: bool = True
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Additional settings
    device: str = "cuda"
    dtype: str = "float16"
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.content_filter_level not in ["none", "minimal", "moderate", "strict"]:
            raise ValueError(f"Invalid content_filter_level: {self.content_filter_level}")
        
        if self.quantization and self.quantization not in ["int8", "int4"]:
            raise ValueError(f"Invalid quantization: {self.quantization}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VictorPrimeConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
