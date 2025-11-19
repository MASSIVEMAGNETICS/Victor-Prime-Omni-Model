"""
Pipeline interface for Victor Prime Omni Model

Provides high-level API for seamless multi-modal generation.
"""

import torch
from typing import Optional, Union, Any, Dict, List
from .model import VictorPrimeModel
from .config import VictorPrimeConfig


class OmniPipeline:
    """
    Unified pipeline for all modality transformations
    
    This provides a simple interface for:
    - Text generation
    - Image generation from text
    - Video generation from text
    - Audio generation from text
    - Voice synthesis from text
    - Multi-modal reasoning
    - And all other modality combinations
    """
    
    def __init__(
        self,
        config: Optional[VictorPrimeConfig] = None,
        model: Optional[VictorPrimeModel] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the pipeline
        
        Args:
            config: Model configuration (creates default if None)
            model: Pre-initialized model (creates new if None)
            device: Device to run on (cuda/cpu)
        """
        if config is None:
            config = VictorPrimeConfig()
        
        if device is not None:
            config.device = device
        
        self.config = config
        
        if model is None:
            self.model = VictorPrimeModel(config)
        else:
            self.model = model
        
        # Move to device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set to eval mode by default
        self.model.eval()
        
    def generate_text(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a text prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        with torch.no_grad():
            result = self.model(
                input_modality='text',
                output_modality='text',
                input_data=prompt,
                max_length=max_length,
                temperature=temperature,
                **kwargs
            )
        return result['output']
    
    def generate_image(
        self,
        prompt: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate image from text prompt
        
        Args:
            prompt: Text description of desired image
            
        Returns:
            Generated image tensor [3, H, W]
        """
        with torch.no_grad():
            result = self.model(
                input_modality='text',
                output_modality='image',
                input_data=prompt,
                **kwargs
            )
        return result['output']
    
    def generate_video(
        self,
        prompt: str,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate video from text prompt
        
        Args:
            prompt: Text description of desired video
            num_frames: Number of frames to generate
            fps: Frames per second
            
        Returns:
            Generated video tensor [T, 3, H, W]
        """
        if num_frames is None:
            num_frames = self.config.video_fps
        
        with torch.no_grad():
            result = self.model(
                input_modality='text',
                output_modality='video',
                input_data=prompt,
                num_frames=num_frames,
                **kwargs
            )
        return result['output']
    
    def generate_audio(
        self,
        prompt: str,
        duration: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate audio from text prompt
        
        Args:
            prompt: Text description of desired audio
            duration: Duration in seconds
            
        Returns:
            Generated audio waveform
        """
        with torch.no_grad():
            result = self.model(
                input_modality='text',
                output_modality='audio',
                input_data=prompt,
                duration=duration,
                **kwargs
            )
        return result['output']
    
    def synthesize_voice(
        self,
        text: str,
        duration: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Synthesize voice from text
        
        Args:
            text: Text to speak
            duration: Duration in seconds
            
        Returns:
            Generated voice waveform
        """
        with torch.no_grad():
            result = self.model(
                input_modality='text',
                output_modality='voice',
                input_data=text,
                duration=duration,
                **kwargs
            )
        return result['output']
    
    def understand_image(
        self,
        image: torch.Tensor,
        **kwargs
    ) -> str:
        """
        Generate text description of an image
        
        Args:
            image: Input image tensor [3, H, W] or [B, 3, H, W]
            
        Returns:
            Text description
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            result = self.model(
                input_modality='image',
                output_modality='text',
                input_data=image,
                **kwargs
            )
        return result['output']
    
    def convert_modality(
        self,
        input_data: Any,
        input_modality: str,
        output_modality: str,
        **kwargs
    ) -> Any:
        """
        Generic modality conversion
        
        Args:
            input_data: Input in source modality
            input_modality: Source modality type
            output_modality: Target modality type
            
        Returns:
            Output in target modality
        """
        with torch.no_grad():
            result = self.model(
                input_modality=input_modality,
                output_modality=output_modality,
                input_data=input_data,
                **kwargs
            )
        return result['output']
    
    def reason(
        self,
        prompt: str,
        modality: str = 'text',
        return_trace: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply reasoning to a prompt and return detailed trace
        
        Args:
            prompt: Input prompt
            modality: Output modality after reasoning
            return_trace: Whether to return reasoning trace
            
        Returns:
            Dictionary with output and reasoning trace
        """
        with torch.no_grad():
            result = self.model(
                input_modality='text',
                output_modality=modality,
                input_data=prompt,
                enable_reasoning=True,
                **kwargs
            )
        
        if return_trace:
            return {
                'output': result['output'],
                'reasoning_trace': result['reasoning_trace'],
                'confidence': result['reasoning_trace']['confidence_scores'] if result['reasoning_trace'] else None,
            }
        else:
            return {'output': result['output']}
    
    def batch_generate(
        self,
        prompts: List[str],
        output_modality: str = 'text',
        **kwargs
    ) -> List[Any]:
        """
        Generate outputs for multiple prompts
        
        Args:
            prompts: List of text prompts
            output_modality: Target modality for all outputs
            
        Returns:
            List of generated outputs
        """
        outputs = []
        for prompt in prompts:
            output = self.convert_modality(
                input_data=prompt,
                input_modality='text',
                output_modality=output_modality,
                **kwargs
            )
            outputs.append(output)
        
        return outputs
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = VictorPrimeConfig.from_dict(checkpoint['config'])
    
    def __call__(
        self,
        prompt: str,
        output_modality: str = 'text',
        **kwargs
    ) -> Any:
        """
        Convenient call interface
        
        Args:
            prompt: Input prompt
            output_modality: Desired output modality
            
        Returns:
            Generated output
        """
        return self.convert_modality(
            input_data=prompt,
            input_modality='text',
            output_modality=output_modality,
            **kwargs
        )
