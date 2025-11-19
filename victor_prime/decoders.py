"""
Modality-specific decoders for Victor Prime Omni Model

Each decoder converts the unified latent representation into its target modality.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Any
from .config import VictorPrimeConfig


class TextDecoder(nn.Module):
    """Decode unified latent space into text"""
    
    def __init__(self, config: VictorPrimeConfig):
        super().__init__()
        self.config = config
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.text_vocab_size)
        
        # Additional decoder layers for autoregressive generation
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                batch_first=True,
            )
            for _ in range(4)  # Fewer layers for decoder
        ])
        
    def forward(
        self,
        latent_repr: torch.Tensor,
        max_length: int = 512,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Decode latent representation into text
        
        Args:
            latent_repr: Latent representation [batch, seq_len, hidden_size]
            max_length: Maximum generation length
            temperature: Sampling temperature (uses config default if None)
            
        Returns:
            Generated text string
        """
        temperature = temperature or self.config.temperature
        
        # Process through decoder layers
        hidden = latent_repr
        for layer in self.decoder_layers:
            hidden = layer(hidden, latent_repr)
        
        # Generate tokens autoregressively (simplified)
        logits = self.lm_head(hidden)
        
        # Sample from distribution (simplified greedy decoding)
        token_ids = torch.argmax(logits, dim=-1)
        
        # Convert to text (simplified)
        text = self._decode_tokens(token_ids[0])
        
        return text
    
    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs to text (placeholder)"""
        # In production, use proper detokenizer
        chars = [chr(tid.item() % 128) for tid in token_ids[:100]]
        return ''.join(c for c in chars if c.isprintable())


class ImageDecoder(nn.Module):
    """Decode unified latent space into images"""
    
    def __init__(self, config: VictorPrimeConfig):
        super().__init__()
        self.config = config
        
        # Decoder projects from latent to image space
        # Using a simple upsampling approach
        
        hidden_size = config.hidden_size
        resolution = config.image_resolution
        
        # Project to larger dimension
        self.input_proj = nn.Linear(hidden_size, hidden_size * 4)
        
        # Convolutional upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_size * 4, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    def forward(
        self,
        latent_repr: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode latent representation into image
        
        Args:
            latent_repr: Latent representation [batch, seq_len, hidden_size]
            
        Returns:
            Generated image [batch, 3, height, width]
        """
        # Use mean pooling of sequence
        latent = latent_repr.mean(dim=1)  # [batch, hidden_size]
        
        # Project and reshape
        features = self.input_proj(latent)  # [batch, hidden_size * 4]
        
        # Reshape to 2D feature map (starting small, will upsample)
        batch_size = features.shape[0]
        channels = self.config.hidden_size * 4
        # Start with 4x4 feature map
        features = features.view(batch_size, channels, 1, 1)
        features = features.expand(-1, -1, 4, 4)
        
        # Decode through convolutional layers
        image = self.decoder(features)
        
        # Resize to target resolution if needed
        if image.shape[-1] != self.config.image_resolution:
            image = torch.nn.functional.interpolate(
                image,
                size=(self.config.image_resolution, self.config.image_resolution),
                mode='bilinear',
                align_corners=False,
            )
        
        return image


class VideoDecoder(nn.Module):
    """Decode unified latent space into video"""
    
    def __init__(self, config: VictorPrimeConfig):
        super().__init__()
        self.config = config
        
        # Reuse image decoder for individual frames
        self.frame_decoder = ImageDecoder(config)
        
        # Temporal smoothing
        self.temporal_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(
        self,
        latent_repr: torch.Tensor,
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode latent representation into video
        
        Args:
            latent_repr: Latent representation [batch, seq_len, hidden_size]
            num_frames: Number of frames to generate
            
        Returns:
            Generated video [batch, frames, 3, height, width]
        """
        if num_frames is None:
            num_frames = min(latent_repr.shape[1], self.config.video_fps)
        
        batch_size = latent_repr.shape[0]
        
        # Interpolate latent representation to desired number of frames
        if latent_repr.shape[1] < num_frames:
            # Upsample sequence
            latent_repr = torch.nn.functional.interpolate(
                latent_repr.transpose(1, 2),
                size=num_frames,
                mode='linear',
            ).transpose(1, 2)
        else:
            # Downsample sequence
            indices = torch.linspace(0, latent_repr.shape[1] - 1, num_frames).long()
            latent_repr = latent_repr[:, indices]
        
        # Apply temporal projection
        latent_repr = self.temporal_proj(latent_repr)
        
        # Decode each frame
        frames = []
        for frame_idx in range(num_frames):
            frame_latent = latent_repr[:, frame_idx:frame_idx+1]
            frame = self.frame_decoder(frame_latent)
            frames.append(frame)
        
        # Stack frames
        video = torch.stack(frames, dim=1)
        
        return video


class AudioDecoder(nn.Module):
    """Decode unified latent space into audio"""
    
    def __init__(self, config: VictorPrimeConfig):
        super().__init__()
        self.config = config
        
        # Upsampling decoder for audio generation
        self.projection = nn.Linear(config.hidden_size, 512)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=10, stride=5, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    def forward(
        self,
        latent_repr: torch.Tensor,
        duration: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Decode latent representation into audio waveform
        
        Args:
            latent_repr: Latent representation [batch, seq_len, hidden_size]
            duration: Target duration in seconds
            
        Returns:
            Generated audio [batch, samples]
        """
        # Project features
        features = self.projection(latent_repr)  # [batch, seq_len, 512]
        features = features.transpose(1, 2)  # [batch, 512, seq_len]
        
        # Decode to waveform
        audio = self.decoder(features)  # [batch, 1, samples]
        audio = audio.squeeze(1)  # [batch, samples]
        
        # Adjust to target duration if specified
        if duration is not None:
            target_samples = int(duration * self.config.audio_sample_rate)
            if audio.shape[-1] < target_samples:
                # Pad with zeros
                padding = target_samples - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, padding))
            elif audio.shape[-1] > target_samples:
                # Truncate
                audio = audio[:, :target_samples]
        
        return audio


class VoiceDecoder(nn.Module):
    """Decode unified latent space into voice/speech"""
    
    def __init__(self, config: VictorPrimeConfig):
        super().__init__()
        self.config = config
        
        # Voice decoder with emphasis on speech quality
        self.projection = nn.Linear(config.hidden_size, 256)
        
        # Upsampling path
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        
        # Voice-specific processing
        self.vocoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )
        
    def forward(
        self,
        latent_repr: torch.Tensor,
        duration: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Decode latent representation into voice waveform
        
        Args:
            latent_repr: Latent representation [batch, seq_len, hidden_size]
            duration: Target duration in seconds
            
        Returns:
            Generated voice [batch, samples]
        """
        # Project features
        features = self.projection(latent_repr)  # [batch, seq_len, 256]
        features = features.transpose(1, 2)  # [batch, 256, seq_len]
        
        # Decode to waveform
        voice = self.decoder(features)  # [batch, 1, samples]
        
        # Apply vocoder for speech quality
        voice = self.vocoder(voice)  # [batch, 1, samples]
        voice = voice.squeeze(1)  # [batch, samples]
        
        # Adjust to target duration if specified
        if duration is not None:
            target_samples = int(duration * self.config.voice_sample_rate)
            if voice.shape[-1] < target_samples:
                # Pad with zeros
                padding = target_samples - voice.shape[-1]
                voice = torch.nn.functional.pad(voice, (0, padding))
            elif voice.shape[-1] > target_samples:
                # Truncate
                voice = voice[:, :target_samples]
        
        return voice
