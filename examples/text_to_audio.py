"""
Example: Text to Audio Generation

Demonstrates generating audio from text descriptions.
"""

from victor_prime import OmniPipeline, VictorPrimeConfig
import torch
import numpy as np
import scipy.io.wavfile as wavfile

# Create pipeline
config = VictorPrimeConfig(
    audio_sample_rate=44100,
    content_filter_level="minimal",
    device="cuda",
)

pipeline = OmniPipeline(config=config)

# Generate audio from text
prompt = "Ocean waves crashing on a beach with seagulls"
audio_tensor = pipeline.generate_audio(prompt, duration=5.0)

# Convert to numpy for saving
audio_np = audio_tensor[0].cpu().numpy()

# Normalize to int16 range
audio_np = (audio_np * 32767).astype(np.int16)

# Save audio
wavfile.write("generated_audio.wav", config.audio_sample_rate, audio_np)
print("Audio saved to generated_audio.wav")
