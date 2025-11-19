"""
Example: Voice Synthesis

Demonstrates converting text to natural-sounding voice.
"""

from victor_prime import OmniPipeline, VictorPrimeConfig
import torch
import numpy as np
import scipy.io.wavfile as wavfile

# Create pipeline
config = VictorPrimeConfig(
    voice_sample_rate=16000,
    content_filter_level="minimal",
    device="cuda",
)

pipeline = OmniPipeline(config=config)

# Synthesize voice from text
text = "Welcome to Victor Prime, the unified multi-modal AGI model."
voice_tensor = pipeline.synthesize_voice(text)

# Convert to numpy for saving
voice_np = voice_tensor[0].cpu().numpy()

# Normalize to int16 range
voice_np = (voice_np * 32767).astype(np.int16)

# Save voice
wavfile.write("generated_voice.wav", config.voice_sample_rate, voice_np)
print("Voice saved to generated_voice.wav")
