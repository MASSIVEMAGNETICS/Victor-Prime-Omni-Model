"""
Example: Text to Image Generation

Demonstrates generating images from text descriptions.
"""

from victor_prime import OmniPipeline, VictorPrimeConfig
import torch
from PIL import Image
import numpy as np

# Create pipeline
config = VictorPrimeConfig(
    image_resolution=512,
    content_filter_level="none",  # No content restrictions
    device="cuda",
)

pipeline = OmniPipeline(config=config)

# Generate image from text
prompt = "A cyberpunk city at night with neon lights and flying cars"
image_tensor = pipeline.generate_image(prompt)

# Convert to PIL Image for saving
# Denormalize from [-1, 1] to [0, 255]
image_np = ((image_tensor[0].cpu().numpy() + 1) * 127.5).astype(np.uint8)
image_np = np.transpose(image_np, (1, 2, 0))
image = Image.fromarray(image_np)

# Save image
image.save("generated_image.png")
print("Image saved to generated_image.png")
