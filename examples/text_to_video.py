"""
Example: Text to Video Generation

Demonstrates generating videos from text descriptions.
"""

from victor_prime import OmniPipeline, VictorPrimeConfig
import torch
import numpy as np
import imageio

# Create pipeline
config = VictorPrimeConfig(
    image_resolution=256,  # Lower resolution for faster video generation
    video_fps=24,
    content_filter_level="minimal",
    device="cuda",
)

pipeline = OmniPipeline(config=config)

# Generate video from text
prompt = "A sunrise over mountains with clouds moving"
video_tensor = pipeline.generate_video(prompt, num_frames=48)  # 2 seconds at 24fps

# Convert to numpy array for saving
# video_tensor shape: [num_frames, 3, H, W]
video_np = ((video_tensor[0].cpu().numpy() + 1) * 127.5).astype(np.uint8)
video_frames = []
for frame_idx in range(video_np.shape[0]):
    frame = np.transpose(video_np[frame_idx], (1, 2, 0))
    video_frames.append(frame)

# Save video
imageio.mimsave("generated_video.mp4", video_frames, fps=24)
print("Video saved to generated_video.mp4")
