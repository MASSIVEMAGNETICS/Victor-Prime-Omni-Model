"""
Example: Multi-Modal Pipeline

Demonstrates seamless transitions between different modalities.
"""

from victor_prime import OmniPipeline, VictorPrimeConfig

# Create pipeline with full capabilities
config = VictorPrimeConfig(
    content_filter_level="minimal",
    enable_chain_of_thought=True,
    enable_self_reflection=True,
    device="cuda",
)

pipeline = OmniPipeline(config=config)

# Example 1: Text → Text (with reasoning)
print("=== Text Generation with Reasoning ===")
text_result = pipeline.reason(
    "What are the implications of AGI?",
    modality='text',
    return_trace=True
)
print(f"Output: {text_result['output']}")
print(f"Confidence: {text_result['confidence']}\n")

# Example 2: Text → Image
print("=== Image Generation ===")
image = pipeline.generate_image("A futuristic AI laboratory")
print(f"Generated image shape: {image.shape}\n")

# Example 3: Text → Video
print("=== Video Generation ===")
video = pipeline.generate_video("Robots working together", num_frames=24)
print(f"Generated video shape: {video.shape}\n")

# Example 4: Text → Audio
print("=== Audio Generation ===")
audio = pipeline.generate_audio("Electronic music with drums", duration=3.0)
print(f"Generated audio shape: {audio.shape}\n")

# Example 5: Text → Voice
print("=== Voice Synthesis ===")
voice = pipeline.synthesize_voice("Hello, I am Victor Prime.")
print(f"Generated voice shape: {voice.shape}\n")

# Example 6: Batch processing
print("=== Batch Processing ===")
prompts = [
    "A sunset over the ocean",
    "Mountains covered in snow",
    "A bustling city street",
]
images = pipeline.batch_generate(prompts, output_modality='image')
print(f"Generated {len(images)} images\n")

print("All examples completed successfully!")
