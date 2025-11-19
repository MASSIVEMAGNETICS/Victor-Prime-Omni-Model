"""
Example: Basic Text Generation

Demonstrates text-to-text generation with reasoning.
"""

from victor_prime import OmniPipeline, VictorPrimeConfig

# Create pipeline with minimal content restrictions
config = VictorPrimeConfig(
    content_filter_level="minimal",
    enable_chain_of_thought=True,
    device="cuda",  # Use "cpu" if no GPU available
)

pipeline = OmniPipeline(config=config)

# Generate text with reasoning
prompt = "Explain how neural networks learn through backpropagation"
result = pipeline.reason(prompt, modality='text', return_trace=True)

print("Generated Text:")
print(result['output'])
print("\nReasoning Confidence Scores:")
print(result['confidence'])
