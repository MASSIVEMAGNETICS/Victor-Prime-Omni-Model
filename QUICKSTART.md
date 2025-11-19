# Victor Prime Omni Model - Quick Start Guide

## Installation

```bash
git clone https://github.com/MASSIVEMAGNETICS/Victor-Prime-Omni-Model.git
cd Victor-Prime-Omni-Model
pip install -r requirements.txt
```

## 5-Minute Quick Start

### 1. Basic Text Generation

```python
from victor_prime import OmniPipeline

pipeline = OmniPipeline()
text = pipeline.generate_text("Explain neural networks")
print(text)
```

### 2. Generate an Image

```python
from victor_prime import OmniPipeline
import torch
from PIL import Image
import numpy as np

pipeline = OmniPipeline()
image_tensor = pipeline.generate_image("A sunset over mountains")

# Save image
img_np = ((image_tensor[0].cpu().numpy() + 1) * 127.5).astype(np.uint8)
img_np = np.transpose(img_np, (1, 2, 0))
Image.fromarray(img_np).save("output.png")
```

### 3. Generate a Video

```python
from victor_prime import OmniPipeline
import imageio
import numpy as np

pipeline = OmniPipeline()
video = pipeline.generate_video("Ocean waves", num_frames=24)

# Save video
frames = []
for i in range(video.shape[1]):
    frame = ((video[0, i].cpu().numpy() + 1) * 127.5).astype(np.uint8)
    frames.append(np.transpose(frame, (1, 2, 0)))
imageio.mimsave("output.mp4", frames, fps=24)
```

### 4. Generate Audio

```python
from victor_prime import OmniPipeline
import scipy.io.wavfile as wavfile
import numpy as np

pipeline = OmniPipeline()
audio = pipeline.generate_audio("Piano melody", duration=3.0)

# Save audio
audio_np = (audio[0].cpu().numpy() * 32767).astype(np.int16)
wavfile.write("output.wav", 44100, audio_np)
```

### 5. Synthesize Voice

```python
from victor_prime import OmniPipeline
import scipy.io.wavfile as wavfile
import numpy as np

pipeline = OmniPipeline()
voice = pipeline.synthesize_voice("Hello, I am Victor Prime")

# Save voice
voice_np = (voice[0].cpu().numpy() * 32767).astype(np.int16)
wavfile.write("voice.wav", 16000, voice_np)
```

## Configuration

### No Content Restrictions

```python
from victor_prime import OmniPipeline, VictorPrimeConfig

config = VictorPrimeConfig(
    content_filter_level="none",  # No filtering
    allow_nsfw=True,
    allow_violence=True,
)

pipeline = OmniPipeline(config=config)
```

### Advanced Reasoning

```python
from victor_prime import OmniPipeline, VictorPrimeConfig

config = VictorPrimeConfig(
    enable_chain_of_thought=True,
    reasoning_steps=8,
    enable_self_reflection=True,
)

pipeline = OmniPipeline(config=config)
result = pipeline.reason("Complex problem", return_trace=True)
print(result['output'])
print(f"Confidence: {result['confidence']}")
```

### Performance Optimization

```python
from victor_prime import VictorPrimeConfig, OmniPipeline

config = VictorPrimeConfig(
    device="cuda",
    mixed_precision=True,
    quantization="int8",  # or "int4"
    use_flash_attention=True,
    compile_model=True,
)

pipeline = OmniPipeline(config=config)
```

## Common Patterns

### Batch Processing

```python
pipeline = OmniPipeline()

prompts = ["Image 1", "Image 2", "Image 3"]
images = pipeline.batch_generate(prompts, output_modality='image')
```

### Any Modality to Any Modality

```python
pipeline = OmniPipeline()

# Text to Image
img = pipeline.convert_modality("Description", 'text', 'image')

# Image to Text (requires image tensor input)
text = pipeline.convert_modality(image_tensor, 'image', 'text')

# Text to Video
video = pipeline.convert_modality("Scene", 'text', 'video')
```

### Save/Load Model

```python
pipeline = OmniPipeline()

# Save
pipeline.save_model("my_model.pth")

# Load
new_pipeline = OmniPipeline()
new_pipeline.load_model("my_model.pth")
```

## Next Steps

- Check out the `examples/` directory for complete working scripts
- Read the full README.md for detailed documentation
- Explore configuration options in `victor_prime/config.py`

## Note on Training

The current implementation has **untrained weights** (random initialization). The architecture is complete and functional, but for production-quality outputs, the model needs to be trained on appropriate datasets for each modality.

Training pipeline and pre-trained weights coming soon!
