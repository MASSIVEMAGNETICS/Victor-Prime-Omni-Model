"""
Basic tests for Victor Prime Omni Model

Tests the core architecture and basic functionality.
"""

import torch
import pytest
from victor_prime import VictorPrimeModel, VictorPrimeConfig, OmniPipeline


def test_config_creation():
    """Test configuration creation and validation"""
    config = VictorPrimeConfig()
    assert config.hidden_size == 2048
    assert config.content_filter_level == "minimal"
    
    # Test invalid config
    with pytest.raises(ValueError):
        VictorPrimeConfig(content_filter_level="invalid")


def test_config_dict_conversion():
    """Test config to/from dict"""
    config = VictorPrimeConfig(hidden_size=1024)
    config_dict = config.to_dict()
    assert config_dict['hidden_size'] == 1024
    
    new_config = VictorPrimeConfig.from_dict(config_dict)
    assert new_config.hidden_size == 1024


def test_model_creation():
    """Test model initialization"""
    config = VictorPrimeConfig(
        hidden_size=256,  # Smaller for testing
        num_layers=2,
        num_attention_heads=4,
    )
    model = VictorPrimeModel(config)
    assert model is not None
    assert model.config.hidden_size == 256


def test_text_encoder():
    """Test text encoder"""
    config = VictorPrimeConfig(hidden_size=256, num_layers=2)
    model = VictorPrimeModel(config)
    
    # Test with string input
    output = model.text_encoder("Hello world")
    assert output.shape[-1] == config.hidden_size


def test_image_encoder():
    """Test image encoder"""
    config = VictorPrimeConfig(
        hidden_size=256,
        num_layers=2,
        image_resolution=64,  # Smaller for testing
    )
    model = VictorPrimeModel(config)
    
    # Create dummy image
    batch_size = 1
    image = torch.randn(batch_size, 3, config.image_resolution, config.image_resolution)
    
    output = model.image_encoder(image)
    assert output.shape[-1] == config.hidden_size
    assert output.shape[0] == batch_size


def test_pipeline_creation():
    """Test pipeline initialization"""
    config = VictorPrimeConfig(
        hidden_size=256,
        num_layers=2,
        device="cpu",
    )
    pipeline = OmniPipeline(config=config)
    assert pipeline is not None
    assert pipeline.device.type == "cpu"


def test_text_generation():
    """Test basic text generation"""
    config = VictorPrimeConfig(
        hidden_size=256,
        num_layers=2,
        num_attention_heads=4,
        device="cpu",
    )
    pipeline = OmniPipeline(config=config)
    
    output = pipeline.generate_text("Hello", max_length=10)
    assert isinstance(output, str)


def test_image_generation():
    """Test basic image generation"""
    config = VictorPrimeConfig(
        hidden_size=256,
        num_layers=2,
        num_attention_heads=4,
        image_resolution=64,
        device="cpu",
    )
    pipeline = OmniPipeline(config=config)
    
    output = pipeline.generate_image("A test image")
    assert isinstance(output, torch.Tensor)
    assert output.shape[1] == 3  # RGB channels
    assert output.shape[2] == config.image_resolution


def test_video_generation():
    """Test basic video generation"""
    config = VictorPrimeConfig(
        hidden_size=256,
        num_layers=2,
        num_attention_heads=4,
        image_resolution=64,
        device="cpu",
    )
    pipeline = OmniPipeline(config=config)
    
    output = pipeline.generate_video("A test video", num_frames=4)
    assert isinstance(output, torch.Tensor)
    assert output.shape[1] == 4  # 4 frames
    assert output.shape[2] == 3  # RGB channels


def test_audio_generation():
    """Test basic audio generation"""
    config = VictorPrimeConfig(
        hidden_size=256,
        num_layers=2,
        num_attention_heads=4,
        device="cpu",
    )
    pipeline = OmniPipeline(config=config)
    
    output = pipeline.generate_audio("A test sound", duration=1.0)
    assert isinstance(output, torch.Tensor)
    assert output.dim() == 2  # [batch, samples]


def test_voice_synthesis():
    """Test voice synthesis"""
    config = VictorPrimeConfig(
        hidden_size=256,
        num_layers=2,
        num_attention_heads=4,
        device="cpu",
    )
    pipeline = OmniPipeline(config=config)
    
    output = pipeline.synthesize_voice("Hello world")
    assert isinstance(output, torch.Tensor)
    assert output.dim() == 2  # [batch, samples]


def test_reasoning():
    """Test reasoning module"""
    config = VictorPrimeConfig(
        hidden_size=256,
        num_layers=2,
        num_attention_heads=4,
        enable_chain_of_thought=True,
        reasoning_steps=4,
        device="cpu",
    )
    pipeline = OmniPipeline(config=config)
    
    result = pipeline.reason("Test prompt", return_trace=True)
    assert 'output' in result
    assert 'reasoning_trace' in result
    assert result['reasoning_trace'] is not None


def test_modality_conversion():
    """Test generic modality conversion"""
    config = VictorPrimeConfig(
        hidden_size=256,
        num_layers=2,
        num_attention_heads=4,
        image_resolution=64,
        device="cpu",
    )
    pipeline = OmniPipeline(config=config)
    
    output = pipeline.convert_modality(
        input_data="Test",
        input_modality='text',
        output_modality='image'
    )
    assert isinstance(output, torch.Tensor)


def test_content_filter_levels():
    """Test different content filter levels"""
    for level in ["none", "minimal", "moderate", "strict"]:
        config = VictorPrimeConfig(content_filter_level=level)
        assert config.content_filter_level == level


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
