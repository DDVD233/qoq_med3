#!/usr/bin/env python3
"""
Simple test script to verify model configuration and ECG-JEPA integration.
"""

import torch
import sys
import os

# Add the model directory to path
sys.path.insert(0, '/Users/dvd/PycharmProjects/verl/QoQ-Med-Omni-7B')

def test_ecg_jepa_integration():
    """Test ECG-JEPA encoder and time-series embedding"""

    print("=" * 60)
    print("Testing ECG-JEPA Integration")
    print("=" * 60)

    # Import required modules
    from configuration_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLConfig
    from ecg_jepa_encoder import MaskTransformer

    print("\n1. Testing configuration...")
    config = TimeSeriesQwen2_5_VLConfig()
    print(f"   Model type: {config.model_type}")
    print(f"   Time-series token ID: {config.time_series_token_id}")
    print(f"   Time-series embed dim: {config.time_series_embed_dim}")
    print(f"   Text hidden size: {config.text_config.hidden_size}")
    print(f"   Freeze encoder: {config.freeze_time_series_encoder}")

    print("\n2. Testing ECG-JEPA encoder standalone...")
    encoder = MaskTransformer(
        embed_dim=768,
        depth=12,
        num_heads=16,
        c=8,
        p=50,
        t=50,
        pos_type='sincos',
        mask_scale=(0, 0),
        leads=list(range(8))
    )

    # Load weights
    checkpoint_path = '/Users/dvd/PycharmProjects/verl/epoch100.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder'], strict=True)
        print(f"   ✓ Weights loaded from {checkpoint_path}")
    else:
        print(f"   ⚠ No weights found at {checkpoint_path}")

    # Test forward pass
    ecg_data = torch.randn(2, 8, 2500)
    with torch.no_grad():
        output = encoder.representation(ecg_data)
    print(f"   ✓ Forward pass successful")
    print(f"   Input: {ecg_data.shape} -> Output: {output.shape}")

    print("\n3. Testing TimeSeriesEmbedding module...")
    from modeling_time_series_qwen2_5_vl import TimeSeriesEmbedding

    ts_embed = TimeSeriesEmbedding(config)

    # Load weights into the embedding module
    if os.path.exists(checkpoint_path):
        ts_embed.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        print(f"   ✓ Weights loaded into TimeSeriesEmbedding")

    # Test forward pass
    with torch.no_grad():
        embedded = ts_embed(ecg_data)
    print(f"   ✓ TimeSeriesEmbedding forward pass")
    print(f"   Input: {ecg_data.shape} -> Output: {embedded.shape}")
    print(f"   Expected output shape: (2, 1, {config.text_config.hidden_size})")

    print("\n4. Testing model initialization (without weights)...")
    try:
        from modeling_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLForConditionalGeneration

        # Create a minimal config to avoid loading large weights
        minimal_config = TimeSeriesQwen2_5_VLConfig()
        minimal_config.text_config.num_hidden_layers = 2  # Reduce layers for testing
        minimal_config.text_config.intermediate_size = 1024  # Smaller intermediate size

        print("   Creating model with minimal config...")
        model = TimeSeriesQwen2_5_VLForConditionalGeneration(minimal_config)
        print(f"   ✓ Model created successfully")

        # Test components
        print(f"   Time-series embedding: {type(model.time_series_embedding).__name__}")
        print(f"   ECG-JEPA encoder: {type(model.time_series_embedding.encoder).__name__}")

    except Exception as e:
        print(f"   ✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✓ Testing completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_ecg_jepa_integration()