#!/usr/bin/env python3
"""
Test script to load and test the QoQ-Med-Omni-7B model locally.
"""

import torch
import sys
import os

# Add the model directory to path
sys.path.insert(0, '/Users/dvd/PycharmProjects/verl/QoQ-Med-Omni-7B')

def test_model_loading():
    """Test loading the model locally"""

    print("=" * 60)
    print("Testing QoQ-Med-Omni-7B Model Loading")
    print("=" * 60)

    # Import the model classes
    from configuration_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLConfig
    from modeling_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLForConditionalGeneration

    print("\n1. Creating model configuration...")
    config = TimeSeriesQwen2_5_VLConfig()
    print(f"   Config created: {config.model_type}")
    print(f"   Time-series token ID: {config.time_series_token_id}")
    print(f"   Time-series embed dim: {config.time_series_embed_dim}")
    print(f"   Text hidden size: {config.text_config.hidden_size}")

    print("\n2. Initializing model...")
    try:
        model = TimeSeriesQwen2_5_VLForConditionalGeneration(config)
        print("   ✓ Model initialized successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"   ✗ Error initializing model: {e}")
        return

    print("\n3. Testing ECG-JEPA encoder...")
    try:
        # Create dummy ECG data
        ecg_data = torch.randn(2, 8, 2500)  # batch=2, channels=8, time=2500

        # Test the time-series embedding module
        with torch.no_grad():
            ts_embeds = model.time_series_embedding(ecg_data)

        print(f"   ✓ ECG encoder works")
        print(f"   Input shape: {ecg_data.shape}")
        print(f"   Output shape: {ts_embeds.shape}")
        print(f"   Expected: (2, 1, {config.text_config.hidden_size})")

    except Exception as e:
        print(f"   ✗ Error in ECG encoder: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n4. Loading ECG-JEPA weights...")
    checkpoint_path = '/Users/dvd/PycharmProjects/verl/epoch100.pth'
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            encoder_state_dict = checkpoint['encoder']

            # Load weights into the encoder
            model.time_series_embedding.encoder.load_state_dict(encoder_state_dict, strict=True)
            print(f"   ✓ ECG-JEPA weights loaded from {checkpoint_path}")

        except Exception as e:
            print(f"   ✗ Error loading weights: {e}")
    else:
        print(f"   ⚠ Checkpoint not found at {checkpoint_path}")

    print("\n5. Testing forward pass with time-series...")
    try:
        # Create input tokens with time-series placeholder
        batch_size = 2
        seq_length = 10

        # Create input_ids with time-series token
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        input_ids[:, 3] = config.time_series_token_id  # Insert time-series token

        # Create ECG data
        ecg_data = torch.randn(batch_size, 8, 2500)

        print(f"   Input IDs shape: {input_ids.shape}")
        print(f"   Time-series token positions: {(input_ids == config.time_series_token_id).nonzero()[:, 1].tolist()}")

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                time_series_data=ecg_data,
                use_cache=False
            )

        print(f"   ✓ Forward pass successful")
        print(f"   Output logits shape: {outputs.logits.shape}")

    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n6. Testing without time-series (regular text)...")
    try:
        # Test without time-series data
        input_ids_text = torch.randint(0, 1000, (1, 20))

        with torch.no_grad():
            outputs_text = model(
                input_ids=input_ids_text,
                time_series_data=None,
                use_cache=False
            )

        print(f"   ✓ Text-only forward pass successful")
        print(f"   Output shape: {outputs_text.logits.shape}")

    except Exception as e:
        print(f"   ✗ Error in text-only pass: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✓ Model testing completed successfully!")
    print("=" * 60)

    return model

def test_model_with_transformers():
    """Test loading the model using transformers AutoModel"""

    print("\n" + "=" * 60)
    print("Testing with Transformers AutoModel")
    print("=" * 60)

    try:
        from transformers import AutoModel, AutoConfig

        # Register the custom model
        from configuration_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLConfig
        from modeling_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLForConditionalGeneration

        AutoConfig.register("time_series_qwen2_5_vl", TimeSeriesQwen2_5_VLConfig)
        AutoModel.register(TimeSeriesQwen2_5_VLConfig, TimeSeriesQwen2_5_VLForConditionalGeneration)

        print("\n1. Loading from directory...")
        model = AutoModel.from_pretrained(
            "/Users/dvd/PycharmProjects/verl/QoQ-Med-Omni-7B",
            trust_remote_code=True,
            local_files_only=True
        )

        print("   ✓ Model loaded via AutoModel.from_pretrained")
        print(f"   Model type: {type(model).__name__}")

    except Exception as e:
        print(f"   ✗ Error loading via AutoModel: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test direct loading
    model = test_model_loading()

    # Test with transformers
    test_model_with_transformers()