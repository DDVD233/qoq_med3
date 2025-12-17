#!/usr/bin/env python3
"""
Complete test for multimodal forward pass with time-series, images, and text.
This properly handles grid dimensions using the processor.
"""

import torch
import sys
import os
import numpy as np
from PIL import Image

# Add the model directory to path
sys.path.insert(0, '/Users/dvd/PycharmProjects/verl/QoQ-Med-Omni-7B')

def create_dummy_image(size=(336, 336)):
    """Create a dummy RGB image with the correct size for Qwen2.5-VL"""
    # Create a simple gradient image for testing
    img_array = np.zeros((*size, 3), dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            img_array[i, j] = [
                int(255 * i / size[0]),  # Red gradient
                int(255 * j / size[1]),  # Green gradient
                128  # Constant blue
            ]
    return Image.fromarray(img_array)

def test_multimodal_with_processor():
    """Test multimodal forward pass using the processor to handle grid dimensions"""

    print("=" * 60)
    print("Complete Multimodal Test with Processor")
    print("=" * 60)

    # Import required modules
    from configuration_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLConfig
    from modeling_time_series_qwen2_5_vl import TimeSeriesQwen2_5_VLForConditionalGeneration

    try:
        from transformers import Qwen2VLProcessor
        # Try to load the processor
        processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        print("✓ Loaded Qwen2VL processor")
    except Exception as e:
        print(f"Could not load processor: {e}")
        print("Testing without processor...")
        processor = None

    print("\n1. Setting up configuration and model...")

    # Create config with reduced size for testing
    config = TimeSeriesQwen2_5_VLConfig()
    config.text_config.num_hidden_layers = 2  # Reduce layers for faster testing
    config.text_config.intermediate_size = 2048  # Smaller intermediate size

    print(f"   Model type: {config.model_type}")
    print(f"   Hidden size: {config.text_config.hidden_size}")
    print(f"   Time-series token ID: {config.time_series_token_id}")

    print("\n2. Initializing model...")
    model = TimeSeriesQwen2_5_VLForConditionalGeneration(config)
    model.eval()  # Set to evaluation mode
    print("   ✓ Model initialized successfully")

    # Load ECG-JEPA weights if available
    checkpoint_path = '/Users/dvd/PycharmProjects/verl/epoch100.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.time_series_embedding.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        print(f"   ✓ ECG-JEPA weights loaded")

    print("\n3. Testing multimodal input with processor...")

    if processor is not None:
        # Create multimodal input
        text = "Patient ECG shows abnormal rhythm. The chest X-ray image indicates potential consolidation."
        image = create_dummy_image((336, 336))  # Use proper size for Qwen2.5-VL

        # Create messages in the format expected by the processor
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text}
            ]
        }]

        # Apply chat template
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Process with the processor - this will calculate proper grid dimensions
        model_inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt"
        )

        print(f"   Input IDs shape: {model_inputs['input_ids'].shape}")
        if 'image_grid_thw' in model_inputs:
            print(f"   Image grid thw: {model_inputs['image_grid_thw']}")

        # Now add time-series data
        ecg_data = torch.randn(1, 8, 2500)
        print(f"   ECG data shape: {ecg_data.shape}")

        # Find where to insert time-series token
        input_ids = model_inputs['input_ids']

        # Insert time-series token at a specific position
        # We'll insert it after the image tokens
        ts_token_id = config.time_series_token_id

        # Find a good position to insert (after first few tokens)
        insert_pos = 10
        if input_ids.shape[1] > insert_pos:
            # Insert the time-series token
            input_ids_with_ts = torch.cat([
                input_ids[:, :insert_pos],
                torch.tensor([[ts_token_id]]),
                input_ids[:, insert_pos:]
            ], dim=1)
            model_inputs['input_ids'] = input_ids_with_ts

        print(f"   Modified input IDs shape: {model_inputs['input_ids'].shape}")

        # Test forward pass with all modalities
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=model_inputs['input_ids'],
                    attention_mask=model_inputs.get('attention_mask'),
                    pixel_values=model_inputs.get('pixel_values'),
                    image_grid_thw=model_inputs.get('image_grid_thw'),
                    time_series_data=ecg_data,
                    use_cache=False
                )

            print("   ✓ Multimodal forward pass successful!")
            print(f"   Output shape: {outputs.logits.shape}")

        except Exception as e:
            print(f"   ✗ Error in multimodal forward pass: {e}")
            import traceback
            traceback.print_exc()

    print("\n4. Testing direct model input (without processor)...")

    # Test with direct inputs
    batch_size = 1
    seq_length = 30

    # Create input with time-series token
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    input_ids[0, 10] = config.time_series_token_id

    # Create ECG data
    ecg_data = torch.randn(batch_size, 8, 2500)

    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   ECG data shape: {ecg_data.shape}")

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                time_series_data=ecg_data,
                use_cache=False
            )

        print("   ✓ Time-series only forward pass successful!")
        print(f"   Output shape: {outputs.logits.shape}")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n5. Testing batch processing with time-series...")

    batch_size = 2
    seq_length = 25

    # Create batch inputs with time-series tokens
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    input_ids[:, 5] = config.time_series_token_id  # Insert at position 5 for both samples

    # Batch ECG data
    ecg_data_batch = torch.randn(batch_size, 8, 2500)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                time_series_data=ecg_data_batch,
                use_cache=False
            )

        print("   ✓ Batch processing successful!")
        print(f"   Batch size: {batch_size}")
        print(f"   Output shape: {outputs.logits.shape}")

    except Exception as e:
        print(f"   ✗ Error in batch processing: {e}")

    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)

    return model

if __name__ == "__main__":
    model = test_multimodal_with_processor()