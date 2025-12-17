#!/usr/bin/env python3
"""
Test script to verify ECG-JEPA weight loading compatibility.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, '/Users/dvd/PycharmProjects/verl/QoQ-Med-Omni-7B')

from ecg_jepa_encoder import MaskTransformer

def test_weight_loading():
    """Test loading ECG-JEPA weights into our MaskTransformer"""

    print("Initializing MaskTransformer...")
    model = MaskTransformer(
        embed_dim=768,
        depth=12,
        num_heads=16,
        c=8,
        p=50,
        t=50,
        pos_type='sincos',
        mask_scale=(0, 0),  # No masking for inference
        leads=list(range(8))
    )

    # Check if checkpoint exists
    checkpoint_path = '/Users/dvd/PycharmProjects/verl/epoch100.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Checking model structure...")

        # Print model structure
        print("\nModel structure:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")

        print("\nModel state dict keys:")
        for key in model.state_dict().keys():
            print(f"  {key}")

        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Print checkpoint structure
    print("\nCheckpoint keys:")
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            print(f"  {key}")

        # Try to find the model state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'encoder' in checkpoint:
            state_dict = checkpoint['encoder']
        else:
            # Assume checkpoint is the state dict itself
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    print("\nState dict keys (first 20):")
    for i, key in enumerate(state_dict.keys()):
        if i >= 20:
            print(f"  ... and {len(state_dict.keys()) - 20} more")
            break
        print(f"  {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else type(state_dict[key])}")

    # Try to load the weights
    print("\nAttempting to load weights...")
    try:
        # Try strict loading first
        model.load_state_dict(state_dict, strict=True)
        print("✓ Successfully loaded all weights (strict=True)")
    except Exception as e:
        print(f"Strict loading failed: {e}")

        # Try non-strict loading to see what matches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"\nPartial loading results:")
        if missing_keys:
            print(f"  Missing keys ({len(missing_keys)}):")
            for key in missing_keys[:10]:
                print(f"    - {key}")
            if len(missing_keys) > 10:
                print(f"    ... and {len(missing_keys) - 10} more")

        if unexpected_keys:
            print(f"  Unexpected keys ({len(unexpected_keys)}):")
            for key in unexpected_keys[:10]:
                print(f"    - {key}")
            if len(unexpected_keys) > 10:
                print(f"    ... and {len(unexpected_keys) - 10} more")

    # Test forward pass
    print("\nTesting forward pass...")
    try:
        # Create dummy ECG data (batch_size=1, channels=8, time_points=2500)
        dummy_ecg = torch.randn(1, 8, 2500)

        # Test representation method
        with torch.no_grad():
            output = model.representation(dummy_ecg)

        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {dummy_ecg.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: (1, 768)")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

if __name__ == "__main__":
    test_weight_loading()