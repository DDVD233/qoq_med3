#!/usr/bin/env python3
"""
Test multimodal forward pass with QoQ-Med-Omni-7B model.
Loads time-series ECG data and X-ray image for inference.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

MODEL_ID = "ddvd233/QoQ-Med-Omni-7B"
ECG_PATH = "tests/time_series/test_ecg.pt"
IMAGE_PATH = "tests/time_series/test_xray.jpg"


def test_multimodal():
    print("Loading model and processor from HuggingFace...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"Model loaded: {type(model).__name__}")

    ecg_data = torch.load(ECG_PATH, weights_only=True).unsqueeze(0)  # Add batch dim: [1, 8, 2500]
    image = Image.open(IMAGE_PATH)
    print(f"ECG shape: {ecg_data.shape}")
    print(f"Image size: {image.size}")

    messages = [{
        "role": "user",
        "content": [
            {"type": "time-series"},
            {"type": "image"},
            {"type": "text", "text": "Given the input ECG and image, how long would the patient stay in the hospital?"}
        ]
    }]

    # Process inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[prompt],
        images=[image],
        time_series_data=ecg_data,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    print("\nGenerating response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )

    response = processor.batch_decode(
        output_ids[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )[0]

    print(f"\nModel response:\n{response}")
    return response

if __name__ == "__main__":
    test_multimodal()
