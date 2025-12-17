# QoQ-Med3: Building Robust Multimodal Diagnosis Foundation Model with Reasoning

## Overview

QoQ-Med is a multimodal medical foundation model designed for clinical diagnosis with interpretable reasoning capabilities. Unlike traditional "black box" clinical AI systems, QoQ-Med generates explicit chain-of-thought reasoning, bounding box annotations highlighting salient regions, and concise clinical diagnoses—enabling clinicians to understand and verify the model's decision-making process.

### Key Features

- **Multimodal Integration**: Processes diverse clinical data including 1D signals (ECG), 2D images (Chest X-ray, dermoscopy, mammography, fundus, pathology), and 3D scans (CT, MRI, ultrasound)
- **Interpretable Reasoning**: Generates step-by-step clinical reasoning traces that can be reviewed by healthcare professionals
- **Visual Grounding**: Produces bounding boxes that localize evidence supporting diagnostic decisions
- **Domain-Aware Training**: Trained using DRPO (Domain-aware Relative Policy Optimization), a novel RL method that balances learning across heterogeneous clinical domains
- **Strong Generalization**: Demonstrates robust transfer learning from general clinical pretraining to EHR-based tasks

## Datasets

### CLIMB Dataset

QoQ-Med is pretrained on [CLIMB](https://github.com/ddvd233/CLIMB), a large-scale multimodal clinical dataset comprising:
- **44 publicly available datasets** across 13 clinical domains
- **707K 2D images** (Chest X-ray, dermoscopy, fundus, pathology, mammography)
- **1.83M 3D scans** (CT, MRI, ultrasound)
- **78.9K ECG recordings**
- **2.61M total QA pairs** with reasoning traces

### MIMIC-IV Dataset

For EHR-based evaluation and fine-tuning, we use [MIMIC-IV](https://physionet.org/content/mimiciv/), which includes:
- Electronic health records from 65,000+ ICU patients and 200,000+ ED patients
- Linked chest radiographs from MIMIC-CXR (377,110 images)
- Discharge summaries and radiology reports from MIMIC-IV-Note
- Comprehensive tabular data including labs, procedures, and diagnoses

## Model Variants

We release several model variants to support different use cases:

| Model | Base | Training Data | Description | Link |
|-------|------|---------------|-------------|------|
| **QoQ-Med-VL-7B** | Qwen2-VL-7B | CLIMB | Previous generation model trained on CLIMB | [🤗 HuggingFace](https://huggingface.co/ddvd233/QoQ-Med-VL-7B) |
| **QoQ-Med3-VL-8B** | Qwen3-VL-8B | CLIMB | Latest model with DRPO training on CLIMB | [🤗 HuggingFace](https://huggingface.co/ddvd233/QoQ-Med3-VL-8B) |
| **QoQ-Med3-VL-8B-MIMIC** | Qwen3-VL-8B | CLIMB → MIMIC-IV | Best overall model: CLIMB pretrained + MIMIC-IV fine-tuned | [🤗 HuggingFace](https://huggingface.co/ddvd233/QoQ-Med3-VL-8B-MIMIC) |
| **Qwen3-VL-8B-MIMIC** | Qwen3-VL-8B | MIMIC-IV | Vanilla Qwen3 trained directly on MIMIC-IV | [🤗 HuggingFace](https://huggingface.co/ddvd233/Qwen3-VL-8B-MIMIC) |

### Model Selection Guide

- **For general medical imaging diagnosis**: Use `QoQ-Med3-VL-8B`
- **For EHR-based clinical tasks**: Use `QoQ-Med3-VL-8B-MIMIC`
- **For MIMIC-IV specific tasks without pretraining**: Use `Qwen3-VL-8B-MIMIC`

## Architecture

QoQ-Med uses a unified multimodal architecture:

1. **Modality-Specific Encoders**: 
   - Vision encoder for 2D/3D medical images
   - ECG-JEPA encoder for time-series data
2. **Linear Projections**: Map modality embeddings to a common token space
3. **LLM Backbone**: Processes interleaved multimodal tokens
4. **Output Generation**: Produces reasoning traces, bounding boxes, and diagnoses

## Training

### Two-Stage Training Process

**Stage 1: Modality Alignment**
- Aligns the ECG encoder, projection layers, and LLM
- Uses DRPO training for high-quality reasoning from the start

**Stage 2: Multimodal Fine-tuning with DRPO**
- Full multimodal corpus training
- Balances learning across different clinical domains and difficulty levels

### Reward Design

Training combines multiple reward signals:
- **Accuracy Reward**: F1 score between predictions and ground truth
- **Semantic Alignment Reward**: IoU between predicted bounding boxes and ground truth segmentations
- **Similarity Reward**: Jaccard similarity for open-ended responses

## Results Highlights

- **Overall F1 of 0.349** across 8 medical imaging domains (14.7pp improvement over best baseline)
- **Strong transfer learning**: Zero-shot QoQ-Med outperforms base Qwen3 on MIMIC-IV tasks
- **Best combined performance**: CLIMB pretraining + MIMIC-IV fine-tuning achieves highest Jaccard similarity (0.207) and F1 (0.471)
- **Interpretable outputs**: Bounding box IoU on par with GPT-o4-mini

## Usage

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "ddvd233/QoQ-Med3-VL-8B",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("ddvd233/QoQ-Med3-VL-8B")

# Example inference
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/chest_xray.png"},
            {"type": "text", "text": "What is the diagnosis of the patient in this X-ray image?"}
        ]
    }
]

inputs = processor(messages, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

## License

Please refer to individual model cards on HuggingFace for license information.

## Acknowledgments

This work was conducted at MIT Media Lab in collaboration with researchers from Harvard Medical School, Johns Hopkins University, and other institutions.

## Contact

For questions or collaborations, please contact: dvdai@mit.edu