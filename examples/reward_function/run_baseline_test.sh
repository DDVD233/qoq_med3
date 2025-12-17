#!/bin/bash

# Example script to run baseline model testing

# Test with OpenAI GPT-4
python test_closed_models.py \
  --model-type openai \
  --model-name gpt-4-vision-preview \
  --api-key "$OPENAI_API_KEY" \
  --dataset geom_valid.jsonl \
  --image-base-path /path/to/image/base \
  --output-dir outputs/gpt4_baseline \
  --num-examples 100  # Start with small number for testing

# Test with Anthropic Claude
python test_closed_models.py \
  --model-type anthropic \
  --model-name claude-3-opus-20240229 \
  --api-key "$ANTHROPIC_API_KEY" \
  --dataset path/to/your/dataset.jsonl \
  --image-base-path /path/to/image/base \
  --output-dir outputs/claude_baseline \
  --num-examples 100

# Run on full dataset (remove --num-examples to run on all)
python test_closed_models.py \
  --model-type openai \
  --model-name gpt-4-vision-preview \
  --api-key "$OPENAI_API_KEY" \
  --dataset path/to/your/dataset.jsonl \
  --image-base-path /path/to/image/base \
  --output-dir outputs/gpt4_full