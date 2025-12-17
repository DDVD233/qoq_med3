#!/usr/bin/env python3
"""
Balance demographic groups in JSONL file by oversampling minority groups.
Each group within each dataset will have the same number of samples.
"""

import json
from collections import defaultdict
from pathlib import Path
import random


def process_demographic_info(demographic_info: str) -> str:
    """
    Process demographic information string to groups, separated by commas.
    Example input: "demo": "sex: Male, age: 68"
    Example output: "M,A3"
    Age is grouped into ranges: 0-25 (A1), 26-50 (A2), 51-75 (A3), 76+ (A4).
    """
    if not demographic_info:
        return "UNK"
    
    groups = []
    items = demographic_info.split(",")
    
    for item in items:
        if ":" not in item:
            continue
            
        key, value = item.split(":", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        
        if key in ['sex', 'gender']:
            if value.startswith('m'):
                groups.append('M')
            elif value.startswith('f'):
                groups.append('F')
            else:
                groups.append('UNK')
        elif key == 'age':
            try:
                # Handle 'nan' string
                if value == 'nan':
                    groups.append("UNK")
                    continue
                    
                age = float(value)
                if age <= 25:
                    groups.append("A1")
                elif age <= 50:
                    groups.append("A2")
                elif age <= 75:
                    groups.append("A3")
                else:
                    groups.append("A4")
            except (ValueError, TypeError):
                groups.append("UNK")
    
    return ",".join(groups) if groups else "UNK"


def get_dataset_from_sample(sample):
    """Extract dataset identifier from sample based on image/video paths."""
    # Check for specific prompts that indicate dataset
    prompt_str = sample.get('problem', '')
    
    if 'How long will the patient stay in the hospital?' in prompt_str:
        return "los_prediction"
    elif 'Will the patient survive for at least 48 hours?' in prompt_str:
        return "48_ihm"
    
    # Check image paths
    if 'images' in sample and sample['images']:
        vision_path = sample['images'][0]
        try:
            parts = vision_path.split("/")
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}"
        except:
            pass
    
    # Check video paths
    if 'videos' in sample and sample['videos']:
        vision_path = sample['videos'][0]
        try:
            parts = vision_path.split("/")
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}"
        except:
            pass
    
    return "unknown"


def main():
    input_file = Path("/Users/dvd/PycharmProjects/verl/scripts/geom_train_demo_only.jsonl")
    output_file = Path("/Users/dvd/PycharmProjects/verl/scripts/geom_train_demo_only_balanced.jsonl")
    
    # Read all samples and group them
    samples_by_dataset_and_demo = defaultdict(lambda: defaultdict(list))
    
    print("Reading input file...")
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                demo_group = process_demographic_info(sample.get('demo', ''))
                dataset = get_dataset_from_sample(sample)
                samples_by_dataset_and_demo[dataset][demo_group].append(sample)
            except json.JSONDecodeError:
                print(f"Error parsing line {line_num}")
                continue
    
    # Print statistics before balancing
    print("\n=== Before Balancing ===")
    for dataset in sorted(samples_by_dataset_and_demo.keys()):
        print(f"\nDataset: {dataset}")
        demo_groups = samples_by_dataset_and_demo[dataset]
        for demo in sorted(demo_groups.keys()):
            count = len(demo_groups[demo])
            print(f"  {demo}: {count} samples")
    
    # Balance by oversampling within each dataset
    balanced_samples = []
    
    print("\n=== Balancing ===")
    for dataset in sorted(samples_by_dataset_and_demo.keys()):
        demo_groups = samples_by_dataset_and_demo[dataset]
        if not demo_groups:
            continue
            
        # Find the maximum count for this dataset
        max_count = max(len(samples) for samples in demo_groups.values())
        print(f"\nDataset: {dataset}, target size per group: {max_count}")
        
        # Oversample each demographic group to match max_count
        for demo, samples in demo_groups.items():
            current_count = len(samples)
            
            if current_count < max_count:
                # Need to oversample
                num_to_add = max_count - current_count
                oversampled = samples.copy()
                
                # Add copies until we reach the target
                while len(oversampled) < max_count:
                    samples_to_add = min(num_to_add, current_count)
                    oversampled.extend(random.sample(samples, min(samples_to_add, len(samples))))
                    num_to_add = max_count - len(oversampled)
                
                # Trim to exact size if we overshot
                oversampled = oversampled[:max_count]
                print(f"  {demo}: {current_count} -> {len(oversampled)} (oversampled by {len(oversampled) - current_count})")
                balanced_samples.extend(oversampled)
            else:
                print(f"  {demo}: {current_count} (no change needed)")
                balanced_samples.extend(samples)
    
    # Shuffle the final dataset
    random.shuffle(balanced_samples)
    
    # Write balanced dataset
    print(f"\nWriting {len(balanced_samples)} samples to {output_file}")
    with open(output_file, 'w') as f:
        for sample in balanced_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Print final statistics
    print("\n=== After Balancing ===")
    final_stats = defaultdict(lambda: defaultdict(int))
    for sample in balanced_samples:
        demo_group = process_demographic_info(sample.get('demo', ''))
        dataset = get_dataset_from_sample(sample)
        final_stats[dataset][demo_group] += 1
    
    for dataset in sorted(final_stats.keys()):
        print(f"\nDataset: {dataset}")
        demo_groups = final_stats[dataset]
        for demo in sorted(demo_groups.keys()):
            print(f"  {demo}: {demo_groups[demo]} samples")
    
    print(f"\nTotal samples: {len(balanced_samples)}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()