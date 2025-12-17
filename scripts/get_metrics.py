import wandb
import numpy as np
from collections import defaultdict


api = wandb.Api()  # 1. connect
run = api.run("ddavid233/verl_climb/tpqowbsw")  # 2. locate the run

# Define specific steps to sample (e.g., 35 and 40)
steps_to_sample = [60,120]

# Collect metrics across specified steps
step_data = defaultdict(list)

# Collect data from specified steps
print(f"Collecting data from steps: {steps_to_sample}")
for step in steps_to_sample:
    for record in run.scan_history(min_step=step, max_step=step+1):
        if record['_step'] == step:
            # Filter and process record
            filtered_record = {k: v for k, v in record.items() if '-aux' not in k}

            # Remove unwanted fields
            # if 'val/f1' in filtered_record:
            #     filtered_record.pop('val/f1')
            # if 'val/accuracy' in filtered_record:
            #     filtered_record.pop('val/accuracy')
            # record['overall/overall_acc'] = record['val/accuracy']
            # record['overall/overall_f1'] = record['val/f1']

            # Calculate derived metrics
            # filtered_record['overall/acc_es'] = record['overall/overall_acc'] / (1 + record['overall/overall_acc_std'])
            # filtered_record['overall/f1_es'] = record['overall/overall_f1'] / (1 + record['overall/overall_f1_std'])

            # Collect values for each metric across steps
            for key, value in filtered_record.items():
                if isinstance(value, (int, float)):
                    step_data[key].append(value)

            print(f"Step {step} data collected")
            break  # Only take the first record at this step

# Calculate mean and standard deviation for each metric
final_results = {}

for metric_name, values in step_data.items():
    # Calculate mean
    final_results[metric_name] = np.mean(values)
    if 'std_acc' in metric_name:
        print(metric_name)

    # Calculate std using /std convention to avoid overwriting existing _std metrics
    if len(values) > 1:
        final_results[f"{metric_name}/std"] = np.std(values, ddof=1)  # Sample std with ddof=1
    else:
        final_results[f"{metric_name}/std"] = 0.0  # If only one value, std is 0

# Filter to only overall metrics
final_results = {k: v for k, v in final_results.items() if
                 ('val-aux' not in k and (k.count('/') == 1 or "std" in k))}

# Print aggregated results
print("\n=== Aggregated Results (Mean ± Std) ===")
metric: str
for metric in sorted(set(k for k in final_results.keys())):
    if metric in final_results and not metric.endswith('/std'):  # Check if base metric exists
        mean_val = final_results[metric]
        std_val = final_results.get(f"{metric}/std", 0.0)
        print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")
