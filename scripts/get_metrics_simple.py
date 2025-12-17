import wandb
import numpy as np
from collections import defaultdict


api = wandb.Api()  # 1. connect
run = api.run("ddavid233/verl_mimic/9g9a2tgq")  # 2. locate the run

# Define specific steps to sample (e.g., 35 and 40)
steps_to_sample = [0]

# Collect metrics across specified steps
step_data = defaultdict(list)

# Collect data from specified steps
print(f"Collecting data from steps: {steps_to_sample}")
for step in steps_to_sample:
    for record in run.scan_history(min_step=step, max_step=step+1):
        if record['_step'] == step:
            # Filter and process record
            filtered_record = {k: v for k, v in record.items() if '-aux' not in k}
            filtered_record['val/similarity'] = (filtered_record['mimic_qa/qa_type_1/similarity'] + filtered_record['mimic_qa/qa_type_2/similarity']) / 2
            filtered_record['val/accuracy'] = (filtered_record['mimic_qa/qa_type_3/accuracy'] + filtered_record['mimic_qa/qa_type_5/accuracy'] + filtered_record['mimic_qa/qa_type_6/accuracy']) / 3
            filtered_record['val/f1'] = (filtered_record['mimic_qa/qa_type_3/f1'] + filtered_record['mimic_qa/qa_type_5/f1'] + filtered_record['mimic_qa/qa_type_6/f1']) / 3
            # Collect values for each metric across steps
            for key, value in filtered_record.items():
                if isinstance(value, (int, float)):
                    step_data[key].append(value)
            # record['overall/overall_acc'] = record['val/accuracy']
            # record['overall/overall_f1'] = record['val/f1']
            #
            # filtered_record['overall/acc_es'] = record['overall/overall_acc'] / (1 + record['overall/overall_acc_std'])
            # filtered_record['overall/f1_es'] = record['overall/overall_f1'] / (1 + record['overall/overall_f1_std'])

            print(f"Step {step} data collected")
            break  # Only take the first record at this step

# Calculate mean and standard deviation for each metric
final_results = {}

for metric_name, values in step_data.items():
    # Calculate mean
    final_results[metric_name] = np.mean(values)

# Filter to only overall metrics
final_results = {k: v for k, v in final_results.items()}

print(final_results)
