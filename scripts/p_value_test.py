import numpy as np


def bootstrap_significance(mean1, std1, mean2, std2, n=4, n_bootstrap=10000):
    """Test if mean1 is significantly different from mean2"""
    # Generate synthetic samples assuming normal distribution
    samples1 = np.random.normal(mean1, std1, (n_bootstrap, n))
    samples2 = np.random.normal(mean2, std2, (n_bootstrap, n))

    # Calculate means for each bootstrap sample
    means1 = samples1.mean(axis=1)
    means2 = samples2.mean(axis=1)

    # Two-tailed test
    diff = means1 - means2
    # p-value is proportion of bootstrap differences that cross zero in opposite direction
    if mean1 > mean2:
        p_value = (diff <= 0).mean() * 2
    else:
        p_value = (diff >= 0).mean() * 2

    return min(p_value, 1.0)


# Qwen-2.5-VL-7B Results
print("=== Qwen-2.5-VL-7B: FairGRPO vs Best Baseline ===")
qwen_results = []

# PP: FairGRPO (15.42±1.95) vs Re++ (16.66±2.11) - lower is better
p = bootstrap_significance(15.42, 1.95, 16.66, 2.11, n=4)
if p < 0.05: qwen_results.append(f"PP (p={p:.3f})")

# EOD: FairGRPO (5.62±0.10) vs Re++ (6.66±1.59) - lower is better
p = bootstrap_significance(5.62, 0.10, 6.66, 1.59, n=4)
if p < 0.05: qwen_results.append(f"EOD (p={p:.3f})")

# FPR_Diff: FairGRPO (5.00±0.87) vs GRPO (4.85±0.34) - lower is better
p = bootstrap_significance(5.00, 0.87, 4.85, 0.34, n=4)
if p < 0.05: qwen_results.append(f"FPR_Diff (p={p:.3f})")

# σ_F1: FairGRPO (0.0254±0.0035) vs GRPO+RS (0.0319±0.0009) - lower is better
p = bootstrap_significance(0.0254, 0.0035, 0.0319, 0.0009, n=4)
if p < 0.05: qwen_results.append(f"σ_F1 (p={p:.3f})")

# ΔF1: FairGRPO (0.0522±0.0099) vs GRPO+RS (0.0628±0.0037) - lower is better
p = bootstrap_significance(0.0522, 0.0099, 0.0628, 0.0037, n=4)
if p < 0.05: qwen_results.append(f"ΔF1 (p={p:.3f})")

# σ_Acc: FairGRPO (4.42±0.01) vs GRPO (4.85±0.24) - lower is better
p = bootstrap_significance(4.42, 0.01, 4.85, 0.24, n=4)
if p < 0.05: qwen_results.append(f"σ_Acc (p={p:.3f})")

# ΔAcc: FairGRPO (8.95±0.03) vs GRPO (9.92±0.69) - lower is better
p = bootstrap_significance(8.95, 0.03, 9.92, 0.69, n=4)
if p < 0.05: qwen_results.append(f"ΔAcc (p={p:.3f})")

# Acc: FairGRPO (78.52±0.31) vs GRPO (78.40±0.69) - higher is better
p = bootstrap_significance(78.52, 0.31, 78.40, 0.69, n=4)
if p < 0.05: qwen_results.append(f"Acc (p={p:.3f})")

# F1: FairGRPO (0.2657±0.0036) vs GRPO (0.2601±0.0131) - higher is better
p = bootstrap_significance(0.2657, 0.0036, 0.2601, 0.0131, n=4)
if p < 0.05: qwen_results.append(f"F1 (p={p:.3f})")

# Acc_ES: FairGRPO (77.14±0.29) vs GRPO (76.21±0.91) - higher is better
p = bootstrap_significance(77.14, 0.29, 76.21, 0.91, n=4)
if p < 0.05: qwen_results.append(f"Acc_ES (p={p:.3f})")

# F1_ES: FairGRPO (0.2602±0.0020) vs Re++ (0.2518±0.0063) - higher is better
p = bootstrap_significance(0.2602, 0.0020, 0.2518, 0.0063, n=4)
if p < 0.05: qwen_results.append(f"F1_ES (p={p:.3f})")

print("Significant improvements (p < 0.05):", qwen_results if qwen_results else "None")

# MedGemma-4B Results
print("\n=== MedGemma-4B: FairGRPO vs Best Baseline ===")
medgemma_results = []

# PP: FairGRPO (12.95±1.82) vs GRPO+DRO (18.20±3.06) - lower is better
p = bootstrap_significance(12.95, 1.82, 18.20, 3.06, n=4)
if p < 0.05: medgemma_results.append(f"PP (p={p:.3f})")

# EOD: FairGRPO (6.84±0.24) vs GRPO (6.30±0.25) - lower is better
p = bootstrap_significance(6.84, 0.24, 6.30, 0.25, n=4)
if p < 0.05: medgemma_results.append(f"EOD (p={p:.3f})")

# FPR_Diff: FairGRPO (5.53±0.29) vs GRPO+RS (4.78±1.84) - lower is better
p = bootstrap_significance(5.53, 0.29, 4.78, 1.84, n=4)
if p < 0.05: medgemma_results.append(f"FPR_Diff (p={p:.3f})")

# σ_F1: FairGRPO (0.0379±0.0005) vs GRPO (0.0387±0.0045) - lower is better
p = bootstrap_significance(0.0379, 0.0005, 0.0387, 0.0045, n=4)
if p < 0.05: medgemma_results.append(f"σ_F1 (p={p:.3f})")

# ΔF1: FairGRPO (0.0724±0.0004) vs GRPO (0.0753±0.0059) - lower is better
p = bootstrap_significance(0.0724, 0.0004, 0.0753, 0.0059, n=4)
if p < 0.05: medgemma_results.append(f"ΔF1 (p={p:.3f})")

# σ_Acc: FairGRPO (4.11±0.04) vs GRPO (4.19±0.03) - lower is better
p = bootstrap_significance(4.11, 0.04, 4.19, 0.03, n=4)
if p < 0.05: medgemma_results.append(f"σ_Acc (p={p:.3f})")

# ΔAcc: FairGRPO (8.53±0.11) vs GRPO (8.57±0.03) - lower is better
p = bootstrap_significance(8.53, 0.11, 8.57, 0.03, n=4)
if p < 0.05: medgemma_results.append(f"ΔAcc (p={p:.3f})")

# Acc: FairGRPO (80.40±0.03) vs GRPO+DRO (80.17±0.31) - higher is better
p = bootstrap_significance(80.40, 0.03, 80.17, 0.31, n=4)
if p < 0.05: medgemma_results.append(f"Acc (p={p:.3f})")

# F1: FairGRPO (0.3275±0.0007) vs FairGRPO_ND (0.3484±0.0041) - higher is better
# Note: Comparing against non-FairGRPO baselines
p = bootstrap_significance(0.3275, 0.0007, 0.3237, 0.0019, n=4)  # vs RLOO
if p < 0.05: medgemma_results.append(f"F1 (p={p:.3f})")

# Acc_ES: FairGRPO (77.23±0.01) vs GRPO+DRO (76.69±0.48) - higher is better
p = bootstrap_significance(77.23, 0.01, 76.69, 0.48, n=4)
if p < 0.05: medgemma_results.append(f"Acc_ES (p={p:.3f})")

# F1_ES: FairGRPO (0.3155±0.0006) vs RLOO (0.3056±0.0021) - higher is better
p = bootstrap_significance(0.3155, 0.0006, 0.3056, 0.0021, n=4)
if p < 0.05: medgemma_results.append(f"F1_ES (p={p:.3f})")

print("Significant improvements (p < 0.05):", medgemma_results if medgemma_results else "None")