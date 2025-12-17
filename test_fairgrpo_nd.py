#!/usr/bin/env python3
"""
Unit test for FairGRPO_ND algorithm with synthetic data
"""

import torch
import numpy as np
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator


def test_fairgrpo_nd_with_synthetic_data():
    """Test FairGRPO_ND with synthetic data showing cluster-demographic alignment"""

    print("\n" + "="*80)
    print("Testing FairGRPO_ND Algorithm with Synthetic Data")
    print("="*80)

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create synthetic data with clear demographic patterns
    n_samples = 120

    # Create 3 domains and 3 hidden demographic groups
    domains = []
    demos = []

    # Domain distribution: 60% domain_0, 30% domain_1, 10% domain_2
    for i in range(n_samples):
        if i < 72:  # 60%
            domains.append("domain_0")
        elif i < 108:  # 30%
            domains.append("domain_1")
        else:  # 10%
            domains.append("domain_2")

    # Demographic distribution: 50% demo_0, 35% demo_1, 15% demo_2
    for i in range(n_samples):
        if i < 60:  # 50%
            demos.append("demo_0")
        elif i < 102:  # 35%
            demos.append("demo_1")
        else:  # 15%
            demos.append("demo_2")

    # Shuffle to mix domains and demos
    indices = np.random.permutation(n_samples)
    domains = np.array(domains)[indices]
    demos = np.array(demos)[indices]

    # Generate UIDs
    uids = np.array([f"sample_{i}" for i in range(n_samples)])

    # Create token-level rewards that correlate with demographics
    # Demo_0: high rewards (easy), Demo_1: medium, Demo_2: low rewards (hard)
    token_level_rewards = []
    response_masks = []

    for i in range(n_samples):
        seq_len = 20  # Fixed sequence length for simplicity

        # Base reward depends on demographic group
        if demos[i] == "demo_0":
            base_reward = 1.0  # Easy samples
        elif demos[i] == "demo_1":
            base_reward = 0.5  # Medium samples
        else:  # demo_2
            base_reward = -0.2  # Hard samples

        # Add domain effect
        if domains[i] == "domain_0":
            domain_multiplier = 1.0
        elif domains[i] == "domain_1":
            domain_multiplier = 0.9
        else:  # domain_2
            domain_multiplier = 0.7

        # Generate rewards with noise
        rewards = torch.randn(seq_len) * 0.2 + (base_reward * domain_multiplier)
        mask = torch.ones(seq_len, dtype=torch.bool)

        token_level_rewards.append(rewards)
        response_masks.append(mask)

    token_level_rewards = torch.stack(token_level_rewards)
    response_mask = torch.stack(response_masks)

    print(f"\nData Statistics:")
    print(f"  Total samples: {n_samples}")
    print(f"  Sequence length: 20 tokens")
    print(f"  Domains: {np.unique(domains)} ")
    print(f"  Demographics (hidden): {np.unique(demos)}")

    # Print distribution
    print(f"\nDomain Distribution:")
    for d in np.unique(domains):
        count = (domains == d).sum()
        print(f"  {d}: {count} samples ({count/n_samples*100:.1f}%)")

    print(f"\nDemographic Distribution (ground truth for analysis):")
    for d in np.unique(demos):
        count = (demos == d).sum()
        print(f"  {d}: {count} samples ({count/n_samples*100:.1f}%)")

    # Run FairGRPO_ND
    print("\n" + "-"*40)
    print("Running FairGRPO_ND Algorithm")
    print("-"*40)
    print("Note: Algorithm treats all samples as UNK but analyzes alignment with ground truth")

    advantages, returns = core_algos.compute_fair_grpo_nd_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=uids,
        domain_info=domains,
        demo_info=demos,  # For analysis only - algorithm doesn't use this
        epsilon=1e-8
    )

    print(f"\nOutput Statistics:")
    print(f"  Advantages shape: {advantages.shape}")
    print(f"  Advantages mean: {advantages.mean():.4f}")
    print(f"  Advantages std: {advantages.std():.4f}")
    print(f"  Advantages min: {advantages.min():.4f}")
    print(f"  Advantages max: {advantages.max():.4f}")

    # Analyze advantages by ground truth demographics
    print(f"\nAdvantage Analysis by Ground Truth Demographics:")
    for demo in np.unique(demos):
        demo_mask = demos == demo
        demo_advs = advantages[demo_mask]
        n = demo_mask.sum()
        print(f"  {demo} (n={n}):")
        print(f"    Mean: {demo_advs.mean():.4f}")
        print(f"    Std:  {demo_advs.std():.4f}")
        print(f"    Positive: {(demo_advs > 0).float().mean():.1%}")

    # Analyze advantages by domain
    print(f"\nAdvantage Analysis by Domain:")
    for domain in np.unique(domains):
        domain_mask = domains == domain
        domain_advs = advantages[domain_mask]
        n = domain_mask.sum()
        print(f"  {domain} (n={n}):")
        print(f"    Mean: {domain_advs.mean():.4f}")
        print(f"    Std:  {domain_advs.std():.4f}")
        print(f"    Positive: {(domain_advs > 0).float().mean():.1%}")

    # Check which samples got upscaled
    print(f"\nUpscaling Analysis:")
    print("(Note: The metrics above already include detailed upscaling analysis)")

    # Verify the algorithm runs without errors
    assert advantages.shape == (n_samples,), f"Unexpected advantages shape: {advantages.shape}"
    assert returns.shape == (n_samples,), f"Unexpected returns shape: {returns.shape}"
    assert not torch.isnan(advantages).any(), "NaN values in advantages"
    assert not torch.isinf(advantages).any(), "Inf values in advantages"

    print("\n" + "="*80)
    print("✓ Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    test_fairgrpo_nd_with_synthetic_data()