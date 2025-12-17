#!/usr/bin/env python3
"""
Test script for FairGRPO_ND algorithm
This algorithm treats all demographics as unknown but analyzes cluster alignment with ground truth
"""

import torch
import numpy as np
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator
import wandb


def generate_test_data(n_samples=100, n_domains=3, n_demos=4, seed=42):
    """Generate synthetic test data with known demographic and domain patterns"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate UIDs
    uids = np.array([f"sample_{i}" for i in range(n_samples)])

    # Assign domains (uneven distribution)
    domain_probs = [0.5, 0.3, 0.2]  # Domain 0 is majority
    domains = np.random.choice([f"domain_{i}" for i in range(n_domains)],
                              size=n_samples, p=domain_probs)

    # Assign demographics (uneven distribution)
    demo_probs = [0.4, 0.3, 0.2, 0.1]  # Demo 0 is majority, Demo 3 is minority
    demos = np.random.choice([f"demo_{i}" for i in range(n_demos)],
                            size=n_samples, p=demo_probs)

    # Generate rewards that correlate with demographics
    # Minority demos tend to get lower rewards (harder samples)
    token_level_rewards = []
    response_masks = []

    for i in range(n_samples):
        seq_len = np.random.randint(10, 50)

        # Base reward depends on demo group (minority demos get lower rewards)
        demo_idx = int(demos[i].split("_")[1])
        base_reward = 1.0 - (demo_idx * 0.2)  # Demo 0: 1.0, Demo 3: 0.4

        # Add domain effect
        domain_idx = int(domains[i].split("_")[1])
        domain_effect = 1.0 - (domain_idx * 0.1)  # Domain 0: 1.0, Domain 2: 0.8

        # Generate token rewards with some noise
        rewards = torch.randn(seq_len) * 0.3 + (base_reward * domain_effect)
        mask = torch.ones(seq_len, dtype=torch.bool)

        # Pad to fixed length
        max_len = 50
        padded_rewards = torch.zeros(max_len)
        padded_mask = torch.zeros(max_len, dtype=torch.bool)
        padded_rewards[:seq_len] = rewards
        padded_mask[:seq_len] = mask

        token_level_rewards.append(padded_rewards)
        response_masks.append(padded_mask)

    token_level_rewards = torch.stack(token_level_rewards)
    response_mask = torch.stack(response_masks)

    return {
        "token_level_rewards": token_level_rewards,
        "response_mask": response_mask,
        "index": uids,
        "domain_info": domains,
        "demo_info": demos
    }


def test_fairgrpo_nd():
    """Test the FairGRPO_ND algorithm"""

    print("=" * 80)
    print("Testing FairGRPO_ND Algorithm")
    print("=" * 80)

    # Initialize wandb (optional)
    try:
        wandb.init(project="fairgrpo_nd_test", name="test_run", mode="offline")
        print("WandB initialized in offline mode")
    except:
        print("WandB not available, will only print metrics")

    # Generate test data
    print("\n1. Generating test data...")
    data = generate_test_data(n_samples=200, n_domains=3, n_demos=4)
    print(f"   - Samples: {len(data['index'])}")
    print(f"   - Unique domains: {len(set(data['domain_info']))}")
    print(f"   - Unique demographics: {len(set(data['demo_info']))}")

    # Print data distribution
    print("\n2. Data distribution:")
    unique_domains, domain_counts = np.unique(data['domain_info'], return_counts=True)
    for d, c in zip(unique_domains, domain_counts):
        print(f"   - {d}: {c} samples ({c/len(data['index'])*100:.1f}%)")

    unique_demos, demo_counts = np.unique(data['demo_info'], return_counts=True)
    for d, c in zip(unique_demos, demo_counts):
        print(f"   - {d}: {c} samples ({c/len(data['index'])*100:.1f}%)")

    # Run FairGRPO_ND
    print("\n3. Running FairGRPO_ND algorithm...")
    print("   (Treating all samples as UNK, but analyzing alignment with ground truth)")

    advantages, returns = core_algos.compute_fair_grpo_nd_outcome_advantage(
        token_level_rewards=data["token_level_rewards"],
        response_mask=data["response_mask"],
        index=data["index"],
        domain_info=data["domain_info"],
        demo_info=data["demo_info"],  # For analysis only
        epsilon=1e-8
    )

    print("\n4. Results:")
    print(f"   - Advantages shape: {advantages.shape}")
    print(f"   - Advantages mean: {advantages.mean():.4f}")
    print(f"   - Advantages std: {advantages.std():.4f}")
    print(f"   - Returns shape: {returns.shape}")
    print(f"   - Returns mean: {returns.mean():.4f}")

    # Analyze advantages by ground truth demographics
    print("\n5. Advantage analysis by ground truth demographics:")
    for demo in unique_demos:
        demo_mask = data['demo_info'] == demo
        demo_advs = advantages[demo_mask]
        print(f"   - {demo}: mean={demo_advs.mean():.4f}, std={demo_advs.std():.4f}, "
              f"positive={(demo_advs > 0).float().mean():.2%}")

    print("\n6. Advantage analysis by domain:")
    for domain in unique_domains:
        domain_mask = data['domain_info'] == domain
        domain_advs = advantages[domain_mask]
        print(f"   - {domain}: mean={domain_advs.mean():.4f}, std={domain_advs.std():.4f}, "
              f"positive={(domain_advs > 0).float().mean():.2%}")

    # Test with standard FairGRPO for comparison
    print("\n7. Comparison with standard FairGRPO (using actual demographics):")

    # Create demo_info_dict for standard FairGRPO
    demo_info_dict = {uid: demo for uid, demo in zip(data['index'], data['demo_info'])}

    fair_advantages, fair_returns = core_algos.compute_fair_grpo_outcome_advantage(
        token_level_rewards=data["token_level_rewards"],
        response_mask=data["response_mask"],
        index=data["index"],
        domain_info=data["domain_info"],
        demo_info_dict=demo_info_dict,
        epsilon=1e-8
    )

    print(f"   - Standard FairGRPO advantages mean: {fair_advantages.mean():.4f}")
    print(f"   - Standard FairGRPO advantages std: {fair_advantages.std():.4f}")

    # Compare the two
    diff = (advantages - fair_advantages).abs().mean()
    print(f"   - Mean absolute difference: {diff:.4f}")

    # Finish wandb run
    try:
        wandb.finish()
        print("\n8. WandB run finished")
    except:
        pass

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_fairgrpo_nd()