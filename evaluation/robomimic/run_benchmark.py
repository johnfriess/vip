#!/usr/bin/env python
"""
Value Function Benchmark on Robomimic.

This script runs a complete benchmark comparing VIP, DERAIL, and IQL value
functions on robomimic datasets. It:
1. Trains value functions on demo data
2. Precomputes rewards with each value function
3. Trains policies using robomimic's offline RL (IQL/TD3-BC)
4. Evaluates policy success rates
5. Generates comparison report

Usage:
    python run_benchmark.py \
        --dataset /path/to/lift_ph_low_dim.hdf5 \
        --output_dir ./benchmark_results \
        --value_functions vip derail iql \
        --policy_algo iql

    # Skip value function training (use existing checkpoints):
    python run_benchmark.py \
        --dataset /path/to/lift_ph_low_dim.hdf5 \
        --output_dir ./benchmark_results \
        --vip_checkpoint /path/to/vip.pt \
        --derail_checkpoint /path/to/derail.pt \
        --iql_checkpoint /path/to/iql.pt \
        --skip_value_training
"""

import argparse
import os
import sys
import json
import subprocess
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: Command failed with return code {result.returncode}")
    return result.returncode == 0


def train_value_function(
    dataset_path,
    output_dir,
    model_type,
    train_steps=100000,
    device="cuda",
):
    """Train a value function (VIP, DERAIL, or IQL) on robomimic data."""
    config_map = {
        "vip": "config_vip_robomimic",
        "derail": "config_derail_robomimic",
        "iql": "config_iql_robomimic",
    }

    config_name = config_map[model_type]
    checkpoint_dir = os.path.join(output_dir, f"value_functions/{model_type}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Change to VIP directory for training
    vip_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'vip')

    cmd = [
        "python", "train_model.py",
        f"--config-name={config_name}",
        f"datapath={dataset_path}",
        f"train_steps={train_steps}",
        f"device={device}",
        f"hydra.run.dir={checkpoint_dir}",
    ]

    success = run_command(cmd, f"Training {model_type.upper()} value function")

    checkpoint_path = os.path.join(checkpoint_dir, "snapshot.pt")
    return checkpoint_path if os.path.exists(checkpoint_path) else None


def precompute_rewards(
    input_path,
    output_path,
    checkpoint_path,
    model_type,
    device="cuda",
):
    """Precompute value-function rewards for dataset."""
    script_path = os.path.join(os.path.dirname(__file__), "precompute_rewards.py")

    model_type_map = {
        "vip": "state_vip",
        "derail": "state_euclidean_derail",
        "iql": "state_iql",
    }

    cmd = [
        "python", script_path,
        "--input", input_path,
        "--output", output_path,
        "--checkpoint", checkpoint_path,
        "--model_type", model_type_map[model_type],
        "--device", device,
    ]

    success = run_command(cmd, f"Precomputing {model_type.upper()} rewards")
    return success


def train_policy(
    dataset_path,
    output_dir,
    policy_algo="iql",
    num_epochs=2000,
    device="cuda",
):
    """Train a policy using robomimic's offline RL."""
    # This requires robomimic to be installed
    # We'll call robomimic's training script

    checkpoint_dir = os.path.join(output_dir, "policy")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create a robomimic config for training
    config = {
        "algo_name": policy_algo,
        "experiment": {
            "name": f"policy_{policy_algo}",
        },
        "train": {
            "data": dataset_path,
            "output_dir": checkpoint_dir,
            "num_epochs": num_epochs,
            "batch_size": 100,
        },
    }

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    cmd = [
        "python", "-m", "robomimic.scripts.train",
        "--config", config_path,
    ]

    success = run_command(cmd, f"Training {policy_algo.upper()} policy")

    # Find checkpoint
    models_dir = os.path.join(checkpoint_dir, "models")
    if os.path.exists(models_dir):
        checkpoints = sorted([f for f in os.listdir(models_dir) if f.endswith('.pth')])
        if checkpoints:
            return os.path.join(models_dir, checkpoints[-1])

    return None


def evaluate_policy(
    policy_path,
    dataset_path,
    output_path,
    num_episodes=100,
    device="cuda",
):
    """Evaluate policy success rate."""
    script_path = os.path.join(os.path.dirname(__file__), "evaluate_policy.py")

    cmd = [
        "python", script_path,
        "--policy", policy_path,
        "--dataset", dataset_path,
        "--num_episodes", str(num_episodes),
        "--device", device,
        "--output", output_path,
    ]

    success = run_command(cmd, "Evaluating policy")
    return success


def run_benchmark(
    dataset_path,
    output_dir,
    value_functions=["vip", "derail", "iql"],
    policy_algo="iql",
    vip_checkpoint=None,
    derail_checkpoint=None,
    iql_checkpoint=None,
    skip_value_training=False,
    value_train_steps=100000,
    policy_epochs=2000,
    eval_episodes=100,
    device="cuda",
):
    """
    Run full benchmark comparing value functions.

    Args:
        dataset_path: Path to robomimic HDF5 dataset
        output_dir: Output directory for all results
        value_functions: List of value functions to benchmark
        policy_algo: Policy algorithm for robomimic training
        vip_checkpoint: Pre-trained VIP checkpoint (optional)
        derail_checkpoint: Pre-trained DERAIL checkpoint (optional)
        iql_checkpoint: Pre-trained IQL checkpoint (optional)
        skip_value_training: Skip value function training
        value_train_steps: Training steps for value functions
        policy_epochs: Training epochs for policies
        eval_episodes: Number of evaluation episodes
        device: torch device
    """
    os.makedirs(output_dir, exist_ok=True)

    # Store checkpoints
    checkpoints = {
        "vip": vip_checkpoint,
        "derail": derail_checkpoint,
        "iql": iql_checkpoint,
    }

    results = {
        "dataset": dataset_path,
        "policy_algo": policy_algo,
        "timestamp": datetime.now().isoformat(),
        "value_functions": {},
    }

    # Also benchmark with original rewards (baseline)
    all_conditions = ["original"] + value_functions

    for vf_name in all_conditions:
        print(f"\n{'#'*60}")
        print(f"# Benchmarking: {vf_name.upper()}")
        print(f"{'#'*60}")

        vf_output = os.path.join(output_dir, vf_name)
        os.makedirs(vf_output, exist_ok=True)

        if vf_name == "original":
            # Use original rewards (baseline)
            reward_dataset = dataset_path
        else:
            # Step 1: Train value function (if needed)
            if not skip_value_training and checkpoints.get(vf_name) is None:
                checkpoint = train_value_function(
                    dataset_path=dataset_path,
                    output_dir=vf_output,
                    model_type=vf_name,
                    train_steps=value_train_steps,
                    device=device,
                )
                checkpoints[vf_name] = checkpoint
            else:
                checkpoint = checkpoints.get(vf_name)

            if checkpoint is None:
                print(f"WARNING: No checkpoint for {vf_name}, skipping...")
                continue

            # Step 2: Precompute rewards
            reward_dataset = os.path.join(vf_output, f"dataset_{vf_name}_reward.hdf5")
            precompute_rewards(
                input_path=dataset_path,
                output_path=reward_dataset,
                checkpoint_path=checkpoint,
                model_type=vf_name,
                device=device,
            )

        # Step 3: Train policy
        policy_path = train_policy(
            dataset_path=reward_dataset,
            output_dir=vf_output,
            policy_algo=policy_algo,
            num_epochs=policy_epochs,
            device=device,
        )

        if policy_path is None:
            print(f"WARNING: Policy training failed for {vf_name}")
            continue

        # Step 4: Evaluate policy
        eval_output = os.path.join(vf_output, "evaluation.json")
        evaluate_policy(
            policy_path=policy_path,
            dataset_path=dataset_path,  # Use original dataset for env
            output_path=eval_output,
            num_episodes=eval_episodes,
            device=device,
        )

        # Load results
        if os.path.exists(eval_output):
            with open(eval_output) as f:
                eval_results = json.load(f)
            results["value_functions"][vf_name] = {
                "checkpoint": checkpoints.get(vf_name),
                "policy_path": policy_path,
                "success_rate": eval_results["success_rate"],
                "avg_reward": eval_results["avg_reward"],
            }

    # Generate comparison report
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print('='*60)
    print(f"Dataset: {dataset_path}")
    print(f"Policy Algorithm: {policy_algo}")
    print(f"Eval Episodes: {eval_episodes}")
    print('-'*60)
    print(f"{'Value Function':<20} {'Success Rate':<15} {'Avg Reward':<15}")
    print('-'*60)

    for vf_name, vf_results in results["value_functions"].items():
        sr = vf_results["success_rate"]
        ar = vf_results["avg_reward"]
        print(f"{vf_name:<20} {sr:.1%} ({sr*eval_episodes:.0f}/{eval_episodes}){'':<5} {ar:.2f}")

    print('='*60)

    # Save results
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Value function benchmark on robomimic")
    parser.add_argument("--dataset", "-d", required=True, help="Robomimic HDF5 dataset")
    parser.add_argument("--output_dir", "-o", default="./benchmark_results", help="Output directory")
    parser.add_argument("--value_functions", "-v", nargs="+", default=["vip", "derail", "iql"],
                        choices=["vip", "derail", "iql"], help="Value functions to benchmark")
    parser.add_argument("--policy_algo", "-p", default="iql", choices=["iql", "td3_bc", "bc"],
                        help="Policy algorithm")
    parser.add_argument("--vip_checkpoint", default=None, help="Pre-trained VIP checkpoint")
    parser.add_argument("--derail_checkpoint", default=None, help="Pre-trained DERAIL checkpoint")
    parser.add_argument("--iql_checkpoint", default=None, help="Pre-trained IQL checkpoint")
    parser.add_argument("--skip_value_training", action="store_true", help="Skip value function training")
    parser.add_argument("--value_train_steps", type=int, default=100000, help="Value function training steps")
    parser.add_argument("--policy_epochs", type=int, default=2000, help="Policy training epochs")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Evaluation episodes")
    parser.add_argument("--device", default="cuda", help="Device")

    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        value_functions=args.value_functions,
        policy_algo=args.policy_algo,
        vip_checkpoint=args.vip_checkpoint,
        derail_checkpoint=args.derail_checkpoint,
        iql_checkpoint=args.iql_checkpoint,
        skip_value_training=args.skip_value_training,
        value_train_steps=args.value_train_steps,
        policy_epochs=args.policy_epochs,
        eval_episodes=args.eval_episodes,
        device=args.device,
    )


if __name__ == "__main__":
    main()
