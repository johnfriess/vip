#!/usr/bin/env python
"""
Evaluate Robomimic Policy Success Rate.

This script loads a trained robomimic policy and evaluates its success rate
in the robosuite environment.

Usage:
    python evaluate_policy.py \
        --policy /path/to/policy.pt \
        --dataset /path/to/dataset.hdf5 \
        --num_episodes 100

    # With video recording:
    python evaluate_policy.py \
        --policy /path/to/policy.pt \
        --dataset /path/to/dataset.hdf5 \
        --num_episodes 50 \
        --video_dir /path/to/videos
"""

import argparse
import h5py
import json
import numpy as np
import torch
from tqdm import tqdm
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def recover_environment(hdf5_path, render_mode=None):
    """Recover robosuite environment from HDF5 metadata."""
    import robosuite as suite

    with h5py.File(hdf5_path, 'r') as f:
        env_args = json.loads(f['data'].attrs['env_args'])

    env_name = env_args.get('env_name', env_args.get('env_type', 'Lift'))
    env_kwargs = env_args.get('env_kwargs', {})

    # Set render options
    env_kwargs['has_renderer'] = render_mode == 'human'
    env_kwargs['has_offscreen_renderer'] = render_mode == 'rgb_array'
    env_kwargs['use_camera_obs'] = False

    env = suite.make(env_name, **env_kwargs)
    print(f"Recovered environment: {env_name}")

    return env, env_name


def load_policy(policy_path, device="cuda"):
    """Load a trained robomimic policy."""
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils

    # Load policy checkpoint
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=policy_path, device=device)
    policy.start_episode()

    print(f"Loaded policy from {policy_path}")
    return policy


def get_observation(env, obs_keys=None):
    """Get observation from environment."""
    obs_dict = env._get_observations()

    if obs_keys is None:
        obs_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]

    obs_arrays = []
    for key in obs_keys:
        if key in obs_dict:
            arr = np.array(obs_dict[key]).flatten()
            obs_arrays.append(arr)

    if not obs_arrays:
        # Use all low-dim observations
        for key, val in obs_dict.items():
            if isinstance(val, np.ndarray) and val.ndim <= 1:
                obs_arrays.append(val.flatten())

    return np.concatenate(obs_arrays).astype(np.float32)


def run_episode(env, policy, obs_keys=None, max_steps=500, render=False):
    """Run a single episode and return success status."""
    env.reset()
    policy.start_episode()

    total_reward = 0.0
    success = False

    for step in range(max_steps):
        # Get observation
        obs = get_observation(env, obs_keys)

        # Get action from policy
        obs_dict = {"flat": obs}  # Simple flat observation
        action = policy(obs_dict)

        # Step environment
        obs_dict_env, reward, done, info = env.step(action)
        total_reward += reward

        if render:
            env.render()

        # Check success
        if env._check_success():
            success = True
            break

        if done:
            break

    return success, total_reward, step + 1


def evaluate_policy(
    policy_path,
    dataset_path,
    num_episodes=100,
    obs_keys=None,
    max_steps=500,
    device="cuda",
    render=False,
    video_dir=None,
    seed=0,
):
    """
    Evaluate policy success rate.

    Args:
        policy_path: Path to trained robomimic policy checkpoint
        dataset_path: Path to robomimic HDF5 (for environment recovery)
        num_episodes: Number of evaluation episodes
        obs_keys: Observation keys to use
        max_steps: Maximum steps per episode
        device: torch device
        render: Whether to render during evaluation
        video_dir: Directory to save videos (if any)
        seed: Random seed

    Returns:
        Dictionary with evaluation results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Recover environment
    render_mode = 'human' if render else None
    env, env_name = recover_environment(dataset_path, render_mode)

    # Load policy
    policy = load_policy(policy_path, device)

    # Run evaluation episodes
    successes = []
    rewards = []
    lengths = []

    print(f"\nEvaluating policy over {num_episodes} episodes...")
    for ep in tqdm(range(num_episodes)):
        success, reward, length = run_episode(
            env, policy, obs_keys, max_steps, render
        )
        successes.append(success)
        rewards.append(reward)
        lengths.append(length)

    # Compute metrics
    success_rate = np.mean(successes)
    avg_reward = np.mean(rewards)
    avg_length = np.mean(lengths)

    results = {
        "policy_path": policy_path,
        "dataset_path": dataset_path,
        "env_name": env_name,
        "num_episodes": num_episodes,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "successes": successes,
        "rewards": rewards,
        "lengths": lengths,
    }

    print(f"\n{'='*50}")
    print(f"Evaluation Results: {env_name}")
    print(f"{'='*50}")
    print(f"  Success Rate: {success_rate:.1%} ({sum(successes)}/{num_episodes})")
    print(f"  Avg Reward:   {avg_reward:.2f}")
    print(f"  Avg Length:   {avg_length:.1f}")
    print(f"{'='*50}")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate robomimic policy success rate")
    parser.add_argument("--policy", "-p", required=True, help="Path to policy checkpoint")
    parser.add_argument("--dataset", "-d", required=True, help="Path to robomimic HDF5")
    parser.add_argument("--num_episodes", "-n", type=int, default=100, help="Number of episodes")
    parser.add_argument("--obs_keys", nargs="+", default=None, help="Observation keys")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    parser.add_argument("--video_dir", default=None, help="Directory for videos")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file for results")

    args = parser.parse_args()

    results = evaluate_policy(
        policy_path=args.policy,
        dataset_path=args.dataset,
        num_episodes=args.num_episodes,
        obs_keys=args.obs_keys,
        max_steps=args.max_steps,
        device=args.device,
        render=args.render,
        video_dir=args.video_dir,
        seed=args.seed,
    )

    if args.output:
        # Save results
        import json
        with open(args.output, 'w') as f:
            # Convert numpy arrays for JSON
            results_json = {
                k: v if not isinstance(v, (list, np.ndarray)) or k in ['successes', 'rewards', 'lengths']
                else [float(x) for x in v]
                for k, v in results.items()
            }
            results_json['successes'] = [bool(s) for s in results['successes']]
            results_json['rewards'] = [float(r) for r in results['rewards']]
            results_json['lengths'] = [int(l) for l in results['lengths']]
            json.dump(results_json, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
