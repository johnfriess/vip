#!/usr/bin/env python
"""
Precompute Value-Function Rewards for Robomimic Datasets.

This script takes a robomimic HDF5 dataset and a trained value function model
(VIP, DERAIL, or IQL), computes rewards for each transition based on the
goal-conditioned value function, and saves a new HDF5 with these rewards.

The reward is the value function output:
- VIP/DERAIL: r(s, g) = -||phi(s) - phi(g)||_2
- IQL: r(s, g) = V(s || g)

Usage:
    python precompute_rewards.py \
        --input /path/to/dataset.hdf5 \
        --output /path/to/dataset_vip_reward.hdf5 \
        --checkpoint /path/to/vip_model.pt \
        --model_type state_vip
"""

import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
import shutil
import json
import sys
import os

# Add VIP to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Default observation keys for robomimic
DEFAULT_OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]


def load_value_function(checkpoint_path, model_type, device="cuda"):
    """Load a trained VIP/IQL/DERAIL model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_cfg = {k: v for k, v in checkpoint['model_cfg'].items() if not k.startswith('_')}

    if model_type == "state_vip":
        from vip.models.model_vip import StateVIP
        model = StateVIP(**model_cfg)
    elif model_type == "state_euclidean_derail":
        from vip.models.model_derail import StateEuclideanDERAIL
        model = StateEuclideanDERAIL(**model_cfg)
    elif model_type == "state_multilinear_derail":
        from vip.models.model_derail import StateMultilinearDERAIL
        model = StateMultilinearDERAIL(**model_cfg)
    elif model_type == "state_iql":
        from vip.models.model_iql import StateIQL
        model = StateIQL(**model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict({
        k.replace('module.', ''): v
        for k, v in checkpoint['model'].items()
    })
    model.eval()
    model.to(device)

    print(f"Loaded {model_type} from {checkpoint_path}")
    return model


def get_obs_from_demo(demo, timestep, obs_keys, use_states=False):
    """Get observation vector from demo at given timestep."""
    if use_states:
        return demo['states'][timestep].astype(np.float32)

    obs_group = demo['obs']
    obs_arrays = []
    for key in obs_keys:
        if key in obs_group:
            arr = obs_group[key][timestep]
            if arr.ndim > 1:
                arr = arr.flatten()
            obs_arrays.append(arr)

    if not obs_arrays:
        return demo['states'][timestep].astype(np.float32)

    return np.concatenate(obs_arrays).astype(np.float32)


def compute_value(model, model_type, state, goal, device="cuda"):
    """Compute value function output for state given goal."""
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        goal_tensor = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(device)

        if model_type == "state_multilinear_derail":
            h1, h2, h3 = model(state_tensor, goal_tensor)
            value = model.value(h1, h2, h3).cpu().numpy().squeeze()
        elif model_type == "state_iql":
            sg = torch.cat([state_tensor, goal_tensor], dim=-1)
            value = model.v_value(sg).cpu().numpy().squeeze()
        else:
            # VIP / Euclidean DERAIL: -||phi(s) - phi(g)||_2
            state_emb = model(state_tensor)
            goal_emb = model(goal_tensor)
            value = -torch.norm(state_emb - goal_emb, dim=-1).cpu().numpy().squeeze()

    return float(value)


def precompute_rewards(
    input_path,
    output_path,
    checkpoint_path,
    model_type="state_vip",
    obs_keys=None,
    use_states=False,
    device="cuda",
    goal_timestep=-1,
):
    """
    Precompute value-based rewards for all transitions in dataset.

    Args:
        input_path: Path to input robomimic HDF5
        output_path: Path to output HDF5 with value-based rewards
        checkpoint_path: Path to trained value function model
        model_type: Type of model (state_vip, state_euclidean_derail, etc.)
        obs_keys: Observation keys to use
        use_states: Use raw MuJoCo states instead of observations
        device: torch device
        goal_timestep: Which timestep to use as goal (-1 = last)
    """
    if obs_keys is None:
        obs_keys = DEFAULT_OBS_KEYS

    # Load model
    model = load_value_function(checkpoint_path, model_type, device)

    # Copy input file to output
    print(f"Copying {input_path} to {output_path}...")
    shutil.copy(input_path, output_path)

    # Open output file for modification
    with h5py.File(output_path, 'r+') as f:
        data = f['data']

        # Get demo keys
        demo_keys = [k for k in data.keys() if k.startswith('demo_')]
        demo_keys = sorted(demo_keys, key=lambda x: int(x.split('_')[1]))

        print(f"Processing {len(demo_keys)} demonstrations...")

        for demo_key in tqdm(demo_keys):
            demo = data[demo_key]

            # Get episode length
            if use_states:
                ep_length = len(demo['states'])
            else:
                first_key = obs_keys[0] if obs_keys[0] in demo['obs'] else list(demo['obs'].keys())[0]
                ep_length = len(demo['obs'][first_key])

            # Determine goal timestep
            if goal_timestep < 0:
                goal_idx = ep_length + goal_timestep
            else:
                goal_idx = min(goal_timestep, ep_length - 1)

            # Get goal observation
            goal_obs = get_obs_from_demo(demo, goal_idx, obs_keys, use_states)

            # Compute rewards for each transition
            num_transitions = len(demo['actions'])
            rewards = np.zeros(num_transitions, dtype=np.float32)

            for t in range(num_transitions):
                state_obs = get_obs_from_demo(demo, t, obs_keys, use_states)
                rewards[t] = compute_value(model, model_type, state_obs, goal_obs, device)

            # Backup original rewards if not already done
            if 'rewards_original' not in demo:
                if 'rewards' in demo:
                    original_rewards = demo['rewards'][:]
                    del demo['rewards']
                    demo.create_dataset('rewards_original', data=original_rewards)

            # Replace rewards
            if 'rewards' in demo:
                del demo['rewards']
            demo.create_dataset('rewards', data=rewards)

            # Also store value-based rewards separately
            reward_key = f'rewards_{model_type}'
            if reward_key in demo:
                del demo[reward_key]
            demo.create_dataset(reward_key, data=rewards)

        # Add metadata
        metadata = {
            'checkpoint_path': checkpoint_path,
            'model_type': model_type,
            'obs_keys': obs_keys,
            'use_states': use_states,
            'goal_timestep': goal_timestep,
        }
        data.attrs['value_function_info'] = json.dumps(metadata)

    print(f"\nDone! Value-based rewards saved to {output_path}")
    print(f"  - Original rewards backed up to 'rewards_original'")
    print(f"  - Value rewards stored in 'rewards' and 'rewards_{model_type}'")


def main():
    parser = argparse.ArgumentParser(description="Precompute value-function rewards for robomimic")
    parser.add_argument("--input", "-i", required=True, help="Input robomimic HDF5 path")
    parser.add_argument("--output", "-o", required=True, help="Output HDF5 path")
    parser.add_argument("--checkpoint", "-c", required=True, help="Trained value function checkpoint")
    parser.add_argument("--model_type", "-m", default="state_vip",
                        choices=["state_vip", "state_euclidean_derail", "state_multilinear_derail", "state_iql"],
                        help="Type of value function model")
    parser.add_argument("--obs_keys", nargs="+", default=None,
                        help="Observation keys to use")
    parser.add_argument("--use_states", action="store_true",
                        help="Use raw MuJoCo states instead of observations")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--goal_timestep", type=int, default=-1,
                        help="Goal timestep (-1 = last)")

    args = parser.parse_args()

    precompute_rewards(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        obs_keys=args.obs_keys,
        use_states=args.use_states,
        device=args.device,
        goal_timestep=args.goal_timestep,
    )


if __name__ == "__main__":
    main()
