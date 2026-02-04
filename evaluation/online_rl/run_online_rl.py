#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Online RL training with VIP rewards using MJRL (NPG/PPO).

This script trains a policy to reach goal states using VIP embedding
distances as the reward signal. It reuses the existing MJRL training
infrastructure (train_agent, samplers, algorithms).

Usage:
    python run_online_rl.py checkpoint_path=/path/to/model.pt

    # Override parameters:
    python run_online_rl.py checkpoint_path=/path/to/model.pt algorithm=PPO

    # Use different config:
    python run_online_rl.py --config-name online_rl_state_iql checkpoint_path=/path/to/iql.pt
"""

import os
import sys
import time as timer
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_rl.vip_reward_env import VIPRewardEnv

# Import MJRL components
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.batch_reinforce import BatchREINFORCE
from mjrl.algos.ppo_clip import PPO
from mjrl.samplers.core import sample_paths
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.utils.gym_env import GymEnv
import copy
import pickle
from tabulate import tabulate


class MJRLEnvWrapper(GymEnv):
    """
    Wrapper to make VIPRewardEnv compatible with MJRL's GymEnv interface.

    Inherits from MJRL's GymEnv so it passes isinstance checks in samplers.
    """

    def __init__(self, vip_env):
        # Don't call super().__init__ - we'll set up attributes directly
        self.env = vip_env
        self.env_id = vip_env.spec.id
        self.act_repeat = 1

        # Get dimensions
        self._observation_dim = vip_env.observation_space.shape[0]
        self._action_dim = vip_env.action_space.shape[0]
        self._horizon = vip_env._max_episode_steps

        # Create spec object matching MJRL's EnvSpec
        self.spec = type('EnvSpec', (), {
            'observation_dim': self._observation_dim,
            'action_dim': self._action_dim,
            'horizon': self._horizon,
        })()

        # Obs mask (required by GymEnv)
        self.obs_mask = np.ones(self._observation_dim)

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, done, info = self.env.step(action)
        # Return empty info dict to avoid MJRL stacking issues with nested gymnasium info
        return self.obs_mask * obs, reward, done, {}

    def get_obs(self):
        return self.obs_mask * self.env.get_obs()

    def render(self):
        return self.env.render()

    def set_seed(self, seed=123):
        self.env.seed(seed)

    def get_env_infos(self):
        return {}

    def get_env_state(self):
        return self.env.get_env_state()

    def set_env_state(self, state_dict):
        return self.env.set_env_state(state_dict)


def setup_agent(cfg, env):
    """Set up MJRL agent (NPG, PPO, or VPG) using existing MJRL components."""
    # Create policy (reuses mjrl.policies.gaussian_mlp.MLP)
    policy = MLP(
        env.spec,
        hidden_sizes=tuple(cfg.policy_hidden_sizes),
        seed=cfg.seed,
        init_log_std=cfg.init_log_std
    )

    # Create baseline (reuses mjrl.baselines.mlp_baseline.MLPBaseline)
    baseline = MLPBaseline(
        env.spec,
        reg_coef=1e-3,
        batch_size=cfg.vf_batch_size,
        hidden_sizes=tuple(cfg.vf_hidden_sizes),
        epochs=cfg.vf_epochs,
        learn_rate=cfg.vf_learn_rate
    )

    # Create agent (reuses mjrl.algos)
    alg_kwargs = OmegaConf.to_container(cfg.get('alg_kwargs', {}), resolve=True)

    if cfg.algorithm == 'NPG':
        agent = NPG(
            env, policy, baseline,
            normalized_step_size=cfg.step_size,
            seed=cfg.seed,
            save_logs=True,
            **alg_kwargs
        )
    elif cfg.algorithm == 'VPG':
        agent = BatchREINFORCE(
            env, policy, baseline,
            learn_rate=cfg.step_size,
            seed=cfg.seed,
            save_logs=True,
            **alg_kwargs
        )
    elif cfg.algorithm == 'PPO':
        agent = PPO(
            env, policy, baseline,
            seed=cfg.seed,
            save_logs=True,
            **alg_kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

    return agent


@hydra.main(config_path="config", config_name="online_rl_state_vip")
def main(cfg: DictConfig):
    """Main entry point."""
    print("=" * 60)
    print("Online RL Training with VIP Rewards")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Create VIP reward environment
    env = VIPRewardEnv(
        env_name=cfg.env_name,
        checkpoint_path=cfg.checkpoint_path,
        model_type=cfg.model_type,
        goal_episode=cfg.goal_episode,
        goal_timestep=cfg.goal_timestep,
        init_timestep=cfg.init_timestep,
        device=cfg.device,
        max_episode_steps=cfg.max_episode_steps,
    )

    # Wrap for MJRL compatibility
    env = MJRLEnvWrapper(env)

    # Set up agent using existing MJRL components
    agent = setup_agent(cfg, env)

    # Get output directory (Hydra sets this automatically)
    output_dir = os.getcwd()

    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    print("=" * 60)
    print(f"Algorithm: {cfg.algorithm}")
    print(f"Environment: {cfg.env_name}")
    print(f"Model: {cfg.model_type}")
    print(f"Checkpoint: {cfg.checkpoint_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Run training loop (custom loop that reuses our VIP reward environment)
    ts = timer.time()
    np.random.seed(cfg.seed)

    # Create output directories
    os.makedirs('iterations', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    best_policy = copy.deepcopy(agent.policy)
    best_perf = -1e8
    train_curve = np.full(cfg.num_iter, -1e8)

    for i in range(cfg.num_iter):
        print("." * 60)
        print(f"ITERATION: {i}")

        # Track best policy
        if i > 0 and train_curve[i-1] > best_perf:
            best_policy = copy.deepcopy(agent.policy)
            best_perf = train_curve[i-1]

        # Training step using MJRL's train_step
        # Pass env explicitly to avoid it using env_id string
        N = cfg.num_traj if cfg.sample_mode == 'trajectories' else cfg.num_samples
        stats = agent.train_step(
            N=N,
            env=env,  # Pass our wrapped env directly
            sample_mode=cfg.sample_mode,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            num_cpu=cfg.num_cpu
        )
        train_curve[i] = stats[0]

        # Evaluation rollouts (reuse same env)
        mean_pol_perf = 0.0
        if cfg.eval_rollouts > 0:
            eval_paths = sample_paths(
                num_traj=cfg.eval_rollouts,
                env=env,  # Reuse our VIP reward env
                policy=agent.policy,
                eval_mode=True,
                base_seed=cfg.seed + i * 1000
            )
            mean_pol_perf = np.mean([np.sum(p['rewards']) for p in eval_paths])
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_pol_perf)

        # Save checkpoints
        if i % cfg.save_freq == 0 and i > 0:
            if agent.save_logs:
                agent.logger.save_log('logs/')
                make_train_plots(log=agent.logger.log, keys=['stoc_pol_mean'], save_loc='logs/')
            pickle.dump(agent.policy, open(f'iterations/policy_{i}.pickle', 'wb'))
            pickle.dump(agent.baseline, open(f'iterations/baseline_{i}.pickle', 'wb'))
            pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))

        # Print progress
        print(f"[{timer.asctime()}] Iter {i:4d} | Train: {train_curve[i]:8.2f} | Eval: {mean_pol_perf:8.2f} | Best: {best_perf:8.2f}")
        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1, agent.logger.get_current_log().items()))
            print(tabulate(print_data))

    # Final save
    pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
    if agent.save_logs:
        agent.logger.save_log('logs/')
        make_train_plots(log=agent.logger.log, keys=['stoc_pol_mean'], save_loc='logs/')

    print(f"\nTotal time: {timer.time() - ts:.1f}s")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
