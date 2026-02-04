# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
VIP Reward Environment Wrapper for MJRL Online RL.

This wrapper creates a gym-compatible environment that:
1. Uses the FrankaKitchen environment from Minari/gymnasium
2. Computes rewards using VIP embedding distance to goal
3. Provides interface compatible with MJRL's GymEnv wrapper
"""

import numpy as np
import torch
import gym
from gym import spaces
import minari


def extract_achieved_goal(obs):
    """Extract 11D achieved_goal from 59D observation."""
    return np.concatenate([
        obs[32:39],  # kettle (7D)
        obs[26:28],  # light switch (2D)
        obs[31:32],  # microwave (1D)
        obs[28:29],  # slide cabinet (1D)
    ])


class VIPRewardEnv(gym.Env):
    """
    Gym environment that wraps FrankaKitchen and uses VIP embeddings for reward.

    The reward is computed as negative L2 distance between current state embedding
    and goal state embedding: reward = -||φ(s) - φ(g)||_2

    This is compatible with MJRL's GymEnv wrapper.
    """

    def __init__(
        self,
        env_name="D4RL/kitchen/complete-v2",
        checkpoint_path=None,
        model_type="state_vip",
        goal_episode=0,
        goal_timestep=-1,
        init_timestep=0,
        device="cuda",
        use_achieved_goal=False,
        max_episode_steps=280,
    ):
        """
        Args:
            env_name: Minari dataset name (e.g., "D4RL/kitchen/complete-v2")
            checkpoint_path: Path to VIP model checkpoint
            model_type: One of "state_vip", "state_euclidean_derail",
                       "state_multilinear_derail", "state_iql"
            goal_episode: Which episode to sample goal from
            goal_timestep: Which timestep in episode for goal (-1 = last)
            init_timestep: Which timestep to initialize from
            device: torch device
            use_achieved_goal: Whether to use achieved_goal (11D) instead of full obs (59D)
            max_episode_steps: Maximum steps per episode
        """
        super().__init__()

        self.env_name = env_name
        self.device = device
        self.use_achieved_goal = use_achieved_goal
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        # Load Minari dataset and recover environment
        self.minari_dataset = minari.load_dataset(env_name, download=True)
        self._env = self.minari_dataset.recover_environment(
            render_mode=None,
            robot_noise_ratio=0.0,
            object_noise_ratio=0.0
        )

        # Set up spaces (59D observation, 9D action)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(59,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float64
        )

        # For MJRL compatibility
        self.spec = gym.envs.registration.EnvSpec(
            id=f"VIPReward-{env_name.replace('/', '-')}-v0",
            max_episode_steps=max_episode_steps
        )

        # Load VIP model
        self.model = None
        self.model_type = model_type
        self.multilinear = False
        self.is_iql = False

        if checkpoint_path:
            self._load_model(checkpoint_path, model_type)

        # Set up goal and init states from dataset
        self.goal_state = None
        self.goal_embedding = None
        self.init_state = None
        self._setup_goal_init(goal_episode, goal_timestep, init_timestep)

    def _load_model(self, checkpoint_path, model_type):
        """Load the VIP model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_cfg = {k: v for k, v in checkpoint['model_cfg'].items() if not k.startswith('_')}

        # Check if model was trained with achieved_goal
        self.use_achieved_goal = checkpoint.get('use_achieved_goal', self.use_achieved_goal)

        if model_type == "state_vip":
            from vip.models.model_vip import StateVIP
            self.model = StateVIP(**model_cfg)
        elif model_type == "state_euclidean_derail":
            from vip.models.model_derail import StateEuclideanDERAIL
            self.model = StateEuclideanDERAIL(**model_cfg)
        elif model_type == "state_multilinear_derail":
            from vip.models.model_derail import StateMultilinearDERAIL
            self.model = StateMultilinearDERAIL(**model_cfg)
            self.multilinear = True
        elif model_type == "state_iql":
            from vip.models.model_iql import StateIQL
            self.model = StateIQL(**model_cfg)
            self.is_iql = True
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.load_state_dict({
            k.replace('module.', ''): v
            for k, v in checkpoint['model'].items()
        })
        self.model.eval()
        self.model.to(self.device)

        print(f"Loaded {model_type} from {checkpoint_path}")
        if self.use_achieved_goal:
            print("  Using achieved_goal (11D) for embeddings")

    def _setup_goal_init(self, goal_episode, goal_timestep, init_timestep):
        """Set up goal and initial states from Minari dataset."""
        episodes = list(self.minari_dataset.iterate_episodes())

        if goal_episode >= len(episodes):
            goal_episode = 0
            print(f"Warning: goal_episode exceeds dataset size, using episode 0")

        episode = episodes[goal_episode]
        obs = episode.observations['observation']  # (T+1, 59)
        ep_length = len(obs)

        # Resolve timestep indices
        if goal_timestep < 0:
            actual_goal_idx = ep_length + goal_timestep
        else:
            actual_goal_idx = min(goal_timestep, ep_length - 1)

        if init_timestep < 0:
            actual_init_idx = ep_length + init_timestep
        else:
            actual_init_idx = min(init_timestep, ep_length - 1)

        print(f"Episode {goal_episode}: {ep_length} timesteps")
        print(f"  Goal: timestep {goal_timestep} -> index {actual_goal_idx}")
        print(f"  Init: timestep {init_timestep} -> index {actual_init_idx}")

        # Store goal state
        self.goal_state = obs[actual_goal_idx].astype(np.float32)

        # Compute goal embedding
        if self.model is not None:
            goal_for_embed = self.goal_state
            if self.use_achieved_goal:
                goal_for_embed = extract_achieved_goal(self.goal_state)

            with torch.no_grad():
                goal_tensor = torch.tensor(goal_for_embed, dtype=torch.float32).unsqueeze(0).to(self.device)
                if self.multilinear:
                    self.goal_embedding = self.model.encoder1(goal_tensor)
                elif self.is_iql:
                    # IQL uses goal directly for concatenation
                    self.goal_embedding = goal_tensor
                else:
                    self.goal_embedding = self.model(goal_tensor)

        # Store init state for reset
        robot_qpos = obs[actual_init_idx][:9]
        robot_qvel = obs[actual_init_idx][9:18]
        obj_qpos = obs[actual_init_idx][18:39]
        obj_qvel = obs[actual_init_idx][39:]

        self.init_state = {
            "qpos": np.concatenate([robot_qpos, obj_qpos]),
            "qvel": np.concatenate([robot_qvel, obj_qvel])
        }

    def _compute_reward(self, obs):
        """Compute VIP embedding reward."""
        if self.model is None:
            # Fall back to environment reward if no model
            return 0.0

        obs_for_embed = obs
        if self.use_achieved_goal:
            obs_for_embed = extract_achieved_goal(obs)

        with torch.no_grad():
            state_tensor = torch.tensor(obs_for_embed, dtype=torch.float32).unsqueeze(0).to(self.device)

            if self.multilinear:
                # V(s, g) = φ1(s) @ M(g) @ φ3(g)
                goal_for_embed = self.goal_state
                if self.use_achieved_goal:
                    goal_for_embed = extract_achieved_goal(self.goal_state)
                goal_tensor = torch.tensor(goal_for_embed, dtype=torch.float32).unsqueeze(0).to(self.device)
                h1, h2, h3 = self.model(state_tensor, goal_tensor)
                reward = self.model.value(h1, h2, h3).cpu().numpy().squeeze()
            elif self.is_iql:
                # V(s || g)
                goal_for_embed = self.goal_state
                if self.use_achieved_goal:
                    goal_for_embed = extract_achieved_goal(self.goal_state)
                goal_tensor = torch.tensor(goal_for_embed, dtype=torch.float32).unsqueeze(0).to(self.device)
                sg = torch.cat([state_tensor, goal_tensor], dim=-1)
                reward = self.model.v_value(sg).cpu().numpy().squeeze()
            else:
                # VIP/Euclidean DERAIL: -||φ(s) - φ(g)||_2
                state_embedding = self.model(state_tensor)
                reward = -torch.norm(state_embedding - self.goal_embedding, dim=-1).cpu().numpy().squeeze()

        return float(reward)

    def reset(self, seed=None):
        """Reset environment to initial state."""
        if seed is not None:
            self._env.reset(seed=seed)
        else:
            self._env.reset()

        self._elapsed_steps = 0

        # Set to init state from dataset
        if self.init_state is not None:
            self._env.unwrapped.robot_env.set_state(
                self.init_state["qpos"],
                self.init_state["qvel"]
            )

        # Get observation
        robot_obs = self._env.unwrapped.robot_env._get_obs()
        obs_dict = self._env.unwrapped._get_obs(robot_obs)
        obs = obs_dict["observation"].astype(np.float64)

        return obs

    def step(self, action):
        """Take a step in the environment."""
        action = np.clip(action, self.action_space.low, self.action_space.high)

        obs_dict, _, terminated, truncated, info = self._env.step(action)
        obs = obs_dict["observation"].astype(np.float64)

        # Compute VIP reward
        reward = self._compute_reward(obs)

        self._elapsed_steps += 1
        done = terminated or truncated or (self._elapsed_steps >= self._max_episode_steps)

        return obs, reward, done, info

    def get_obs(self):
        """Get current observation (for MJRL compatibility)."""
        robot_obs = self._env.unwrapped.robot_env._get_obs()
        obs_dict = self._env.unwrapped._get_obs(robot_obs)
        return obs_dict["observation"].astype(np.float64)

    def render(self, mode='human'):
        """Render the environment."""
        return self._env.render()

    def close(self):
        """Close the environment."""
        self._env.close()

    def seed(self, seed=None):
        """Set random seed (for MJRL compatibility)."""
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    # For MJRL trajectory optimization compatibility
    def get_env_state(self):
        """Get environment state for MJRL."""
        return {
            'qpos': self._env.unwrapped.data.qpos.copy(),
            'qvel': self._env.unwrapped.data.qvel.copy(),
        }

    def set_env_state(self, state_dict):
        """Set environment state for MJRL."""
        self._env.unwrapped.robot_env.set_state(
            state_dict['qpos'],
            state_dict['qvel']
        )


def make_vip_env(
    env_name="D4RL/kitchen/complete-v2",
    checkpoint_path=None,
    model_type="state_vip",
    goal_episode=0,
    goal_timestep=-1,
    init_timestep=0,
    device="cuda",
    **kwargs
):
    """Factory function to create VIP reward environment."""
    return VIPRewardEnv(
        env_name=env_name,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        goal_episode=goal_episode,
        goal_timestep=goal_timestep,
        init_timestep=init_timestep,
        device=device,
        **kwargs
    )
